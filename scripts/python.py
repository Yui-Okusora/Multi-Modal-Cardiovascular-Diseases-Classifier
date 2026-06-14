import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Core Framework Dependencies
from src.TimelineDataset import BVTDTimelineDataset
from src.TJEPA import VitalsContextEncoder, PhoBERTTargetEncoder, PredictorBridge, MultimodalDownstreamClassifier
from scripts.config import CardioConfig

class JEPAClinicalTrainer:
    def __init__(self, config, models_dict, train_loader, val_loader=None):
        self.cfg = config
        self.loaders = {'train': train_loader, 'val': val_loader}
        
        # Unpack modular model registry onto execution target device
        self.context_enc = models_dict['context_encoder'].to(self.cfg.device)
        self.target_enc = models_dict['target_encoder'].to(self.cfg.device)
        self.predictor = models_dict['predictor_bridge'].to(self.cfg.device)
        self.clf_head = models_dict['classifier_head'].to(self.cfg.device)
        
        # Objective Functions
        self.alignment_criterion = nn.MSELoss()
        self.classification_criterion = nn.CrossEntropyLoss()

    def run_alignment_epoch(self, optimizer):
        self.context_enc.train()
        self.predictor.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.loaders['train']):
            batch_start_time = time.time()
            optimizer.zero_grad()
            
            vitals = batch['vitals_12d'].to(self.cfg.device)
            ids = batch['input_ids'].to(self.cfg.device)
            mask = batch['attention_mask'].to(self.cfg.device)
            batch_size = vitals.size(0)
            
            # Forward coordinate transformation mapping
            latent_context = self.context_enc(vitals)
            with torch.no_grad():
                true_target = self.target_enc(ids, mask)
                
            predicted_target = self.predictor(latent_context)
            loss = self.alignment_criterion(predicted_target, true_target)
            
            loss.backward()
            
            # Telemetry Tracking: Compute Raw Gradient Norm before clipping boundary walls
            raw_grad_norm = 0.0
            trainable_params = list(self.context_enc.parameters()) + list(self.predictor.parameters())
            for p in trainable_params:
                if p.grad is not None:
                    raw_grad_norm += p.grad.detach().data.norm(2).item() ** 2
            raw_grad_norm = raw_grad_norm ** 0.5
            
            # Prevent optimization spikes via hard constraint wall
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.cfg.grad_clip_norm)
            
            optimizer.step()
            total_loss += loss.item()
            
            # Interval Logging Telemetry Meter
            if batch_idx % self.cfg.log_interval == 0 or batch_idx == len(self.loaders['train']) - 1:
                duration = time.time() - batch_start_time
                throughput = batch_size / duration if duration > 0 else 0
                vram = torch.cuda.memory_allocated(self.cfg.device) / (1024 ** 2) if torch.cuda.is_available() else 0
                
                print(f"Batch: [{batch_idx:3d}/{len(self.loaders['train'])}] | Loss: {loss.item():.4f} | "
                      f"Grad Norm: {raw_grad_norm:5.2f} | Speed: {throughput:5.1f} smpl/s | VRAM: {vram:.0f} MB")
                      
        return total_loss / len(self.loaders['train'])

    def fit(self, mode="alignment"):
        if mode == "alignment":
            print("\n--- Running Phase 1: Pre-training T-JEPA Real Alignment ---")
            trainable_params = list(self.context_enc.parameters()) + list(self.predictor.parameters())

            # Inject Weight Decay directly into AdamW
            optimizer = torch.optim.AdamW(
                trainable_params, 
                lr=self.cfg.pretrain_lr, 
                weight_decay=self.cfg.weight_decay
            )
            
            # Cosine Annealing drops the learning rate smoothly as alignment finishes
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.cfg.pretrain_epochs
            )
            
            for epoch in range(self.cfg.pretrain_epochs):
                epoch_start_time = time.time()
                print(f"\n🚀 Starting Alignment Epoch {epoch+1}/{self.cfg.pretrain_epochs}")
                print("-" * 90)
                avg_loss = self.run_alignment_epoch(optimizer)
                scheduler.step()
                duration = time.time() - epoch_start_time
                print("-" * 90)
                print(f"✅ Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.5f} | Time: {duration:.1f}s")
                print("-" * 90)
                
        elif mode == "downstream":
            print("\n--- Running Phase 3: Downstream Linear Probe Classification ---")
            self.context_enc.eval()  # Freeze backbone features
            
            optimizer = torch.optim.AdamW(
                self.clf_head.parameters(), 
                lr=self.cfg.downstream_lr, 
                weight_decay=self.cfg.weight_decay
            )
            
            for epoch in range(self.cfg.downstream_epochs):
                self.clf_head.train()
                total_loss = 0.0
                
                for batch in self.loaders['train']:
                    optimizer.zero_grad()
                    vitals = batch['vitals_12d'].to(self.cfg.device)
                    ids = batch['input_ids'].to(self.cfg.device)
                    mask = batch['attention_mask'].to(self.cfg.device)
                    labels = batch['maicd_label_id'].to(self.cfg.device)
                    
                    with torch.no_grad():
                        v_feat = self.context_enc(vitals)
                        t_feat = self.target_enc(ids, mask)
                        
                    logits = self.clf_head(v_feat, t_feat)
                    loss = self.classification_criterion(logits, labels)
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    
                avg_loss = total_loss / len(self.loaders['train'])
                print(f"Probe Tuning Optimization Step {epoch+1}/{self.cfg.downstream_epochs} | CrossEntropy Loss: {avg_loss:.5f}")

    def harvest_and_serialize(self, dataloader, output_csv_name):
        """Passes an uncompromised dataloader through frozen weights to extract representations."""
        print(f"Harvesting clinical representations for mapping matrix to storage target: {output_csv_name}")
        self.context_enc.eval()
        self.target_enc.eval()
        harvested_matrix = []
        
        with torch.no_grad():
            for batch in dataloader:
                vitals = batch['vitals_12d'].to(self.cfg.device)
                ids = batch['input_ids'].to(self.cfg.device)
                mask = batch['attention_mask'].to(self.cfg.device)
                target_ids = batch['maicd_label_id'].cpu().numpy()
                
                v_feat = self.context_enc(vitals).cpu().numpy()
                t_feat = self.target_enc(ids, mask).cpu().numpy()
                combined_features = np.concatenate([v_feat, t_feat], axis=1)
                
                for i in range(len(combined_features)):
                    matrix_row = np.append(combined_features[i], target_ids[i])
                    harvested_matrix.append(matrix_row)
                    
        df_features = pd.DataFrame(harvested_matrix)
        df_features.to_csv(output_csv_name, index=False)
        print(f"✅ Successfully exported {len(df_features)} profiles to '{output_csv_name}'.")

    def save_checkpoint(self, filename="jepa_checkpoint.pt"):
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        checkpoint = {
            'context_encoder': self.context_enc.state_dict(),
            'predictor_bridge': self.predictor.state_dict(),
            'classifier_head': self.clf_head.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.cfg.checkpoint_dir, filename))
        print(f"Serialized structural training checkpoints to storage engine.")


def print_model_telemetry(model_source, framework_name="Clinical Engine"):
    """
    Computes exact parameter scale states and static VRAM/RAM footprints.
    Accepts a standalone nn.Module or a structured dictionary of nn.Modules.
    """
    def _compute_footprint(module):
        # Count elements
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # Compute exact bytes (Parameters size + State Buffers size)
        param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers())
        total_bytes = param_bytes + buffer_bytes
        
        megabytes = total_bytes / (1024 ** 2)
        return total_params, trainable_params, megabytes

    print("\n" + "=" * 70)
    print(f" 🛠️  HARDWARE TELEMETRY LOGS: {framework_name.upper()} ")
    print("=" * 70)

    if isinstance(model_source, dict):
        grand_total, grand_train, grand_mb = 0, 0, 0.0
        
        for module_name, sub_module in model_source.items():
            if isinstance(sub_module, nn.Module):
                tot, trn, mb = _compute_footprint(sub_module)
                print(f"Sub-module [{module_name:<20}]: {tot:11,} params | "
                      f"{trn:11,} trainable | {mb:7.2f} MB")
                grand_total += tot
                grand_train += trn
                grand_mb += mb
                
        print("-" * 70)
        print(f"SYSTEM ACCUMULATED PARAMETERS : {grand_total:,}")
        print(f"SYSTEM ACTIVE TRAINABLE PATHS : {grand_train:,}")
        print(f"ESTIMATED STATIC MEMORY SIZE  : {grand_mb:.2f} MB ({grand_mb / 1024:.3f} GB)")
        
    elif isinstance(model_source, nn.Module):
        tot, trn, mb = _compute_footprint(model_source)
        print(f"Total Parameter Count        : {tot:,}")
        print(f"Active Trainable Parameters  : {trn:,}")
        print(f"Frozen Layer Parameters      : {tot - trn:,}")
        print(f"Estimated Static Footprint   : {mb:.2f} MB ({mb / 1024:.3f} GB)")
        
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Initialize shared configuration profile
    cfg = CardioConfig()
    print(f"Executing pipeline on target core device engine: {cfg.device}")

    # =====================================================================
    # STEP 1: UNCOMPROMISED PATIENT-LEVEL RAIL SPLITTING
    # =====================================================================
    print("Loading extracted clinical master databases...")
    cdha_df = pd.read_csv(cfg.cdha_path, dtype=str)
    xn_df = pd.read_csv(cfg.xn_path, dtype=str)

    all_patients = list(cdha_df['mabn'].unique())
    np.random.seed(42)  
    np.random.shuffle(all_patients)
    split_idx = int(len(all_patients) * 0.8)
    
    train_mabns = set(all_patients[:split_idx])
    val_mabns = set(all_patients[split_idx:])

    train_cdha = cdha_df[cdha_df['mabn'].isin(train_mabns)].copy()
    train_xn = xn_df[xn_df['mabn'].isin(train_mabns)].copy()
    
    val_cdha = cdha_df[cdha_df['mabn'].isin(val_mabns)].copy()
    val_xn = xn_df[xn_df['mabn'].isin(val_mabns)].copy()

    print(f"-> Setup complete. Training Cohort: {len(train_mabns)} patients | Validation Cohort: {len(val_mabns)} patients.")

    # Convert alphanumeric target strings safely into isolated matrix IDs
    train_cdha['maicd'] = train_cdha['maicd'].fillna("UNKNOWN_CODE").astype(str).str.strip()
    train_cdha['maicd_label_id'] = train_cdha['maicd'].astype('category').cat.codes
    
    icd_codebook = dict(zip(train_cdha['maicd'], train_cdha['maicd_label_id']))
    num_classes = train_cdha['maicd_label_id'].nunique()
    
    val_cdha['maicd'] = val_cdha['maicd'].fillna("UNKNOWN_CODE").astype(str).str.strip()
    val_cdha['maicd_label_id'] = val_cdha['maicd'].map(icd_codebook).fillna(-1).astype(int)
    val_cdha = val_cdha[val_cdha['maicd_label_id'] != -1]  # Strip classes missing from training distribution

    # =====================================================================
    # STEP 2: PYTORCH INFRASTRUCTURE POOLING
    # =====================================================================
    print("Initializing tokenization engine...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    
    train_dataset = BVTDTimelineDataset(train_cdha, train_xn, tokenizer, max_len=cfg.max_sequence_len)
    val_dataset = BVTDTimelineDataset(val_cdha, val_xn, tokenizer, max_len=cfg.max_sequence_len)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # =====================================================================
    # STEP 3: OBJECT INTERACTIVE EXECUTION LIFE-CYCLE
    # =====================================================================
    # Instantiate structural networks
    models = {
        'context_encoder': VitalsContextEncoder(input_dim=cfg.vitals_dim, latent_dim=cfg.latent_dim),
        'target_encoder': PhoBERTTargetEncoder(model_name=cfg.text_model_name),
        'predictor_bridge': PredictorBridge(latent_dim=cfg.latent_dim, text_dim=cfg.text_dim),
        'classifier_head': MultimodalDownstreamClassifier(num_classes=num_classes)
    }

    print_model_telemetry(models, framework_name="T-JEPA System Array")

    # Bind elements cleanly inside the framework controller
    trainer = JEPAClinicalTrainer(cfg, models, train_loader, val_loader)

    # RUN PHASE 1: Coordinate Latent Space Mapping Optimization
    trainer.fit(mode="alignment")
    trainer.save_checkpoint("vitals_context_encoder.pt")

    # RUN PHASE 2: Uncompromised Representation Harvesting Passes
    print("\n--- Running Phase 2: Feature Matrix Sequential Harvesting ---")
    trainer.harvest_and_serialize(train_loader, "train_features.csv")
    trainer.harvest_and_serialize(val_loader, "val_features.csv")

    # RUN PHASE 3: Linear Probe Execution Loop Tracking
    trainer.fit(mode="downstream")
    trainer.save_checkpoint("vitals_context_encoder.pt")
    print("\n🚀 Framework pipeline completed structural execution safely.")