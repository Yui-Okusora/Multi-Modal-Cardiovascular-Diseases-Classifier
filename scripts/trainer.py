# trainer.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.TimelineDataset import BVTDFlattenedDataset  
from src.ModelModules import *
from src.BaseEngine import *
from config import CardioConfig
from Pipeline import *

import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class VICRegProjector(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        if x.dim() == 3:
            B, K, D = x.size()
            return self.net(x.contiguous().view(B * K, D)).view(B, K, -1)
        return self.net(x)

class DualPhaseTrainingEngine(BaseExecutionEngine):
    def __init__(self, cfg: CardioConfig):
        super().__init__(cfg)
        with open(cfg.codebook_json_path, "r", encoding="utf-8") as f:
            self.meta = __import__('json').load(f)["metadata"]

        self.cfg = cfg
        self.pipeline = ClinicalPipeline(cfg, self.device)

        self.context_encoder = self.pipeline.context_encoder
        self.predictor = self.pipeline.predictor
        self.target_encoder = self.pipeline.target_encoder
        
        self.context_projector = VICRegProjector(in_dim=cfg.latent_dim).to(self.device)
        self.target_projector = VICRegProjector(in_dim=cfg.latent_dim).to(self.device)
        self.linear_probe = self.pipeline.linear_probe

        # 🏥 VALIDATION LIFECYCLE TRACKER HOOK
        self.val_loader = DataLoader(
            BVTDFlattenedDataset(cfg.val_csv_path, max_seq_len=cfg.max_sequence_len, max_targets=cfg.max_targets), 
            batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

    def execute_validation_pass(self, phase="Pretraining"):
        """🛡️ READ-ONLY SUB-GRAPH COHORT EVALUATION MATRIX EXTRACTION"""
        self.context_encoder.eval()
        self.predictor.eval()
        self.context_projector.eval()
        self.target_projector.eval()
        if phase == "Probe":
            self.linear_probe.eval()

        all_probs, all_targets = [], []
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                out = self.pipeline.process_batch(batch, self.device, run_teacher=(phase == "Pretraining"))
                
                if phase == "Pretraining":
                    p_c = self.context_projector(out['z_hat_slots'])
                    p_t = self.target_projector(out['z_t']).detach()
                    
                    loss_align = F.smooth_l1_loss(p_c, p_t, beta=0.5)
                    loss_var = self.compute_variance_loss(p_c) + self.compute_variance_loss(p_t)
                    loss_cov = self.compute_covariance_loss(p_c) + self.compute_covariance_loss(p_t)
                    loss_slot_diversity = self.compute_cross_slot_orthogonal_loss(p_c)
                    
                    val_loss = (self.cfg.alpha_align * loss_align + 
                                self.cfg.alpha_var * loss_var + 
                                self.cfg.alpha_cov * loss_cov + 
                                self.cfg.alpha_diverse * loss_slot_diversity)
                    total_val_loss += val_loss.item()
                else:
                    logits = self.linear_probe(out['z_hat_slots'])
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.append(probs)
                    all_targets.append(out['multi_hot_targets'].cpu().numpy())

        if phase == "Pretraining":
            mean_loss = total_val_loss / len(self.val_loader)
            print(f"🔍 [VALIDATION AUDIT] Phase 1 │ Unsupervised Representation Loss: {mean_loss:.4f}")
            return mean_loss
        else:
            probs_matrix = np.concatenate(all_probs, axis=0)
            targets_matrix = np.concatenate(all_targets, axis=0)
            
            # 🚀 CLEAN REUSE: Call the audit engine silently to fetch scores for tracking
            audit_data = execute_clinical_audit(
                targets_matrix, probs_matrix, 
                calibrate_per_class=False, silent=True
            )
            
            macro_auc_roc = audit_data["macro_auc_roc"]
            print(f"🔍 [VALIDATION AUDIT] Phase 2 │ Patient Manifold Macro AUC-ROC: {macro_auc_roc:.2f}%")
            return macro_auc_roc

    def run_phase1_pretraining(self, train_loader):
        print("\n" + "="*80 + "\n櫨 PHASE 1: OPTIMIZING FOUNDATIONAL PHYSIOLOGICAL WORLD MODEL\n" + "="*80)
        
        phase1_models = [self.context_encoder, self.predictor, self.context_projector, self.target_projector]
        p1_optimizer = torch.optim.AdamW(
            [p for m in phase1_models for p in m.parameters() if p.requires_grad], 
            lr=self.cfg.pretrain_lr, weight_decay=self.cfg.pretrain_wgt_decay
        )

        grad_accum = getattr(self.cfg, "grad_accum_steps", 1)
        total_steps = (len(train_loader) * self.cfg.pretrain_epochs) // grad_accum
        warmup_steps = int(total_steps * 0.10)

        p1_scheduler = self.create_warmup_cosine_scheduler(
            optimizer=p1_optimizer, num_warmup_steps=warmup_steps, num_total_steps=total_steps, min_lr_ratio=0.0
        )

        with torch.no_grad():
            for param_s, param_t in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_t.data.copy_(param_s.data)
                param_t.requires_grad = False

        def absolute_jepa_closure(batch, step, total_steps_val):
            out = self.pipeline.process_batch(batch, self.device, run_teacher=True)
            p_c = self.context_projector(out['z_hat_slots'])
            with torch.no_grad():
                p_t = self.target_projector(out['z_t']).detach()
            
            loss_align = F.smooth_l1_loss(p_c, p_t, beta=0.5)
            loss_var = self.compute_variance_loss(p_c) + self.compute_variance_loss(p_t)
            loss_cov = self.compute_covariance_loss(p_c) + self.compute_covariance_loss(p_t)
            loss_slot_diversity = self.compute_cross_slot_orthogonal_loss(p_c)
            
            return {
                "loss_total_align": [self.cfg.alpha_align, loss_align],
                "loss_variance":    [self.cfg.alpha_var, loss_var],
                "loss_covariance":  [self.cfg.alpha_cov, loss_cov],
                "loss_diversity":   [self.cfg.alpha_diverse, loss_slot_diversity]
            }

        def apply_momentum_teacher_update():
            with torch.no_grad():
                for param_s, param_t in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                    param_t.data = self.cfg.tau * param_t.data + (1.0 - self.cfg.tau) * param_s.data

        # 🤝 MEMORY ALIGNED PERSISTENCE: Tracked snapshot memory
        metrics = self.telemetry.setdefault("Pure-SSL JEPA", {"loss": []})
        early_stop_mem = metrics.setdefault("early_stop_memory", {"best_score": float('inf'), "patience_counter": 0})

        def phase1_epoch_callback(epoch_idx):
            # Compute the unsupervised representation loss across the validation cohort
            val_loss = self.execute_validation_pass(phase="Pretraining")
            
            # Minimize Self-Supervised Loss to track and checkpoint the best model weights
            if val_loss < early_stop_mem["best_score"]:
                early_stop_mem["best_score"] = val_loss
                print(f"🔥 [CHECKPOINT] New optimal representation baseline discovered. Exporting backbone states...")
                self._export_checkpoint({
                    "context_encoder_state": self.context_encoder.state_dict(), 
                    "predictor_state": self.predictor.state_dict()
                }, "best_ssl_backbone.pt")
            else:
                print(f"ℹ️ Epoch {epoch_idx} complete. Current Val Loss: {val_loss:.4f} | Best Historical Loss: {early_stop_mem['best_score']:.4f}")
            
            # 🚀 REMOVED EARLY STOPPING BREAKOUT: Always return False to enforce complete training across full epoch budget
            return False

        self._execute_epoch_loop(
            "Pure-SSL JEPA", phase1_models, p1_optimizer, train_loader, 
            absolute_jepa_closure, self.cfg.pretrain_epochs, p1_scheduler, 
            after_step=apply_momentum_teacher_update, after_epoch=phase1_epoch_callback
        )

        #self.pipeline.load_checkpoint("./checkpoints/best_ssl_backbone.pt", strict=False)

    def run_phase2_probe_fitting(self, train_loader, load_checkpoint_path=None):
        print("\n" + "="*80 + "\n識 PHASE 2: EXECUTING HIGH-VELOCITY REPRODUCIBLE BACKPROPAGATION\n" + "="*80)
        
        if load_checkpoint_path is not None:
            if not os.path.exists(load_checkpoint_path):
                raise FileNotFoundError(f"Checkpoint missing at: {load_checkpoint_path}")
            checkpoint_weights = torch.load(load_checkpoint_path, map_location=self.device)
            self.context_encoder.load_state_dict(checkpoint_weights['context_encoder_state'])
            self.predictor.load_state_dict(checkpoint_weights['predictor_state'])
        
        phase1_models = [self.context_encoder, self.predictor, self.context_projector, self.target_projector]
        for m in phase1_models:
            for param in m.parameters(): param.requires_grad = False
        
        self.linear_probe.train()

        all_train_targets = []
        with torch.no_grad():
            for batch in train_loader:
                out = self.pipeline.process_batch(batch, self.device, run_teacher=False)
                all_train_targets.append(out['multi_hot_targets'].cpu())
        
        compiled_targets = torch.cat(all_train_targets, dim=0)
        pos_counts = compiled_targets.sum(dim=0)
        neg_counts = compiled_targets.size(0) - pos_counts
        pos_weight_vector = torch.sqrt(neg_counts / (pos_counts + 1e-5))
        pos_weight_vector = torch.clamp(pos_weight_vector, min=1.0, max=6.0).to(self.device)

        p2_optimizer = torch.optim.AdamW(self.linear_probe.parameters(), lr=self.cfg.probe_lr, weight_decay=self.cfg.probe_wgt_decay)

        grad_accum = getattr(self.cfg, "grad_accum_steps", 1)
        total_p2_steps = (len(train_loader) * self.cfg.probe_epochs) // grad_accum
        p2_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            p2_optimizer, max_lr=self.cfg.probe_lr, total_steps=total_p2_steps,
            pct_start=0.10, anneal_strategy='cos', div_factor=10.0, final_div_factor=1e4
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_vector)

        def phase2_closure(batch, step, total_steps_val):
            out = self.pipeline.process_batch(batch, self.device, run_teacher=False)
            logits = self.linear_probe(out['z_hat_slots'])
            y = out['multi_hot_targets'].float()
            return criterion(logits, y)

        # 🤝 MEMORY ALIGNED PERSISTENCE: Saved inside the telemetry snapshot automatically
        p2_metrics = self.telemetry.setdefault("ASL Probe-Fitting", {"loss": []})
        p2_early_stop_mem = p2_metrics.setdefault("early_stop_memory", {"best_score": -float('inf'), "patience_counter": 0})

        def phase2_epoch_callback(epoch_idx):
            macro_auc = self.execute_validation_pass(phase="Probe")
            
            # Maximize Macro AUC-ROC Metrics
            if macro_auc > p2_early_stop_mem["best_score"]:
                p2_early_stop_mem["best_score"] = macro_auc
                p2_early_stop_mem["patience_counter"] = 0
                self._export_unified_checkpoint()
            else:
                p2_early_stop_mem["patience_counter"] += 1
                print(f"⚠️ [PATIENCE WARNING] Phase 2 has stalled for {p2_early_stop_mem['patience_counter']}/{self.cfg.patience} epochs.")
                
            return p2_early_stop_mem["patience_counter"] >= self.cfg.patience

        self._execute_epoch_loop(
            "ASL Probe-Fitting", [self.linear_probe], p2_optimizer, 
            train_loader, phase2_closure, num_epochs=self.cfg.probe_epochs, scheduler=p2_scheduler,
            after_epoch=phase2_epoch_callback
        )

    def _export_unified_checkpoint(self):
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, "unified_jepa_and_probe.pt")
        torch.save({
            'context_encoder_state': self.context_encoder.state_dict(),
            'predictor_state': self.predictor.state_dict(),
            'linear_probe_state': self.linear_probe.state_dict()
        }, checkpoint_path)
        print(f"📌 UNIFIED PRODUCTION ARTIFACT SAVED COMPLETELY -> {checkpoint_path}")

if __name__ == "__main__":
    cfg = CardioConfig()
    cfg.train_csv_path = "train_patient_flattened.csv"
    cfg.patience = 7  # Early Stopping Epoch Threshold Bound Constraint
    
    train_loader = DataLoader(
        BVTDFlattenedDataset(cfg.train_csv_path, max_seq_len=cfg.max_sequence_len, max_targets=cfg.max_targets), 
        batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True
    )
    
    engine = DualPhaseTrainingEngine(cfg)
    #engine.run_phase1_pretraining(train_loader)
    engine.run_phase2_probe_fitting(train_loader, "./checkpoints/unified_jepa_and_probe.pt")
    engine.dump_telemetry()