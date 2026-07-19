import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import DataLoader

from src.TimelineDataset import *
from src.ModelModules import *
from src.BaseEngine import *
from src.LoRAWrapper import *
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

# =====================================================================
# 🏥 DUAL PHASE CLINICAL TRAINING ENGINE ASSEMBLY (STANDALONE LORA)
# =====================================================================

class DualPhaseTrainingEngine(BaseExecutionEngine):
    def __init__(self, cfg: CardioConfig):
        super().__init__(cfg)
        with open(cfg.codebook_json_path, "r", encoding="utf-8") as f:
            self.meta = __import__('json').load(f)["metadata"]

        self.cfg = cfg
        self.pipeline = ClinicalPipeline(cfg, self.device)

        # ─── CORE BACKBONE POINTERS ───
        self.context_encoder = self.pipeline.context_encoder
        self.predictor       = self.pipeline.predictor
        self.assembler       = self.pipeline.assembler
        self.target_encoder  = self.pipeline.target_encoder
        
        # Self-Supervised Alignment Projection Space
        self.context_projector = VICRegProjector(in_dim=cfg.latent_dim).to(self.device)
        
        # ─── DEFERRED STANDALONE TASK MODULE POINTERS (PHASE 2) ───
        self.probe_module    = None
        self.cardinal_module = None

        # ─── DATASET TRACKING LOADING CHANNELS ───
        self.val_loader = DataLoader(
            BVTDFlattenedDataset(cfg.val_csv_path, max_seq_len=cfg.max_sequence_len, max_targets=cfg.max_targets), 
            batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True,
            persistent_workers=True, prefetch_factor=2
        )

        self.frequencies = compute_static_class_frequencies(cfg.train_csv_path, num_classes=self.meta['num_icd_classes'])

    @torch.no_grad()
    def audit_phase1_manifold_health(self, z_context: torch.Tensor):
        """ Runs a structural check on the latent representations to monitor stability """
        B, K, D = z_context.shape
        if B <= 1:
            return 1.0, 0.0, float(D)
            
        # 1. Compute average batch standard deviation
        var_per_dim = z_context.var(dim=0)
        mean_batch_std = torch.sqrt(var_per_dim + 1e-6).mean().item()
        
        # 2. Monitor cross-slot similarity correlations
        z_norm = F.normalize(z_context, p=2, dim=-1)
        slot_sim = torch.bmm(z_norm, z_norm.transpose(1, 2))
        triu_indices = torch.triu_indices(K, K, offset=1, device=z_context.device)
        mean_slot_cross_talk = slot_sim[:, triu_indices[0], triu_indices[1]].mean().item()
        
        # 3. Compute structural Effective Rank via float32 SVD checks
        flattened_latents = z_context.contiguous().view(-1, D).float()
        try:
            _, S, _ = torch.svd(flattened_latents)
            singular_energy = S / (S.sum() + 1e-9)
            effective_rank = torch.exp(-torch.sum(singular_energy * torch.log(singular_energy + 1e-9))).item()
        except Exception:
            effective_rank = float('nan')
            
        return mean_batch_std, mean_slot_cross_talk, effective_rank

    def execute_validation_pass(self, phase="Pretraining"):
        self.context_encoder.eval()
        self.assembler.eval()
        
        if hasattr(self, 'predictor') and self.predictor is not None:
            self.predictor.eval()
        if hasattr(self, 'context_projector') and self.context_projector is not None:
            self.context_projector.eval()
        if hasattr(self, 'target_encoder') and self.target_encoder is not None:
            self.target_encoder.eval()

        if phase != "Pretraining" and self.probe_module is not None:
            self.probe_module.eval()
            self.cardinal_module.eval()

        # 🪙 INITIALIZATION: Added missing validation tracker list
        all_probs, all_targets, all_pred_counts = [], [], []
        all_val_z_hat = [] 
        total_val_loss = 0.0

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.cfg.amp_dtype, enabled=self.cfg.use_amp):
                for batch in self.val_loader:
                    out = self.pipeline.process_batch(batch, self.device, run_teacher=(phase == "Pretraining"))
                    
                    if phase == "Pretraining":
                        loss_align = F.smooth_l1_loss(out['z_hat_slots'], out['z_t'].detach(), beta=0.5)
                        p_c = self.context_projector(out['z_c_slots'])
                        val_loss = (self.cfg.alpha_align * loss_align + 
                                    self.cfg.alpha_var * self.compute_variance_loss(p_c) + 
                                    self.cfg.alpha_var * self.compute_variance_loss(out['z_c_slots'], target_std=0.03) + 
                                    self.cfg.alpha_cov * self.compute_covariance_loss(p_c) + 
                                    self.cfg.alpha_diverse * self.compute_cross_slot_orthogonal_loss(out['z_c_slots']))
                        total_val_loss += val_loss.item()
                        
                        # 🧬 POPULATION: Stage validation latents onto CPU host memory safely
                        all_val_z_hat.append(out['z_c_slots'].cpu())
                    else:
                        all_probs.append(torch.sigmoid(out['logits']).float().cpu().numpy())
                        all_targets.append(out['multi_hot_targets'].cpu().numpy())
                        all_pred_counts.append(out['predicted_cardinalities'].float().cpu().numpy())

        if phase == "Pretraining":
            mean_loss = total_val_loss / len(self.val_loader)
            
            # 🧬 Gather intermediate representations from the validation tracking array
            concat_z_hat = torch.cat(all_val_z_hat, dim=0).to(self.device)
            m_std, m_talk, eff_rank = self.audit_phase1_manifold_health(concat_z_hat)
            
            # 🔥 Bring back the live terminal audit display
            print(f"🔍 [VALIDATION AUDIT] Phase 1 │ Loss: {mean_loss:.4f} │ Cohort Std: {m_std:.3f} │ Rank: {eff_rank:.1f}")
            
            return {
                "loss": mean_loss,
                "batch_std": m_std,
                "slot_cross_talk": m_talk,
                "effective_rank": eff_rank
            }
        else:
            return execute_clinical_audit(
                np.concatenate(all_targets, axis=0), 
                np.concatenate(all_probs, axis=0), 
                predicted_cardinalities=np.concatenate(all_pred_counts, axis=0), 
                silent=True
            )

    def run_phase1_pretraining(self, train_loader):
        print("\n" + "="*80 + "\n🧬 PHASE 1: FOUNDATIONAL PHYSIOLOGICAL WORLD MODEL INITIALIZATION\n" + "="*80)
        
        self.context_encoder.train()
        self.predictor.train()
        self.assembler.train()
        self.context_projector.train()
        self.target_encoder.eval()

        phase1_models = [self.context_encoder, self.predictor, self.assembler, self.context_projector]

        decay_params = []
        no_decay_params = []

        for model in phase1_models:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if (
                    "norm" in name.lower() or 
                    "bias" in name.lower() or 
                    "embedding" in name.lower() or
                    "frequencies" in name.lower()
                ):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
                
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.cfg.pretrain_wgt_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        p1_optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.cfg.pretrain_lr)

        total_steps = (len(train_loader) * self.cfg.pretrain_epochs)
        warmup_steps = int(total_steps * 0.20)

        p1_scheduler = self.create_warmup_cosine_scheduler(
            optimizer=p1_optimizer, num_warmup_steps=warmup_steps, num_total_steps=total_steps, min_lr_ratio=0.0
        )

        with torch.no_grad():
            print("🔄 Synchronizing pristine base parameters to Target Encoder state maps...")
            for param_s, param_t in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_t.data.copy_(param_s.data)
                param_t.requires_grad = False
            print("✨ Symmetrical teacher alignment finalized successfully.")

        def get_active_alignment_weight(step, warmup_steps=1000, max_alpha=25.0):
            if step >= warmup_steps:
                return max_alpha
            progress = step / warmup_steps
            cos_multiplier = 0.5 * (1.0 - math.cos(progress * math.pi))
            return max_alpha * cos_multiplier

        def absolute_jepa_closure(batch, step, total_steps_val):
            out = self.pipeline.process_batch(batch, self.device, run_teacher=True)
            
            loss_align = self.compute_alignment_loss(out['z_hat_slots'], out['z_t'].detach(), beta=0.5)
            p_c = self.context_projector(out['z_c_slots'])
            
            loss_var = self.compute_variance_loss(p_c)
            loss_cov = self.compute_covariance_loss(p_c)
            loss_slot_diversity = self.compute_cross_slot_orthogonal_loss(out['z_c_slots'])
            loss_backbone_var = self.compute_variance_loss(out['z_c_slots'], target_std=0.03)
            
            if step % self.cfg.log_interval == 0:
                m_std, m_talk, eff_rank = self.audit_phase1_manifold_health(out['z_c_slots'].detach())
                print(f"   [MANIFOLD HEALTH Step {step}] Batch Std: {m_std:.3f} │ Slot Cross-Talk: {m_talk:.3f} │ Effective Rank: {eff_rank:.1f}")

            active_alpha_align = get_active_alignment_weight(step, max_alpha=self.cfg.alpha_align)

            return {
                "loss_total_align": [active_alpha_align, loss_align],
                "loss_variance":    [self.cfg.alpha_var, loss_var],
                "loss_backbone_v":  [self.cfg.alpha_backbone_v, loss_backbone_var], 
                "loss_covariance":  [self.cfg.alpha_cov, loss_cov],
                "loss_diversity":   [self.cfg.alpha_diverse, loss_slot_diversity]
            }

        def apply_momentum_teacher_update():
            metrics_ref = self.telemetry.get("Pure-SSL JEPA", {})
            curr_step = len(metrics_ref.get("loss", []))
            
            progress = min(max(float(curr_step) / float(max(1, total_steps)), 0.0), 1.0)
            escalated_tau = self.cfg.tau + (0.9999 - self.cfg.tau) * (0.5 * (1.0 - math.cos(math.pi * progress)))
            
            with torch.no_grad():
                for param_s, param_t in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                    param_t.data.copy_(escalated_tau * param_t.data + (1.0 - escalated_tau) * param_s.data)

        metrics = self.telemetry.setdefault("Pure-SSL JEPA", {"loss": []})
        early_stop_mem = metrics.setdefault("early_stop_memory", {"best_score": -float('inf'), "patience_counter": 0})

        def phase1_epoch_callback(epoch_idx):
            audit_dict = self.execute_validation_pass(phase="Pretraining")
            
            self.context_encoder.train()
            self.predictor.train()
            self.context_projector.train()
            self.target_encoder.eval()
            
            m_std = audit_dict["batch_std"]
            m_talk = audit_dict["slot_cross_talk"]
            eff_rank = audit_dict["effective_rank"]
            
            # Recompute original index scaling balance
            manifold_health_score = m_std + (eff_rank / 100.0) - m_talk
            
            # 🖼️ High-order structural health scorecard print
            print(
                f"📊 [MANIFOLD HEALTH SCORECARD Epoch {epoch_idx:02d}] "
                f"Global Index: {manifold_health_score:.4f} │ "
                f"Batch Std: {m_std:.3f} │ "
                f"Slot Cross-Talk: {m_talk:.3f} │ "
                f"Effective Rank: {eff_rank:.1f}"
            )
            
            # 🛡️ EARLY STOP PATIENCE ENGINE REGISTRY
            best_score = early_stop_mem["best_score"]
            
            if manifold_health_score > best_score:
                print(f"🔥 [CHECKPOINT] Higher manifold health index achieved ({best_score:.4f} -> {manifold_health_score:.4f}). Exporting backbone...")
                early_stop_mem["best_score"] = manifold_health_score
                early_stop_mem["patience_counter"] = 0  # Reset patience on breakthrough
                
                self._export_checkpoint({
                    "context_encoder_state": self.context_encoder.state_dict(), 
                    "predictor_state": self.predictor.state_dict()
                }, "best_ssl_backbone.pt")
            else:
                early_stop_mem["patience_counter"] += 1
                print(f"⚠️ [PATIENCE] Phase 1 manifold health has stalled for {early_stop_mem['patience_counter']}/{self.cfg.patience} epochs.")
                
            # Returning True commands BaseEngine to break out of the training loop natively
            if early_stop_mem["patience_counter"] >= self.cfg.patience:
                print(f"🛑 [EARLY STOP BREAKOUT] Phase 1 exhausted all patience limits. Terminating pretraining loop.")
                return True
                
            return False

        self._execute_epoch_loop(
            "Pure-SSL JEPA", phase1_models, p1_optimizer, train_loader, 
            absolute_jepa_closure, self.cfg.pretrain_epochs, p1_scheduler, 
            after_step=apply_momentum_teacher_update, after_epoch=phase1_epoch_callback
        )

        best_checkpoint_filename = os.path.join(self.cfg.checkpoint_dir, "best_ssl_backbone.pt")
        if os.path.exists(best_checkpoint_filename):
            print(f"\n🥇 [PHASE 1 COMPLETE] Automatically reloading maximum physiological health checkpoint parameters:")
            self.pipeline.load_checkpoint(best_checkpoint_filename, strict=False)
        else:
            print(f"\n⚠️ [PHASE 1 WARNING] Could not find saved snapshot. Maintaining terminal weights.")

    def run_phase2_probe_fitting(self, train_loader, load_checkpoint_path=None):
        """⚙️ PHASE 2: STANDALONE PROBE INFRASTRUCTURE TUNING FLOW"""
        print("\n" + "="*80 + "\n⚙️ PHASE 2: INJECTING ADAPTER TRACKS & REPRODUCIBLE BACKPROPAGATION\n" + "="*80)
        
        self.pipeline.inject_phase2_infrastructure()
        self.probe_module = self.pipeline.probe
        self.cardinal_module = self.pipeline.cardinal
        
        if load_checkpoint_path is not None:
            self.pipeline.load_checkpoint(load_checkpoint_path, strict=False)

        self.pipeline.discard_phase1_components()
        self.predictor, self.target_encoder, self.context_projector = None, None, None
        import gc; gc.collect(); torch.cuda.empty_cache()

        self.probe_module.train()
        self.cardinal_module.train()
        self.assembler.train()

        # Isolate parameters: Segment by job type for discriminative learning rates
        backbone_lora_params = [p for p in self.context_encoder.parameters() if p.requires_grad]
        assembler_params     = [p for p in self.assembler.parameters() if p.requires_grad]
        probe_params         = [p for p in self.probe_module.parameters() if p.requires_grad]
        cardinal_params      = [p for p in self.cardinal_module.parameters() if p.requires_grad]

        optimized_parameters = [
            {"params": backbone_lora_params, "lr": self.cfg.probe_lr * 0.02, "weight_decay": self.cfg.probe_wgt_decay * 2.0},
            {"params": assembler_params,     "lr": self.cfg.probe_lr,        "weight_decay": 1e-4},
            {"params": probe_params,         "lr": self.cfg.probe_lr,        "weight_decay": 1e-4},
            {"params": cardinal_params,      "lr": self.cfg.probe_lr,        "weight_decay": 1e-4}
        ]
        p2_optimizer = torch.optim.AdamW(optimized_parameters)
        p2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(p2_optimizer, T_max=(len(train_loader) * self.cfg.probe_epochs), eta_min=1e-7)

        criterion_cls = ClassAwareASLWithLogitAdjustment(
            class_frequencies=self.frequencies, tau=0.5, gamma_pos=0.0, gamma_neg_base=4.5, beta_neg=3.0
        ).to(self.cfg.device)

        criterion_reg = nn.MSELoss()
        lambda_reg = 0.25

        def phase2_closure(batch, step_idx, total_steps):
            """🚀 CLEAN PASSTHROUGH STREAM CLOSURE: PRESERVES FULL CLINICAL TIMELINE"""
            # Pass the pristine batch directly—no more data stripping or timeline masking
            out = self.pipeline.process_batch(batch, self.device, run_teacher=False)
            
            cls_loss = criterion_cls(out['logits'], out['multi_hot_targets'])
            card_loss = criterion_reg(out['predicted_cardinalities'].view(-1), out['true_cardinalities'].view(-1))
            
            return {
                "loss_classification": [1.0, cls_loss], 
                "loss_cardinality_mse": [lambda_reg, card_loss]
            }

        p2_metrics = self.telemetry.setdefault("ASL Probe-Fitting", {"loss": []})
        p2_early_stop_mem = p2_metrics.setdefault("early_stop_memory", {"best_score": -float('inf'), "patience_counter": 0})

        def phase2_epoch_callback(epoch_idx):
            audit_data = self.execute_validation_pass(phase="Probe")
            
            self.probe_module.train()
            self.cardinal_module.train()
            self.assembler.train()

            # ⚖️ Re-injecting the All-Inclusive Weighted Metric Evaluation Index
            composite_score = (
                0.08 * audit_data["macro_auc_roc"] +
                0.08 * audit_data["micro_auc_roc"] +
                0.12 * audit_data["macro_auc_pr"] +
                0.06 * audit_data["macro_f1"] +
                0.07 * audit_data["macro_precision"] +
                0.08 * audit_data["macro_sensitivity"] +
                0.07 * audit_data["macro_specificity"] +
                0.10 * audit_data["adaptive_hit_rate"] +   
                0.10 * audit_data["adaptive_precision"] +  
                0.03 * audit_data["top1_rate"] +
                0.03 * audit_data["top3_rate"] +
                0.03 * audit_data["top5_rate"] +
                0.03 * audit_data["top8_rate"] +
                0.03 * audit_data["precision_at_1"] +
                0.03 * audit_data["precision_at_3"] +
                0.03 * audit_data["precision_at_5"] +
                0.03 * audit_data["precision_at_8"]
            )
            
            best_score = p2_early_stop_mem["best_score"]
            
            # 🖼️ Pristine Old Scorecard Terminal Prints
            print("\n" + "╒" + "═"*78 + "╕")
            print(f" │ 🏥 ALL-INCLUSIVE PHASE 2 VALIDATION SCORECARD (EPOCH {epoch_idx:02d})")
            print(" ├" + "─"*78 + "┤")
            print(f" │ ✨ GLOBAL MULTI-DIMENSIONAL INDEX: {composite_score:.4f}% │ Best Peak: {best_score:.2f}%")
            print(" ├" + "─"*78 + "┤")
            print(f" │ 🩺 [TIER 1] Macro ROC: {audit_data['macro_auc_roc']:6.2f}% │ Micro ROC: {audit_data['micro_auc_roc']:6.2f}% │ PR-AUC: {audit_data['macro_auc_pr']:6.2f}%")
            print(f" │ 🛡️ [TIER 2] Macro F1:  {audit_data['macro_f1']:6.2f}% │ Precision: {audit_data['macro_precision']:6.2f}% │ Sens (Recall):   {audit_data['macro_sensitivity']:6.2f}%")
            print(f" │            Spec (TNR):      {audit_data['macro_specificity']:6.2f}%")
            print(" ├" + "─"*78 + "┤")
            print(f" │ 🚀 [DYNAMIC] Adaptive Hit Rate: {audit_data['adaptive_hit_rate']:5.2f}% │ Adaptive Precision: {audit_data['adaptive_precision']:5.2f}%")
            print(" ├" + "─"*78 + "┤")
            print(f" │ 🛡️ [TIER 3] Presence (Hit Rates)       │ 📈 Density (Precision@K)")
            print(f" │            Top-1 Hit Rate: {audit_data['top1_rate']:6.2f}% │ Precision@1: {audit_data['precision_at_1']:6.2f}%")
            print(f" │            Top-3 Hit Rate: {audit_data['top3_rate']:6.2f}% │ Precision@3: {audit_data['precision_at_3']:6.2f}%")
            print(f" │            Top-5 Hit Rate: {audit_data['top5_rate']:6.2f}% │ Precision@5: {audit_data['precision_at_5']:6.2f}%")
            print(f" │            Top-8 Hit Rate: {audit_data['top8_rate']:6.2f}% │ Precision@8: {audit_data['precision_at_8']:6.2f}%")
            print("╘" + "═"*78 + "╛\n")

            if composite_score > best_score:
                print(f"🔥 [CHECKPOINT] Target maximum surpassed ({best_score:.2f}% -> {composite_score:.2f}%). Saving structures...")
                p2_early_stop_mem["best_score"] = composite_score
                p2_early_stop_mem["patience_counter"] = 0
                self._export_unified_checkpoint(is_final=False)
            else:
                p2_early_stop_mem["patience_counter"] += 1
                print(f"⚠️ [PATIENCE] Phase 2 has stalled for {p2_early_stop_mem['patience_counter']}/{self.cfg.patience} epochs.")
                
            return p2_early_stop_mem["patience_counter"] >= self.cfg.patience

        self._execute_epoch_loop(
            "ASL Probe-Fitting", [self.context_encoder, self.probe_module, self.cardinal_module], p2_optimizer, 
            train_loader, phase2_closure, num_epochs=self.cfg.probe_epochs, scheduler=p2_scheduler,
            after_epoch=phase2_epoch_callback
        )

        print("\n📥 Training phase concluded. Rolling back to historical maximum validation stage...")
        best_checkpoint_path = os.path.join(self.cfg.checkpoint_dir, "unified_jepa_and_probe_checkpoint.pt")
        
        if os.path.exists(best_checkpoint_path):
            print(f"🔄 Reloading peak weights from: {best_checkpoint_path}")
            self.pipeline.load_checkpoint(best_checkpoint_path, strict=False)
        else:
            print("⚠️ Warning: Best snapshot ledger missing. Maintaining terminal memory state.")

        print("\n🎛️ Training complete. Commencing total system weight de-factorization...")
        defactorize_entire_architecture(self.context_encoder)
        
        self._export_unified_checkpoint(is_final=True)

    def _export_unified_checkpoint(self, is_final=False):
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        filename = "unified_jepa_and_probe.pt" if is_final else "unified_jepa_and_probe_checkpoint.pt"
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, filename)
        
        self.pipeline.save_checkpoint(checkpoint_path=checkpoint_path)
        
        status_tag = "🚀 FINAL PRODUCTION MULTI-MODAL ARTIFACT" if is_final else "📌 INTERMEDIARY MONITORING TRACKER"
        print(f"{status_tag} SAVED COMPLETELY -> {checkpoint_path}")

if __name__ == "__main__":
    cfg = CardioConfig()
    
    train_loader = DataLoader(
        BVTDFlattenedDataset(cfg.train_csv_path, max_seq_len=cfg.max_sequence_len, max_targets=cfg.max_targets), 
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    engine = DualPhaseTrainingEngine(cfg)
    
    engine.run_phase1_pretraining(train_loader)
    engine.run_phase2_probe_fitting(train_loader, "./checkpoints/best_ssl_backbone.pt")
    engine.dump_telemetry()