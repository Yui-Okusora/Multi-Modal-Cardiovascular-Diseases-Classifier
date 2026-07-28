# scripts/trainer.py
import os
import gc
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from src.TimelineDataset import BVTDFlattenedDataset, compute_static_class_frequencies
from src.ModelModules import compute_comprehensive_manifold_diagnostics, ClassAwareASL, execute_clinical_audit
from src.BaseEngine import BaseExecutionEngine
from src.LoRAWrapper import defactorize_entire_architecture
from config import CardioConfig
from Pipeline import ClinicalPipeline

import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 
logging.getLogger('matplotlib').setLevel(logging.WARNING)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class VICRegProjector(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.GELU(), 
            nn.Linear(hidden_dim, out_dim)
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
            self.meta = json.load(f)["metadata"]

        self.cfg = cfg
        self.pipeline = ClinicalPipeline(cfg, self.device)

        self.context_encoder = self.pipeline.context_encoder
        self.predictor       = self.pipeline.predictor
        self.assembler       = self.pipeline.assembler
        self.target_encoder  = self.pipeline.target_encoder
        
        self.context_projector = VICRegProjector(in_dim=cfg.latent_dim).to(self.device)
        self.probe_module    = None
        self.cardinal_module = None

        self.val_loader = DataLoader(
            BVTDFlattenedDataset(
                cfg.val_csv_path, 
                max_seq_len=cfg.max_sequence_len, 
                max_targets=cfg.max_targets,
                is_train=False
            ), 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )

        self.S_target = self.compute_label_cooccurrence_matrix(cfg.train_csv_path, num_classes=self.meta['num_icd_classes'])
        self.frequencies = compute_static_class_frequencies(cfg.train_csv_path, num_classes=self.meta['num_icd_classes'])

    def compute_label_cooccurrence_matrix(self, csv_path: str, num_classes: int) -> torch.Tensor:
        import pandas as pd
        df = pd.read_csv(csv_path).fillna("")
        targets = np.zeros((len(df), num_classes), dtype=np.float32)

        for idx, target_str in enumerate(df['icd_targets']):
            tokens = str(target_str).strip().split()
            for t in tokens:
                try:
                    c_id = int(t)
                    if c_id < num_classes:
                        targets[idx, c_id] = 1.0
                except ValueError:
                    pass

        Y = torch.tensor(targets, dtype=torch.float32)
        y_norms = torch.norm(Y, p=2, dim=0, keepdim=True) + 1e-6
        Y_norm = Y / y_norms
        S_target = Y_norm.T @ Y_norm
        return S_target.to(self.device)
    
    @torch.no_grad()
    def audit_phase1_manifold_health(self, z_context: torch.Tensor):
        diagnostics = compute_comprehensive_manifold_diagnostics(z_context)
        return diagnostics["batch_std"], diagnostics["slot_cross_talk"], diagnostics["effective_rank"]

    def execute_validation_pass(self, phase="Pretraining"):
        self.context_encoder.eval()
        self.assembler.eval()
        
        if getattr(self, 'predictor', None) is not None: self.predictor.eval()
        if getattr(self, 'context_projector', None) is not None: self.context_projector.eval()
        if getattr(self, 'target_encoder', None) is not None: self.target_encoder.eval()

        if phase != "Pretraining" and self.probe_module is not None:
            self.probe_module.eval()
            self.cardinal_module.eval()

        all_probs, all_targets, all_pred_counts = [], [], []
        all_val_z_hat = [] 
        total_val_loss = 0.0

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.cfg.amp_dtype, enabled=self.cfg.use_amp):
                for batch in self.val_loader:
                    out = self.pipeline.process_batch(batch, self.device, run_teacher=(phase == "Pretraining"))

                    if phase == "Pretraining":
                        loss_align = self.compute_alignment_loss(
                            out['z_hat_slots'], out['z_t'].detach(), beta=self.cfg.alignment_smooth_l1_beta
                        )
                        p_c = self.context_projector(out['z_c_slots'])
                        
                        val_loss = (
                            self.cfg.alpha_align * loss_align + 
                            self.cfg.alpha_var * self.compute_variance_loss(p_c) + 
                            self.cfg.alpha_backbone_v * self.compute_variance_loss(out['z_c_slots'], target_std=self.cfg.target_std) + 
                            self.cfg.alpha_cov * self.compute_covariance_loss(p_c) + 
                            self.cfg.alpha_diverse * self.compute_cross_slot_orthogonal_loss(out['z_c_slots'])
                        )
                        total_val_loss += val_loss.item()
                        all_val_z_hat.append(out['z_c_slots'].cpu())
                    else:
                        all_probs.append(torch.sigmoid(out['logits']).float().cpu().numpy())
                        all_targets.append(out['multi_hot_targets'].cpu().numpy())
                        all_pred_counts.append(out['predicted_cardinalities'].float().cpu().numpy())

        if phase == "Pretraining":
            mean_loss = total_val_loss / max(1, len(self.val_loader))
            concat_z_hat = torch.cat(all_val_z_hat, dim=0).to(self.device)
            m_std, m_talk, eff_rank = self.audit_phase1_manifold_health(concat_z_hat)

            del concat_z_hat, all_val_z_hat
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
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
                temp_alpha=self.cfg.eval_temp_alpha,
                silent=True
            )

    def run_phase1_pretraining(self, train_loader: DataLoader):
        print("\n" + "="*80 + "\n🧬 PHASE 1: FOUNDATIONAL PHYSIOLOGICAL WORLD MODEL INITIALIZATION\n" + "="*80)
        
        self.context_encoder.train()
        self.predictor.train()
        self.context_projector.train()
        self.target_encoder.eval()

        phase1_models = [self.context_encoder, self.predictor, self.context_projector]

        decay_params, no_decay_params = [], []
        for model in phase1_models:
            for name, param in model.named_parameters():
                if not param.requires_grad: 
                    continue
                if any(k in name.lower() for k in ["norm", "bias", "embedding", "frequencies"]):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
                
        p1_optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": self.cfg.pretrain_wgt_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ], lr=self.cfg.pretrain_lr)

        total_steps = len(train_loader) * self.cfg.pretrain_epochs
        warmup_steps = int(total_steps * self.cfg.pretrain_warmup_ratio)
        p1_scheduler = self.create_warmup_cosine_scheduler(p1_optimizer, warmup_steps, total_steps, min_lr_ratio=0.0)

        with torch.no_grad():
            print("🔄 Synchronizing pristine base parameters to Target Encoder state maps...")
            for param_s, param_t in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_t.data.copy_(param_s.data)
                param_t.requires_grad = False
            print("✨ Symmetrical teacher alignment finalized successfully.")

        def get_active_alignment_weight(step: int, warmup_steps: int, max_alpha: float):
            if step >= warmup_steps: 
                return max_alpha
            return max_alpha * (0.5 * (1.0 - math.cos((step / warmup_steps) * math.pi)))

        def absolute_jepa_closure(batch, step: int, total_steps_val: int):
            out = self.pipeline.process_batch(batch, self.device, run_teacher=True)

            loss_align = self.compute_alignment_loss(
                out['z_hat_slots'], out['z_t'].detach(), beta=self.cfg.alignment_smooth_l1_beta
            )
            p_c = self.context_projector(out['z_c_slots'])
            
            if step % self.cfg.log_interval == 0:
                m_std, m_talk, eff_rank = self.audit_phase1_manifold_health(out['z_c_slots'].detach())
                print(f"   [MANIFOLD HEALTH Step {step}] Batch Std: {m_std:.3f} │ Slot Cross-Talk: {m_talk:.3f} │ Effective Rank: {eff_rank:.1f}")

            active_alpha_align = get_active_alignment_weight(step, warmup_steps=warmup_steps, max_alpha=self.cfg.alpha_align)

            return {
                "loss_total_align": [active_alpha_align, loss_align],
                "loss_variance":    [self.cfg.alpha_var, self.compute_variance_loss(p_c)],
                "loss_backbone_v":  [self.cfg.alpha_backbone_v, self.compute_variance_loss(out['z_c_slots'], target_std=self.cfg.target_std)], 
                "loss_covariance":  [self.cfg.alpha_cov, self.compute_covariance_loss(p_c)],
                "loss_diversity":   [self.cfg.alpha_diverse, self.compute_cross_slot_orthogonal_loss(out['z_c_slots'])]
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

        def phase1_epoch_callback(epoch_idx: int):
            audit_dict = self.execute_validation_pass(phase="Pretraining")
            
            self.context_encoder.train()
            self.predictor.train()
            self.assembler.train()
            self.context_projector.train()
            self.target_encoder.eval()

            val_loss = audit_dict["loss"]
            m_std = audit_dict["batch_std"]
            m_talk = audit_dict["slot_cross_talk"]
            eff_rank = audit_dict["effective_rank"]
            
            std_health = max(0.5, min(m_std / self.cfg.target_std, 1.0))
            talk_health = 1.0 - max(0.0, min(m_talk, 0.5))
            health_multiplier = std_health * talk_health
            manifold_health_score = - (val_loss / max(health_multiplier, 1e-6))
            
            print(
                f"📊 [MANIFOLD HEALTH SCORECARD Epoch {epoch_idx:02d}] "
                f"Global Index: {manifold_health_score:.4f} │ "
                f"Val Loss: {val_loss:.4f} │ "
                f"Batch Std: {m_std:.3f} │ "
                f"Slot Cross-Talk: {m_talk:.3f} │ "
                f"Effective Rank: {eff_rank:.1f}"
            )
            
            best_score = early_stop_mem["best_score"]
            if manifold_health_score > best_score:
                print(f"🔥 [CHECKPOINT] Higher manifold health index achieved ({best_score:.4f} -> {manifold_health_score:.4f}). Exporting backbone...")
                early_stop_mem["best_score"] = manifold_health_score
                early_stop_mem["patience_counter"] = 0
                
                self._export_checkpoint({
                    "context_encoder_state": self.context_encoder.state_dict(), 
                    "predictor_state": self.predictor.state_dict()
                }, self.cfg.best_ssl_backbone_filename)
            else:
                early_stop_mem["patience_counter"] += 1
                print(f"⚠️ [PATIENCE] Phase 1 manifold health has stalled for {early_stop_mem['patience_counter']}/{self.cfg.patience} epochs.")
                
            return early_stop_mem["patience_counter"] >= self.cfg.patience

        self._execute_epoch_loop(
            "Pure-SSL JEPA", phase1_models, p1_optimizer, train_loader, 
            absolute_jepa_closure, self.cfg.pretrain_epochs, p1_scheduler, 
            after_step=apply_momentum_teacher_update, after_epoch=phase1_epoch_callback
        )

    def run_phase2_probe_fitting(self, train_loader: DataLoader, load_checkpoint_path: str = None):
        print("\n" + "="*80 + "\n⚙️ PHASE 2: INJECTING ADAPTER TRACKS & REPRODUCIBLE BACKPROPAGATION\n" + "="*80)
        
        self.predictor, self.target_encoder, self.context_projector = None, None, None
        self.pipeline.discard_phase1_components()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        if load_checkpoint_path is not None:
            self.pipeline.load_checkpoint(load_checkpoint_path, strict=False)

        self.context_encoder.eval()
        for param in self.context_encoder.parameters():
            param.requires_grad = False

        self.pipeline.inject_phase2_infrastructure()
        self.probe_module = self.pipeline.probe
        self.cardinal_module = self.pipeline.cardinal

        self.probe_module.train()
        self.cardinal_module.train()
        self.assembler.train()

        backbone_lora_params = [p for p in self.context_encoder.parameters() if p.requires_grad]
        assembler_params     = [p for p in self.assembler.parameters() if p.requires_grad]
        probe_params         = [p for p in self.probe_module.parameters() if p.requires_grad]
        cardinal_params      = [p for p in self.cardinal_module.parameters() if p.requires_grad]

        optimized_parameters = [
            {"params": backbone_lora_params, "lr": self.cfg.probe_lr * self.cfg.probe_lr_backbone_scale,  "weight_decay": 1e-2},
            {"params": assembler_params,     "lr": self.cfg.probe_lr * self.cfg.probe_lr_assembler_scale, "weight_decay": self.cfg.probe_wgt_decay},
            {"params": probe_params,         "lr": self.cfg.probe_lr,                                      "weight_decay": self.cfg.probe_wgt_decay},
            {"params": cardinal_params,      "lr": self.cfg.probe_lr,                                      "weight_decay": self.cfg.probe_wgt_decay}
        ]
        p2_optimizer = torch.optim.AdamW(optimized_parameters)

        total_steps = len(train_loader) * self.cfg.probe_epochs
        warmup_steps = len(train_loader)
        p2_scheduler = self.create_warmup_cosine_scheduler(
            p2_optimizer, num_warmup_steps=warmup_steps, num_total_steps=total_steps, min_lr_ratio=0.01
        )

        criterion_cls = ClassAwareASL(
            class_frequencies=self.frequencies,
            gamma_pos=self.cfg.asl_gamma_pos,
            gamma_neg_base=self.cfg.asl_gamma_neg_base,
            beta_neg_base=self.cfg.asl_beta_neg_base,
            delta_beta=self.cfg.asl_delta_beta
        ).to(self.cfg.device)
        criterion_reg = nn.MSELoss()

        def phase2_closure(batch, step_idx: int, total_steps: int):
            out = self.pipeline.process_batch(batch, self.device, run_teacher=False)

            cls_loss = criterion_cls(out['logits'], out['multi_hot_targets'])
            card_loss = criterion_reg(out['predicted_cardinalities'].view(-1), out['true_cardinalities'].view(-1))
            
            proto_loss = torch.tensor(0.0, device=self.device)
            if hasattr(self.probe_module, 'prototype_memory') and hasattr(self.probe_module, 'k_proj'):
                K_proto = self.probe_module.k_proj(self.probe_module.prototype_memory)
                K_norm = F.normalize(K_proto, p=2, dim=-1)
                Gram = K_norm @ K_norm.T
                I = torch.eye(K_norm.size(0), device=self.device)
                num_off_diag = Gram.size(0) * (Gram.size(0) - 1)
                proto_loss = (((Gram - I) ** 2).sum() / num_off_diag) * self.cfg.proto_loss_scale

            cooccur_loss = torch.tensor(0.0, device=self.device)
            if hasattr(self.probe_module, 'weight_class'):
                W_norm = F.normalize(self.probe_module.weight_class, p=2, dim=-1)
                S_pred = W_norm @ W_norm.T
                cooccur_loss = F.mse_loss(S_pred, self.S_target)

            return {
                "loss_classification":      [self.cfg.loss_weight_cls, cls_loss],
                "loss_cardinality_mse":     [self.cfg.loss_weight_cardinality_mse, card_loss],
                "loss_prototype_diversity": [self.cfg.loss_weight_prototype_diversity, proto_loss],
                "loss_label_cooccurrence":  [self.cfg.loss_weight_label_cooccurrence, cooccur_loss]
            }

        p2_metrics = self.telemetry.setdefault("ASL Probe-Fitting", {"loss": []})
        p2_early_stop_mem = p2_metrics.setdefault("early_stop_memory", {"best_score": -float('inf'), "patience_counter": 0})

        def phase2_epoch_callback(epoch_idx: int):
            audit_data = self.execute_validation_pass(phase="Probe")
            
            self.probe_module.train()
            self.cardinal_module.train()
            self.assembler.train()

            p1 = max(audit_data["macro_auc_pr"], 1e-4)
            p2 = max(audit_data["macro_f1"], 1e-4)
            p3 = max(audit_data["adaptive_precision"], 1e-4)
            harmonic_core = 3.0 / ((1.0 / p1) + (1.0 / p2) + (1.0 / p3))
            
            composite_score = (self.cfg.harmonic_core_weight * harmonic_core) + (self.cfg.top5_rate_weight * audit_data["top5_rate"])
            best_score = p2_early_stop_mem["best_score"]
            
            print("\n" + "╒" + "═"*78 + "╕")
            print(f" │ 🏥 ALL-INCLUSIVE PHASE 2 VALIDATION SCORECARD (EPOCH {epoch_idx:02d})")
            print(" ├" + "─"*78 + "┤")
            print(f" │ ✨ GLOBAL MULTI-DIMENSIONAL INDEX (Harmonic): {composite_score:.4f}% │ Best Peak: {best_score:.2f}%")
            print(f" │ 🧬 Harmonic Core (PR-AUC / F1 / AdaptPrec):    {harmonic_core:.4f}%")
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
            "ASL Probe-Fitting", [self.assembler, self.probe_module, self.cardinal_module], 
            p2_optimizer, train_loader, phase2_closure, num_epochs=self.cfg.probe_epochs, scheduler=p2_scheduler,
            after_epoch=phase2_epoch_callback
        )

        print("\n📥 Training phase concluded. Rolling back to historical maximum validation stage...")
        best_checkpoint_path = os.path.join(self.cfg.checkpoint_dir, self.cfg.unified_intermediary_filename)
        
        if os.path.exists(best_checkpoint_path):
            print(f"🔄 Reloading peak weights from: {best_checkpoint_path}")
            self.pipeline.load_checkpoint(best_checkpoint_path, strict=False)

        print("\n🎛️ Training complete. Commencing total system weight de-factorization...")
        defactorize_entire_architecture(self.context_encoder)
        self._export_unified_checkpoint(is_final=True)

    def _export_unified_checkpoint(self, is_final: bool = False):
        filename = self.cfg.unified_final_filename if is_final else self.cfg.unified_intermediary_filename
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, filename)
        
        self.pipeline.save_checkpoint(checkpoint_path=checkpoint_path)
        status_tag = "🚀 FINAL PRODUCTION MULTI-MODAL ARTIFACT" if is_final else "📌 INTERMEDIARY MONITORING TRACKER"
        print(f"{status_tag} SAVED COMPLETELY -> {checkpoint_path}")


if __name__ == "__main__":
    cfg = CardioConfig()
    
    p1_train_loader = DataLoader(
        BVTDFlattenedDataset(
            cfg.train_csv_path, 
            max_seq_len=cfg.max_sequence_len, 
            max_targets=cfg.max_targets,
            is_train=True,
            k_min=cfg.k_min,
            k_max=cfg.k_max
        ), 
        batch_size=cfg.batch_size,
        shuffle=True, drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    p2_train_loader = DataLoader(
        BVTDFlattenedDataset(
            cfg.train_csv_path, 
            max_seq_len=cfg.max_sequence_len, 
            max_targets=cfg.max_targets,
            is_train=False
        ), 
        batch_size=cfg.batch_size,
        shuffle=True, drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    
    engine = DualPhaseTrainingEngine(cfg)
    best_ssl_path = os.path.join(cfg.checkpoint_dir, cfg.best_ssl_backbone_filename)
    
    # ─── EXECUTION STAGES ───
    # Uncomment run_phase1_pretraining if training from scratch:
    # engine.run_phase1_pretraining(p1_train_loader)
    
    # Run Phase 2 probe fitting:
    engine.run_phase2_probe_fitting(
        p2_train_loader, 
        load_checkpoint_path=best_ssl_path if os.path.exists(best_ssl_path) else None
    )
    engine.dump_telemetry()