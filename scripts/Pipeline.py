# Pipeline.py
import torch
import json
from config import *
from src.ModelModules import *

class ClinicalPipeline:
    """🎯 VECTORIZED INFRASTRUCTURE PIPELINE (REVERTED & CLEANED)"""
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        with open(cfg.codebook_json_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)["metadata"]
        self.num_icd_classes = self.meta['num_icd_classes']

        self.context_encoder = ContextEncoder(
            self.meta['num_total_features'], self.meta['num_cat_results'], cfg.latent_dim, cfg.num_slots, cfg.encoder_layers
        ).to(device)
        self.predictor = Predictor(cfg.num_slots, cfg.latent_dim).to(device)
        self.target_encoder = TargetEncoder(
            self.meta['num_total_features'], self.meta['num_cat_results'], cfg.latent_dim, cfg.num_slots, cfg.encoder_layers
        ).to(device)
        self.linear_probe = LinearProbeHead(cfg.num_slots, cfg.latent_dim, self.num_icd_classes).to(device)

    def load_checkpoint(self, checkpoint_path, strict=True):
        """
        💾 PLUGGABLE ARTIFACT LOADER:
        Dynamically restores available module parameters based on checkpoint state keys.
        Supports both standalone SSL backbones and fully unified probe assemblies.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ Target checkpoint file missing at: {checkpoint_path}")

        print(f"📥 Loading pipeline components from checkpoint artifact -> {checkpoint_path}")
        # Enforce device synchronization to prevent VRAM memory mismatches
        weights = torch.load(checkpoint_path, map_location=self.device)
        
        # Track loaded modules for terminal diagnostic visibility
        loaded_modules = []

        # 🔹 Extract Phase 1/2 Shared Foundation Backbone
        if 'context_encoder_state' in weights:
            self.context_encoder.load_state_dict(weights['context_encoder_state'], strict=strict)
            loaded_modules.append("ContextEncoder")
            
        if 'predictor_state' in weights:
            self.predictor.load_state_dict(weights['predictor_state'], strict=strict)
            loaded_modules.append("Predictor")

        # 🔹 Extract Target Encoder state (if explicitly saved during training)
        if 'target_encoder_state' in weights:
            self.target_encoder.load_state_dict(weights['target_encoder_state'], strict=strict)
            loaded_modules.append("TargetEncoder")
        elif 'context_encoder_state' in weights:
            # Fallback alignment: synchronize the teacher to match the student state if initializing fresh
            for param_s, param_t in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_t.data.copy_(param_s.data)
            loaded_modules.append("TargetEncoder (Synchronized from Context)")

        # 🔹 Extract Phase 2 Linear Probe Head
        if 'linear_probe_state' in weights:
            self.linear_probe.load_state_dict(weights['linear_probe_state'], strict=strict)
            loaded_modules.append("LinearProbeHead")

        print(f"✨ Successfully restored modules: {', '.join(loaded_modules)}")
        
        # Switch components to read-only evaluation posture automatically
        self.context_encoder.eval()
        self.predictor.eval()
        self.linear_probe.eval()

    def process_batch(self, batch, device, run_teacher=False):
        # Tensors arrive cleanly pre-shaped at [B, T] from the DataLoader channel
        f_ids  = batch['feature_ids'].to(device)
        v_nums = batch['numeric_values'].to(device)
        c_ids  = batch['cat_result_ids'].to(device)
        times  = batch['timestamps'].to(device)
        s_mask = batch['student_mask'].to(device)

        # Process the full tensor block in a single parallel operation pass
        z_c = self.context_encoder(f_ids, v_nums, c_ids, times, s_mask)
        z_hat = self.predictor(z_c)
        
        z_t = None
        if run_teacher:
            t_mask = batch['teacher_mask'].to(device)
            z_t = self.target_encoder(f_ids, v_nums, c_ids, times, t_mask)

        # Map and expand multi-hot labels cleanly
        tgt_icd, tgt_mask = batch['icd_targets'].to(device), batch['target_mask'].to(device)
        B = f_ids.size(0)
        multi_hot = torch.zeros(B, self.num_icd_classes, device=device)
        for b_idx in range(B):
            multi_hot[b_idx, tgt_icd[b_idx][~tgt_mask[b_idx]]] = 1.0

        return {
            'z_c_norm': z_c,
            'z_hat_slots': z_hat,
            'z_t': z_t,
            'multi_hot_targets': multi_hot
        }