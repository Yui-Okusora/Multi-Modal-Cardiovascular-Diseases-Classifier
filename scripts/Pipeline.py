import torch
import json

from config import *
from src.ModelModules import *

class ClinicalPipeline:
    """
    🎯 CENTRALIZED ASYMMETRIC TIMELINE PIPELINE:
    Splits variable patient histories into chronological past (student) and 
    future (teacher) views for pure self-supervised prediction loops.
    """
    def __init__(self, cfg: CardioConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        
        # Load the dimensional metadata schema dynamically from the codebook artifact
        with open(cfg.codebook_json_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)["metadata"]
            
        # Extract target class constraints for tracking and logging
        self.num_icd_classes = self.meta['num_icd_classes']

        # ─────────────── PHASE 1 STRUCTURAL REPRESENTATION BLOCK ───────────────
        self.context_encoder = ContextEncoder(
            num_total_features=self.meta['num_total_features'],
            num_cat_results=self.meta['num_cat_results'],
            d_model=cfg.latent_dim,
            num_slots=cfg.num_slots,
            nlayers=cfg.encoder_layers
        ).to(self.device)

        self.predictor = Predictor(
            num_slots=cfg.num_slots, 
            d_model=cfg.latent_dim
        ).to(self.device)

        self.target_encoder = TargetEncoder(
            num_total_features=self.meta['num_total_features'],
            num_cat_results=self.meta['num_cat_results'],
            d_model=cfg.latent_dim,
            num_slots=cfg.num_slots,
            nlayers=cfg.encoder_layers
        ).to(self.device)

        # ─────────────── 🎯 PHASE 2 STRUCTURAL PROBING HEAD ───────────────
        self.linear_probe = LinearProbeHead(
            in_slots=cfg.num_slots, 
            in_dim=cfg.latent_dim, 
            num_classes=self.num_icd_classes
        ).to(self.device)

    def process_batch(self, batch, device, run_teacher=False, contextual_split_ratio=0.70):
        # 1. Gather global multi-hot targets for validation classifier heads downstream
        tgt_icd = batch['icd_targets'].to(device, non_blocking=True)
        tgt_mask = batch['target_mask'].to(device, non_blocking=True)
        batch_size = tgt_icd.size(0)
        multi_hot_targets = torch.zeros(batch_size, self.num_icd_classes, device=device)
        for b_idx in range(batch_size):
            multi_hot_targets[b_idx, tgt_icd[b_idx][~tgt_mask[b_idx]]] = 1.0

        # 2. Chronological Slicing along the Sequence Timeline Axis
        seq_len = batch['feature_ids'].size(1)
        split_idx = max(1, int(seq_len * contextual_split_ratio))
        
        # Student Inputs (Past Block)
        f_ids_past = batch['feature_ids'][:, :split_idx].to(device)
        v_nums_past = batch['numeric_values'][:, :split_idx].to(device)
        c_ids_past = batch['cat_result_ids'][:, :split_idx].to(device)
        times_past = batch['timestamps'][:, :split_idx].to(device)
        pad_mask_past = batch['padding_mask'][:, :split_idx].to(device)

        # 3. Forward Student Path
        z_c_norm = self.context_encoder(f_ids_past, v_nums_past, c_ids_past, times_past, pad_mask_past)
        z_hat_slots = self.predictor(z_c_norm)
        
        # 4. Forward Teacher Path (Future Block)
        z_t = None
        if run_teacher and self.target_encoder is not None:
            f_ids_fut = batch['feature_ids'][:, split_idx:].to(device)
            v_nums_fut = batch['numeric_values'][:, split_idx:].to(device)
            c_ids_fut = batch['cat_result_ids'][:, split_idx:].to(device)
            times_fut = batch['timestamps'][:, split_idx:].to(device)
            pad_mask_fut = batch['padding_mask'][:, split_idx:].to(device)
            
            z_t = self.target_encoder(f_ids_fut, v_nums_fut, c_ids_fut, times_fut, pad_mask_fut)
            
        return {
            'z_c_norm': z_c_norm,       # Encoded summary of the past
            'z_hat_slots': z_hat_slots, # Projected view of the future
            'z_t': z_t,                 # Actual encoded future coordinates
            'multi_hot_targets': multi_hot_targets
        }