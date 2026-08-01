# Pipeline.py
import os
import gc
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from config import CardioConfig
from src.ModelModules import (
    ContextEncoder, Predictor, PatientManifoldAssembler, ContinuousHopfieldMemory,
    LinearProbeHead, LabelAttentiveProbe, AuxiliaryCardinalityHead
)
from src.LoRAWrapper import inject_lora_infrastructure


class ClinicalPipeline:
    def __init__(self, cfg: CardioConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        
        with open(cfg.codebook_json_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)["metadata"]
        self.num_icd_classes = self.meta['num_icd_classes']

        self.context_encoder = ContextEncoder(
            num_total_features=self.meta['num_total_features'], 
            num_cat_results=self.meta['num_cat_results'], 
            d_model=cfg.latent_dim, 
            num_slots=cfg.num_slots, 
            nlayers=cfg.encoder_layers
        ).to(device)

        self.target_encoder = ContextEncoder(
            num_total_features=self.meta['num_total_features'], 
            num_cat_results=self.meta['num_cat_results'], 
            d_model=cfg.latent_dim, 
            num_slots=cfg.num_slots, 
            nlayers=cfg.encoder_layers
        ).to(device)
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = Predictor(
            num_slots=cfg.num_slots, 
            d_model=cfg.latent_dim
        ).to(device)

        self.assembler = PatientManifoldAssembler(
            num_cat_results=self.meta['num_cat_results'], 
            latent_dim=cfg.latent_dim,
            covariate_scale=cfg.covariate_scale
        ).to(device)
        
        self.augmented_slots = cfg.augmented_slots
        self.hopfield = None
        self.probe = None
        self.cardinal = None

    def inject_phase2_infrastructure(self):
        if self.cfg.encoder_lora:
            inject_lora_infrastructure(
                self.context_encoder, 
                rank=self.cfg.lora_rank, 
                alpha=self.cfg.lora_alpha,
                target_names=self.cfg.lora_target_names
            )

        if self.cfg.use_hopfield_memory:
            self.hopfield = ContinuousHopfieldMemory(
                in_dim=self.cfg.latent_dim,
                num_prototypes=self.cfg.num_prototypes,
                num_heads=self.cfg.num_hopfield_heads
            ).to(self.device)
        
        if self.cfg.probe_type == "attentive":
            self.probe = LabelAttentiveProbe(
                in_slots=self.augmented_slots, 
                in_dim=self.cfg.latent_dim, 
                num_classes=self.num_icd_classes,
                num_prototypes=self.cfg.num_prototypes,
                num_hopfield_heads=self.cfg.num_hopfield_heads,
                dropout_p=self.cfg.probe_dropout
            ).to(self.device)
        else:
            self.probe = LinearProbeHead(
                in_slots=self.augmented_slots, 
                in_dim=self.cfg.latent_dim, 
                num_classes=self.num_icd_classes
            ).to(self.device)
            
        self.cardinal = AuxiliaryCardinalityHead(
            in_slots=self.augmented_slots, 
            in_dim=self.cfg.latent_dim
        ).to(self.device)

    def discard_phase1_components(self):
        print("\n🗑️ Purging Phase 1 dead weight from VRAM...")
        if getattr(self, 'predictor', None) is not None:
            self.predictor.zero_grad(set_to_none=True)
            del self.predictor
            self.predictor = None
            
        if getattr(self, 'target_encoder', None) is not None:
            self.target_encoder.zero_grad(set_to_none=True)
            del self.target_encoder
            self.target_encoder = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✨ VRAM reclamation complete.")

    def save_checkpoint(self, checkpoint_path: str):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        state = {
            'context_encoder_state': self.context_encoder.state_dict(), 
            'assembler_state':       self.assembler.state_dict(),
            'hopfield_state':        self.hopfield.state_dict() if self.hopfield is not None else None,
            'probe_state':           self.probe.state_dict() if self.probe is not None else None, 
            'cardinal_state':        self.cardinal.state_dict() if self.cardinal is not None else None,
        }
        if getattr(self, 'predictor', None) is not None: 
            state['predictor_state'] = self.predictor.state_dict()
        if getattr(self, 'target_encoder', None) is not None: 
            state['target_encoder_state'] = self.target_encoder.state_dict()
            
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str, strict: bool = True):
        if not os.path.exists(checkpoint_path): 
            raise FileNotFoundError(f"Missing artifact at: {checkpoint_path}")
            
        print(f"📥 Loading components from unified artifact -> {checkpoint_path}")
        weights = torch.load(checkpoint_path, map_location='cpu')

        # 1. Base Backbone Components
        if 'context_encoder_state' in weights:
            self.context_encoder.load_state_dict(weights['context_encoder_state'], strict=False)
        if 'assembler_state' in weights: 
            self.assembler.load_state_dict(weights['assembler_state'], strict=strict)

        if weights.get('hopfield_state') is not None:
            if self.hopfield is None and self.cfg.use_hopfield_memory: 
                self.inject_phase2_infrastructure()
            if self.hopfield is not None:
                self.hopfield.load_state_dict(weights['hopfield_state'], strict=strict)

        # 2. Hardcoded Phase 2 Heads
        if weights.get('probe_state') is not None:
            if self.probe is None: 
                self.inject_phase2_infrastructure()
            self.probe.load_state_dict(weights['probe_state'], strict=strict)

        if weights.get('cardinal_state') is not None:
            if self.cardinal is None:
                self.inject_phase2_infrastructure()
            self.cardinal.load_state_dict(weights['cardinal_state'], strict=strict)
            print("✨ [CARDINAL HEAD] Successfully restored trained weights!")

        del weights
        gc.collect()

    def process_batch(self, batch: Dict[str, Any], device: torch.device, run_teacher: bool = False) -> Dict[str, Any]:
        B = batch['feature_ids'].size(0)

        active_padding_mask = batch['student_mask'].to(device) if run_teacher else batch['base_mask'].to(device)

        z_c_raw = self.context_encoder(
            feature_ids=batch['feature_ids'].to(device), 
            numeric_values=batch['numeric_values'].to(device), 
            cat_result_ids=batch['cat_result_ids'].to(device), 
            timestamps=batch['timestamps'].to(device), 
            padding_mask=active_padding_mask
        )

        if run_teacher:
            dt_target = batch['dt_target'].to(device).unsqueeze(-1)  # Shape: [B, 1]
            future_time_emb = self.context_encoder.tokenizer.time_embedder(dt_target)  # Shape: [B, 1, 512]

            z_hat = self.predictor(z_c_raw, future_time_emb=future_time_emb)
            with torch.no_grad():
                z_t = self.target_encoder(
                    feature_ids=batch['feature_ids'].to(device), 
                    numeric_values=batch['numeric_values'].to(device), 
                    cat_result_ids=batch['cat_result_ids'].to(device), 
                    timestamps=batch['timestamps'].to(device), 
                    padding_mask=batch['teacher_mask'].to(device)
                ).detach()
            z_c_final = z_c_raw
            logits, predicted_cardinalities = None, None
        else:
            z_c_final = self.assembler(
                z_c_raw, 
                batch['age'].to(device).float(), 
                batch['gender'].to(device).long()
            )
            z_hat, z_t = None, None
            logits, predicted_cardinalities = None, None
            if self.probe is not None:
                predicted_cardinalities = self.cardinal(z_c_final)

                if self.hopfield is not None:
                    z_c_final = self.hopfield(z_c_final)
                logits = self.probe(z_c_final)

        icd_targets = batch['icd_targets'].to(device).long()
        target_mask = batch['target_mask'].to(device)

        valid_indices = icd_targets.masked_fill(target_mask, -1)
        valid_mask = (valid_indices >= 0) & (valid_indices < self.num_icd_classes)
        
        safe_indices = valid_indices.clamp(min=0, max=self.num_icd_classes - 1)
        multi_hot = torch.zeros(B, self.num_icd_classes, device=device)
        multi_hot.scatter_(1, safe_indices, valid_mask.float())

        return {
            'z_c_slots': z_c_final, 
            'z_hat_slots': z_hat, 
            'z_t': z_t,
            'logits': logits, 
            'predicted_cardinalities': predicted_cardinalities,
            'multi_hot_targets': multi_hot, 
            'true_cardinalities': multi_hot.sum(dim=-1)
        }