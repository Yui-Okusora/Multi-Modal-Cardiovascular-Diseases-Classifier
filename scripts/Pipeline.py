# Pipeline.py
import os
import gc
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from config import CardioConfig
from src.ModelModules import (
    ContextEncoder, Predictor, PatientManifoldAssembler, 
    LinearProbeHead, LabelAttentiveSlotProbe, AuxiliaryCardinalityHead
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
        
        if self.cfg.probe_type == "attentive":
            self.probe = LabelAttentiveSlotProbe(
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
            raise FileNotFoundError(f"❌ Missing artifact at: {checkpoint_path}")
            
        print(f"📥 Loading components from unified artifact -> {checkpoint_path}")
        weights = torch.load(checkpoint_path, map_location='cpu')

        if 'context_encoder_state' in weights:
            target_state = weights['context_encoder_state']
            current_state = self.context_encoder.state_dict()
            adjusted_state = {}
            has_lora_keys = any('lora_' in k for k in target_state.keys())
            
            for k, v in target_state.items():
                if not has_lora_keys and (k.endswith('.weight') or k.endswith('.bias')) and '.base_layer.' not in k:
                    wrapper_key = k.replace('.weight', '.base_layer.weight').replace('.bias', '.base_layer.bias')
                    if wrapper_key in current_state:
                        adjusted_state[wrapper_key] = v
                        continue
                adjusted_state[k] = v
            self.context_encoder.load_state_dict(adjusted_state, strict=strict if not has_lora_keys else False)

        if 'predictor_state' in weights and getattr(self, 'predictor', None) is not None: 
            self.predictor.load_state_dict(weights['predictor_state'], strict=strict)
            
        if 'assembler_state' in weights: 
            self.assembler.load_state_dict(weights['assembler_state'], strict=strict)
            
        if 'target_encoder_state' in weights and getattr(self, 'target_encoder', None) is not None: 
            self.target_encoder.load_state_dict(weights['target_encoder_state'], strict=strict)

        if 'probe_state' in weights or 'ensemble_probes_state' in weights:
            if self.probe is None: 
                self.inject_phase2_infrastructure()
            probe_key = 'probe_state' if 'probe_state' in weights else 'ensemble_probes_state'
            
            # 🚀 FIXED: Robust key resolution for cardinal head variants
            card_key = next((k for k in ['cardinal_state', 'ensemble_cardinals_state', 'cardinal_head_state', 'cardinality_head_state'] if k in weights and weights[k] is not None), None)
            
            p_sd = {k.replace("0.", ""): v for k, v in weights[probe_key].items()}
            missing, unexpected = self.probe.load_state_dict(p_sd, strict=False)
            if missing:
                print(f"⚠️ [PROBE WARNING] Unloaded missing parameters: {missing}")
            if unexpected:
                print(f"⚠️ [PROBE WARNING] Unexpected parameters skipped: {unexpected}")

            if card_key:
                c_sd = {k.replace("0.", ""): v for k, v in weights[card_key].items()}
                self.cardinal.load_state_dict(c_sd, strict=False)

        del weights
        gc.collect()

    def process_batch(self, batch: Dict[str, Any], device: torch.device, run_teacher: bool = False) -> Dict[str, Any]:
        B = batch['feature_ids'].size(0)

        z_c_raw = self.context_encoder(
            feature_ids=batch['feature_ids'].to(device), 
            numeric_values=batch['numeric_values'].to(device), 
            cat_result_ids=batch['cat_result_ids'].to(device), 
            timestamps=batch['timestamps'].to(device), 
            padding_mask=batch['student_mask'].to(device)
        )

        if run_teacher:
            z_hat = self.predictor(z_c_raw)
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
                logits = self.probe(z_c_final)
                predicted_cardinalities = self.cardinal(z_c_final)

        multi_hot = torch.zeros(B, self.num_icd_classes, device=device)
        for b_idx in range(B):
            valid_targets = batch['icd_targets'][b_idx][~batch['target_mask'][b_idx]].to(device)
            multi_hot[b_idx, valid_targets] = 1.0

        return {
            'z_c_slots': z_c_final, 
            'z_hat_slots': z_hat, 
            'z_t': z_t,
            'logits': logits, 
            'predicted_cardinalities': predicted_cardinalities,
            'multi_hot_targets': multi_hot, 
            'true_cardinalities': multi_hot.sum(dim=-1)
        }