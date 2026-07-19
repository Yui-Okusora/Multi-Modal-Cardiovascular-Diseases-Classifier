# Pipeline.py
import os
import torch
import torch.nn as nn
import json
from config import CardioConfig
from src.ModelModules import *
from src.LoRAWrapper import *

class ClinicalPipeline:
    """🎯 UNIFIED SYSTEMIC INFRASTRUCTURE PIPELINE (Standalone Production Stream)"""
    def __init__(self, cfg : CardioConfig, device):
        self.cfg = cfg
        self.device = device
        with open(cfg.codebook_json_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)["metadata"]
        self.num_icd_classes = self.meta['num_icd_classes']

        self.context_encoder = ContextEncoder(self.meta['num_total_features'], self.meta['num_cat_results'], cfg.latent_dim, cfg.num_slots, cfg.encoder_layers).to(device)
        self.target_encoder = TargetEncoder(self.meta['num_total_features'], self.meta['num_cat_results'], cfg.latent_dim, cfg.num_slots, cfg.encoder_layers).to(device)
        self.predictor = Predictor(cfg.num_slots, cfg.latent_dim).to(device)
        self.assembler = PatientManifoldAssembler(num_cat_results=self.meta['num_cat_results'], latent_dim=cfg.latent_dim).to(device)
        
        self.augmented_slots = self.cfg.num_slots + 2
        
        # Standalone Task Heads
        self.probe = None
        self.cardinal = None

    def inject_phase2_infrastructure(self):
        print("⚙️ Initializing Standalone Phase 2 LoRA Tracks...")
        inject_lora_infrastructure(self.context_encoder, rank=16, alpha=32)
        
        if self.cfg.probe_type == "attentive":
            self.probe = LabelAttentiveSlotProbe(self.augmented_slots, self.cfg.latent_dim, self.num_icd_classes).to(self.device)
        else:
            self.probe = LinearProbeHead(self.augmented_slots, self.cfg.latent_dim, self.num_icd_classes).to(self.device)
            
        self.cardinal = AuxiliaryCardinalityHead(self.augmented_slots, self.cfg.latent_dim).to(self.device)

    def discard_phase1_components(self):
        print("\n🗑️ Purging Phase 1 dead weight from VRAM...")
        if getattr(self, 'predictor', None) is not None: del self.predictor; self.predictor = None
        if getattr(self, 'target_encoder', None) is not None: del self.target_encoder; self.target_encoder = None
        torch.cuda.empty_cache()
        print("✨ VRAM reclamation complete.")

    def save_checkpoint(self, checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        state = {
            'context_encoder_state': self.context_encoder.state_dict(), 
            'assembler_state':       self.assembler.state_dict(),
            'probe_state':           self.probe.state_dict() if self.probe is not None else None, 
            'cardinal_state':        self.cardinal.state_dict() if self.cardinal is not None else None,
        }
        if getattr(self, 'predictor', None) is not None: state['predictor_state'] = self.predictor.state_dict()
        if getattr(self, 'target_encoder', None) is not None: state['target_encoder_state'] = self.target_encoder.state_dict()
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, strict=True):
        if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"❌ Missing artifact at: {checkpoint_path}")
        print(f"📥 Loading components from unified artifact -> {checkpoint_path}")
        weights = torch.load(checkpoint_path, map_location=self.device)

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

        if 'predictor_state' in weights and getattr(self, 'predictor', None) is not None: self.predictor.load_state_dict(weights['predictor_state'], strict=strict)
        if 'assembler_state' in weights: self.assembler.load_state_dict(weights['assembler_state'], strict=strict)
        if 'target_encoder_state' in weights and getattr(self, 'target_encoder', None) is not None: self.target_encoder.load_state_dict(weights['target_encoder_state'], strict=strict)

        if 'probe_state' in weights or 'ensemble_probes_state' in weights:
            if self.probe is len(self.context_encoder.state_dict()) == 0 or self.probe is None: 
                self.inject_phase2_infrastructure()
            probe_key = 'probe_state' if 'probe_state' in weights else 'ensemble_probes_state'
            card_key = 'cardinal_state' if 'cardinal_state' in weights else 'ensemble_cardinals_state'
            
            # Legacy maps cleanup step if shifting straight from old ledger files
            p_sd = {k.replace("0.", ""): v for k, v in weights[probe_key].items()}
            c_sd = {k.replace("0.", ""): v for k, v in weights[card_key].items()}
            self.probe.load_state_dict(p_sd, strict=False)
            self.cardinal.load_state_dict(c_sd, strict=False)

    def process_batch(self, batch, device, member_idx=None, run_teacher=False):
        """🏃‍♂️ STANDARD MATRIX STREAM: Linear standalone execution path"""
        B = batch['feature_ids'].size(0)

        # ─── STEP 1: COMPUTE CHRONOLOGICAL TIMELINE REPS ───
        z_c_raw = self.context_encoder(
            batch['feature_ids'].to(device), batch['numeric_values'].to(device), 
            batch['cat_result_ids'].to(device), batch['timestamps'].to(device), 
            batch['student_mask'].to(device)
        )

        # ─── STEP 2: PHASE ROUTING ───
        if run_teacher:
            z_c = z_c_raw
            z_hat = self.predictor(z_c)
            z_t = self.target_encoder(
                batch['feature_ids'].to(device), batch['numeric_values'].to(device), 
                batch['cat_result_ids'].to(device), batch['timestamps'].to(device), 
                batch['teacher_mask'].to(device)
            ).detach()
            logits, predicted_cardinalities = None, None
        else:
            # Assembly flow appending patient covariates natively
            z_c = self.assembler(z_c_raw, batch['age'].to(device).float(), batch['gender'].to(device).long())
            z_hat, z_t = None, None
            
            logits, predicted_cardinalities = None, None
            if self.probe is not None:
                logits = self.probe(z_c)
                predicted_cardinalities = self.cardinal(z_c)

        # ─── STEP 3: REPLICATED CLINICAL TARGET GENERATION ───
        multi_hot = torch.zeros(B, self.num_icd_classes, device=device)
        for b_idx in range(B):
            multi_hot[b_idx, batch['icd_targets'][b_idx][~batch['target_mask'][b_idx]].to(device)] = 1.0

        return {
            'z_c_slots': z_c, 'z_hat_slots': z_hat, 'z_t': z_t,
            'logits': logits, 'predicted_cardinalities': predicted_cardinalities,
            'multi_hot_targets': multi_hot, 'true_cardinalities': multi_hot.sum(dim=-1)
        }