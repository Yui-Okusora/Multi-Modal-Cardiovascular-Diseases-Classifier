# scripts/xai_analytics.py
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from torch.utils.data import DataLoader
from captum.attr import LayerIntegratedGradients

from config import CardioConfig
from src.TimelineDataset import BVTDFlattenedDataset  
from src.ModelModules import *
from Pipeline import ClinicalPipeline

import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class AdvancedClinicalAnalyticsEngine:
    def __init__(self, cfg: CardioConfig):
        self.cfg = cfg
        self.device = cfg.device
        os.makedirs("./xai_exports", exist_ok=True)
        sns.set_theme(style="ticks")
        
        with open(cfg.codebook_json_path, "r", encoding="utf-8") as f:
            self.codebooks = json.load(f)
        self.id_to_icd = self.codebooks.get("inverse_icd_codes", {})
        self.decoder = ClinicalDecoder(cfg.codebook_json_path)

        # ⚡ OPERATIONAL CHUNK: Step down to 16 to avoid Captum sequential graph memory buildup
        self.val_loader = DataLoader(
            BVTDFlattenedDataset(cfg.val_csv_path, max_seq_len=cfg.max_sequence_len, max_targets=cfg.max_targets), 
            batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True
        )
        self.target_class = 16  # Primary cardiovascular tracking channel

    def _load_pipeline(self):
        print("🏭 Instantiating Clinical Pipeline Core Framework...")
        pipeline = ClinicalPipeline(self.cfg, self.device)
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, "unified_jepa_and_probe.pt")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            pipeline.context_encoder.load_state_dict(checkpoint["context_encoder_state"])
            pipeline.predictor.load_state_dict(checkpoint["predictor_state"])
            pipeline.linear_probe.load_state_dict(checkpoint["linear_probe_state"])
            print("🎯 Model checkpoint weights completely restored and synced.")
        else:
            print("⚠️ Checkpoint file missing. Running engine via random initialization.")
        
        pipeline.context_encoder.eval()
        pipeline.predictor.eval()
        pipeline.linear_probe.eval()
        return pipeline

    def calculate_latent_metrics(self, z_matrix: torch.Tensor):
        z_centered = z_matrix - z_matrix.mean(dim=0, keepdim=True)
        _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
        p = S / (S.sum() + 1e-10)
        eff_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-10))).item()
        sparsity_index = torch.mean(torch.norm(z_matrix, p=1, dim=1) / torch.norm(z_matrix, p=2, dim=1)).item()
        return eff_rank, sparsity_index

    def execute_evaluation_loop(self):
        pipeline = self._load_pipeline()
        print("⚡ Processing population arrays and intercepting gradient pathways...")

        def captum_forward(f, v, c, t, s_mask):
            z = pipeline.context_encoder(f, v, c, t, s_mask)
            return pipeline.linear_probe(pipeline.predictor(z))

        target_layer = list(pipeline.context_encoder.children())[0]
        lig = LayerIntegratedGradients(captum_forward, target_layer)

        attn_maps = []
        def attn_hook(m, i, o):
            if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                attn_maps.append(o[1].cpu().numpy())

        def patch_attention_modules(module):
            if isinstance(module, torch.nn.MultiheadAttention):
                orig_forward = module.forward
                def wrapped_forward(*args, **kwargs):
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = True
                    return orig_forward(*args, **kwargs)
                module.forward = wrapped_forward

        for m in pipeline.context_encoder.modules(): patch_attention_modules(m)
        target_attn_block = pipeline.context_encoder.temporal_backbone.layers[0].self_attn

        z_slots_accum, y_accum = [], []
        cohort_counterfactual_deltas = []
        cohort_ig_curves, cohort_attn_maps = [], []

        # 📊 TELEMETRY INITIALIZATION
        total_batches = len(self.val_loader)
        total_samples = len(self.val_loader.dataset)
        processed_samples = 0
        loop_start_time = time.perf_counter()

        for batch_idx, batch in enumerate(self.val_loader):
            # 🛡️ NO-GRAD ACCELERATION PASS FOR GLOBAL FEATURES
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = pipeline.process_batch(batch, self.device)
            
            z_slots_accum.append(out['z_hat_slots'].detach().float().cpu())  
            y_accum.append(out['multi_hot_targets'].cpu())

            f_ids_b = batch['feature_ids'].to(self.device)
            v_nums_b = batch['numeric_values'].to(self.device)
            c_ids_b = batch['cat_result_ids'].to(self.device)
            times_b = batch['timestamps'].to(self.device)
            s_mask_b = batch['student_mask'].to(self.device)

            # Counterfactual Risk Pass
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    orig_probs = torch.sigmoid(captum_forward(f_ids_b, v_nums_b, c_ids_b, times_b, s_mask_b))[:, self.target_class].cpu().numpy()
                    f_ids_mod = f_ids_b.clone()
                    f_ids_mod[:, f_ids_mod.size(1)//2:] = 0  
                    mod_probs = torch.sigmoid(captum_forward(f_ids_mod, v_nums_b, c_ids_b, times_b, s_mask_b))[:, self.target_class].cpu().numpy()
            cohort_counterfactual_deltas.extend((mod_probs - orig_probs) * 100)

            # Single-sample micro-loops process items safely without graph bloat
            for p_idx in range(f_ids_b.size(0)):
                f_s, v_s, c_s = f_ids_b[p_idx:p_idx+1], v_nums_b[p_idx:p_idx+1], c_ids_b[p_idx:p_idx+1]
                t_s, m_s = times_b[p_idx:p_idx+1], s_mask_b[p_idx:p_idx+1]

                ig_attr = lig.attribute(inputs=(f_s, v_s, c_s, t_s, m_s), target=self.target_class, n_steps=12, internal_batch_size=2)[0].detach().cpu().numpy()
                cohort_ig_curves.append(np.sum(np.abs(ig_attr), axis=-1))

                attn_maps.clear()
                handle = target_attn_block.register_forward_hook(attn_hook)
                with torch.no_grad(): 
                    _ = captum_forward(f_s, v_s, c_s, t_s, m_s)
                handle.remove()
                if attn_maps:
                    heatmap_data = attn_maps[0][0]
                    if heatmap_data.ndim == 3: heatmap_data = np.mean(heatmap_data, axis=0)
                    cohort_attn_maps.append(heatmap_data)

            # 📈 HEARTBEAT COMPILATION
            processed_samples += f_ids_b.size(0)
            
            if batch_idx % 50 == 0 or batch_idx == total_batches - 1:
                elapsed_time = time.perf_counter() - loop_start_time
                completion_ratio = processed_samples / total_samples
                estimated_total_time = elapsed_time / completion_ratio if completion_ratio > 0 else 0.0
                eta_minutes = (estimated_total_time - elapsed_time) / 60.0
                
                print(
                    f"⏳ [XAI MANIFOLD HARVEST] Batch {batch_idx:04d}/{total_batches:04d} │ "
                    f"Samples: {processed_samples:,}/{total_samples:,} ({completion_ratio*100:.1f}%) │ "
                    f"Elapsed: {elapsed_time/60.:.1f}m │ ETA: {eta_minutes:.1f}m"
                )

            if processed_samples >= 5000:
                print(f"\n🛑 [SAMPLE CAP REACHED] Harvested {processed_samples:,} high-dimensional sequences.")
                print("⏭️ Bypassing remaining cohort rows and jumping straight to visualization compilation...")
                break

        z_slots = torch.cat(z_slots_accum, dim=0) 
        y_cohort = torch.cat(y_accum, dim=0).numpy() 
        z_flattened = z_slots.view(z_slots.size(0), -1) 
        z_mean_pooled = z_slots.mean(dim=1)             

        eff_rank, sparsity_index = self.calculate_latent_metrics(z_mean_pooled)
        print(f"\n📊 COHORT LATENT QUANTIZATION METRICS:")
        print(f"  • Manifold Effective Rank: {eff_rank:.2f} / {self.cfg.latent_dim}")
        print(f"  • Layer Representation Sparsity Index: {sparsity_index:.4f}")

        probe_linear = next(m for m in pipeline.linear_probe.modules() if isinstance(m, torch.nn.Linear))
        blueprint_weights = probe_linear.weight[self.target_class].detach().cpu().numpy().reshape(self.cfg.num_slots, self.cfg.latent_dim)

        self._render_all_exports(
            z_flattened.numpy(), z_mean_pooled, y_cohort, blueprint_weights, 
            np.mean(cohort_ig_curves, axis=0), np.mean(cohort_attn_maps, axis=0), 
            cohort_counterfactual_deltas, eff_rank
        )

    def _render_all_exports(self, z_flat, z_pooled, y_cohort, blueprint, mean_ig, mean_attn, cf_deltas, eff_rank):
        print("\n🖼️ Compiling consolidated presentation sheets to disk...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        axes[0].fill_between(range(len(mean_ig)), mean_ig, color="#8e44ad", alpha=0.15)
        axes[0].plot(mean_ig, color="#8e44ad", linewidth=1.5, marker='o', markersize=3)
        axes[0].set_title("Cohort-Mean Integrated Gradients Attribution", fontweight='bold')
        axes[0].grid(True, linestyle=":")
        
        sns.heatmap(mean_attn, cmap="crest", ax=axes[1])
        axes[1].set_title("Cohort-Mean Layer 0 Attention Matrix Routing", fontweight='bold')
        plt.tight_layout(); plt.savefig("./xai_exports/unified_local_diagnostics.png", dpi=300); plt.close()

        TOP_N_CONDITIONS = 8
        class_frequencies = y_cohort.sum(axis=0)
        top_class_indices = np.argsort(class_frequencies)[::-1][:TOP_N_CONDITIONS]
        icd_mapping = {idx: self.decoder.id_to_icd.get(str(idx), f"ICD-{idx}") for idx in top_class_indices}

        f_z, labels = [], []
        for i in range(len(z_flat)):
            matches = [idx for idx in top_class_indices if y_cohort[i, idx] == 1.0]
            if matches: f_z.append(z_flat[i]); labels.append(icd_mapping[matches[0]])
            
        p_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        p_umap = p_reducer.fit_transform(np.array(f_z))
        df_p = pd.DataFrame({'UMAP 1': p_umap[:, 0], 'UMAP 2': p_umap[:, 1], 'ICD Diagnosis': labels})

        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=df_p, x='UMAP 1', y='UMAP 2', hue='ICD Diagnosis', style='ICD Diagnosis', palette='Set1', s=55)
        plt.title(f"T-JEPA Cross-Modal Latent Patient Topology Manifold", fontweight='bold')
        plt.tight_layout(); plt.savefig("./xai_exports/global_patient_manifold.png", dpi=300); plt.close()

        plt.figure(figsize=(8, 4))
        sns.histplot(cf_deltas, kde=True, color="#e74c3c", bins=40, edgecolor='white', alpha=0.7)
        plt.axvline(0, color='black', linewidth=1.2, linestyle='--')
        plt.title("Population Counterfactual Risk Modulation Spectrum", fontweight='bold')
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.savefig("./xai_exports/population_counterfactual_spectrum.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 4.2))
        sns.heatmap(blueprint, cmap="vlag", center=0)
        plt.title(f"Linear Probe Parametric Weight Structural Blueprint (Track ID: {self.target_class})", fontweight='bold')
        plt.tight_layout()
        plt.savefig("./xai_exports/probe_blueprint.png", dpi=300)
        plt.close()
        print("\n🎉 Comprehensive analytical evaluation complete. Asset path -> ./xai_exports/")

if __name__ == "__main__":
    cfg = CardioConfig()
    cfg.val_csv_path = "val_patient_flattened.csv"
    cfg.batch_size = 64  # Safeguards execution profile
    
    engine = AdvancedClinicalAnalyticsEngine(cfg)
    engine.execute_evaluation_loop()