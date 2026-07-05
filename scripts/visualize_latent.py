import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from torch.utils.data import DataLoader
from captum.attr import LayerIntegratedGradients

from config import CardioConfig
from src.TimelineDataset import BVTDTimelineDataset
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
        
        # Ingest codebooks for string mapping lookups
        with open(cfg.codebook_json_path, "r", encoding="utf-8") as f:
            self.codebooks = json.load(f)
        self.id_to_icd = self.codebooks.get("inverse_icd_codes", {})
        self.decoder = ClinicalDecoder(cfg.codebook_json_path)

        # Initialize dataset loader channels
        self.val_loader = DataLoader(
            BVTDTimelineDataset(cfg.val_csv_path, max_seq_len=cfg.max_sequence_len, max_targets=cfg.max_targets), 
            batch_size=cfg.batch_size, shuffle=False
        )
        
        self.split_idx = max(1, int(cfg.max_sequence_len * 0.70))
        self.target_class = 16  # Primary target diagnostic track for local explainability

    def _load_pipeline(self):
        print("🏭 Instantiating Clinical Pipeline Core Framework...")
        pipeline = ClinicalPipeline(self.cfg, self.device)
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, "unified_jepa_and_probe.pt")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            pipeline.context_encoder.load_state_dict(checkpoint["context_encoder_state"])
            pipeline.predictor.load_state_dict(checkpoint["predictor_state"])
            if "target_encoder_state" in checkpoint and pipeline.target_encoder is not None:
                pipeline.target_encoder.load_state_dict(checkpoint["target_encoder_state"])
            pipeline.linear_probe.load_state_dict(checkpoint["linear_probe_state"])
            print("🎯 Model checkpoint weights completely restored and synced.")
        else:
            print("⚠️ Checkpoint file missing. Running engine via random seed fields.")
        
        pipeline.context_encoder.eval()
        pipeline.predictor.eval()
        pipeline.linear_probe.eval()
        return pipeline

    def calculate_latent_metrics(self, z_matrix: torch.Tensor):
        """⚡ Evaluates mathematical constraints across the hidden representation layer."""
        # 1. Effective Rank Matrix Analysis
        z_centered = z_matrix - z_matrix.mean(dim=0, keepdim=True)
        _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
        p = S / (S.sum() + 1e-10)
        eff_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-10))).item()
        
        # 2. Activity Sparsity Index (Normalized L1 Norm)
        sparsity_index = torch.mean(torch.norm(z_matrix, p=1, dim=1) / torch.norm(z_matrix, p=2, dim=1)).item()
        return eff_rank, sparsity_index

    def execute_evaluation_loop(self):
        pipeline = self._load_pipeline()
        print("⚡ Processing population arrays and intercepting gradient pathways...")

        def forward_wrapper(f, v, c, t, p):
            z = pipeline.context_encoder(f, v, c, t, p)
            return pipeline.linear_probe(pipeline.predictor(z))

        # Setup Layer Integrated Gradients
        target_layer = list(pipeline.context_encoder.children())[0]
        lig = LayerIntegratedGradients(forward_wrapper, target_layer)

        # Hook config for intercepting self-attention matrices safely
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

        for m in pipeline.context_encoder.modules():
            patch_attention_modules(m)
        target_attn_block = pipeline.context_encoder.temporal_backbone.layers[0].self_attn

        # Multi-sample diagnostic registers
        z_slots_accum, y_accum = [], []
        cohort_counterfactual_deltas = []
        cohort_ig_curves, cohort_attn_maps = [], []

        for batch_idx, batch in enumerate(self.val_loader):
            out = pipeline.process_batch(batch, self.device)
            z_slots_accum.append(out['z_hat_slots'].detach().cpu())
            y_accum.append(out['multi_hot_targets'].cpu())

            # Isolate timeline parameters
            f_ids_b = batch['feature_ids'][:, :self.split_idx].to(self.device)
            v_nums_b = batch['numeric_values'][:, :self.split_idx].to(self.device)
            c_ids_b = batch['cat_result_ids'][:, :self.split_idx].to(self.device)
            times_b = batch['timestamps'][:, :self.split_idx].to(self.device)
            p_mask_b = batch['padding_mask'][:, :self.split_idx].to(self.device)

            # Fast Batch Counterfactual Pass
            with torch.no_grad():
                orig_probs = torch.sigmoid(forward_wrapper(f_ids_b, v_nums_b, c_ids_b, times_b, p_mask_b))[:, self.target_class].cpu().numpy()
                f_ids_mod = f_ids_b.clone()
                f_ids_mod[:, f_ids_mod.size(1)//2:] = 0
                mod_probs = torch.sigmoid(forward_wrapper(f_ids_mod, v_nums_b, c_ids_b, times_b, p_mask_b))[:, self.target_class].cpu().numpy()
            cohort_counterfactual_deltas.extend((mod_probs - orig_probs) * 100)

            # Micro-Batch loop for safe gradient integrations
            for p_idx in range(f_ids_b.size(0)):
                f_s, v_s, c_s = f_ids_b[p_idx:p_idx+1], v_nums_b[p_idx:p_idx+1], c_ids_b[p_idx:p_idx+1]
                t_s, p_s = times_b[p_idx:p_idx+1], p_mask_b[p_idx:p_idx+1]

                ig_attr = lig.attribute(inputs=(f_s, v_s, c_s, t_s, p_s), target=self.target_class, n_steps=15, internal_batch_size=2)[0].detach().cpu().numpy()
                cohort_ig_curves.append(np.sum(np.abs(ig_attr), axis=-1))

                attn_maps.clear()
                handle = target_attn_block.register_forward_hook(attn_hook)
                with torch.no_grad():
                    _ = forward_wrapper(f_s, v_s, c_s, t_s, p_s)
                handle.remove()
                if attn_maps:
                    heatmap_data = attn_maps[0][0]
                    if heatmap_data.ndim == 3: heatmap_data = np.mean(heatmap_data, axis=0)
                    cohort_attn_maps.append(heatmap_data)

        # Concatenate raw structural cohort tensors
        z_slots = torch.cat(z_slots_accum, dim=0) # [N, Num_Slots, Latent_Dim]
        y_cohort = torch.cat(y_accum, dim=0).numpy() # [N, Num_ICD_Classes]
        
        # Derive structural pool variations natively
        z_flattened = z_slots.view(z_slots.size(0), -1) # [N, 4096] for Patient UMAP
        z_mean_pooled = z_slots.mean(dim=1)             # [N, 512] for Label Space Matrix

        # Compute extended mathematical properties
        eff_rank, sparsity_index = self.calculate_latent_metrics(z_mean_pooled)
        print(f"\n📊 COHORT LATENT QUANTIZATION METRICS:")
        print(f"  • Manifold Effective Rank: {eff_rank:.2f} / {self.cfg.latent_dim}")
        print(f"  • Layer Representation Sparsity Index (L1 Activity Scale): {sparsity_index:.4f}")

        # Extract Probe Blueprint Weights
        probe_linear = next(m for m in pipeline.linear_probe.modules() if isinstance(m, torch.nn.Linear))
        blueprint_weights = probe_linear.weight[self.target_class].detach().cpu().numpy().reshape(self.cfg.num_slots, self.cfg.latent_dim)

        self._render_all_exports(
            z_flattened.numpy(), z_mean_pooled, y_cohort, blueprint_weights, 
            np.mean(cohort_ig_curves, axis=0), np.mean(cohort_attn_maps, axis=0), 
            cohort_counterfactual_deltas, eff_rank
        )

    def _render_all_exports(self, z_flat, z_pooled, y_cohort, blueprint, mean_ig, mean_attn, cf_deltas, eff_rank):
        print("\n🖼️ Compiling consolidated presentation sheets to disk...")

        # 📈 FIGURE 1: Unified Local Explanation Diagnostic Sheet (IG & Attention)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        axes[0].fill_between(range(len(mean_ig)), mean_ig, color="#8e44ad", alpha=0.15)
        axes[0].plot(mean_ig, color="#8e44ad", linewidth=1.5, marker='o', markersize=3)
        axes[0].set_title("Cohort-Mean Integrated Gradients Attribution", fontweight='bold')
        axes[0].set_xlabel("Timeline Events Schedule Step"); axes[0].set_ylabel("Causal Blame Scale")
        axes[0].grid(True, linestyle=":")
        
        sns.heatmap(mean_attn, cmap="crest", ax=axes[1], cbar_kws={'label': 'Attention Weight Magnitude'})
        axes[1].set_title("Cohort-Mean Layer 0 Attention Matrix Routing", fontweight='bold')
        axes[1].set_xlabel("Key Memory Timeline Target"); axes[1].set_ylabel("Query Vector Coordinates")
        plt.tight_layout()
        plt.savefig("./xai_exports/unified_local_diagnostics.png", dpi=300); plt.close()

        # 📊 FIGURE 2: Multi-Class Patient Topology Manifold Scatter Plot
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
        df_p['ICD Diagnosis'] = pd.Categorical(df_p['ICD Diagnosis'], categories=[icd_mapping[idx] for idx in top_class_indices], ordered=True)

        plt.figure(figsize=(10, 7))
        plt.grid(True, linestyle="--", alpha=0.4, color="#dcdde1", zorder=0)
        sns.scatterplot(data=df_p.sort_values('ICD Diagnosis'), x='UMAP 1', y='UMAP 2', hue='ICD Diagnosis', style='ICD Diagnosis', palette='Set1', s=55, alpha=0.85, edgecolor='w', linewidth=0.4, zorder=3)
        plt.title(f"T-JEPA Cross-Modal Latent Patient Topology Manifold\n(Validation Cohort Slices - Top {TOP_N_CONDITIONS} Dynamic Conditions)", fontsize=11, fontweight='bold', pad=12)
        plt.legend(title='ICD Diagnosis', title_fontproperties={'weight': 'bold'}, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, edgecolor='#bdc3c7')
        plt.tight_layout(); plt.savefig("./xai_exports/global_patient_manifold.png", dpi=300); plt.close()

        # 🗺️ FIGURE 3: Clinical Label Prototype Taxonomy Map (From visualize_latent.py)
        prototypes, valid_classes = [], []
        for c_idx in range(y_cohort.shape[1]):
            pos_mask = y_cohort[:, c_idx] == 1.0
            if pos_mask.sum() >= 2:
                prototypes.append(z_pooled[pos_mask].mean(dim=0).unsqueeze(0))
                valid_classes.append(c_idx)

        if len(prototypes) >= 3:
            proto_tensor = torch.cat(prototypes, dim=0).numpy()
            
            # Metric tracking: Compute Inter-Class Discrepancy Index
            from sklearn.metrics.pairwise import cosine_distances
            discrepancy_idx = np.mean(cosine_distances(proto_tensor))
            print(f"  • Global Cross-Modal Label Discrepancy Index: {discrepancy_idx:.4f} (Higher = Better Category Separation)")

            l_reducer = umap.UMAP(n_neighbors=min(15, len(proto_tensor) - 1), min_dist=0.1, metric='cosine', random_state=42)
            l_umap = l_reducer.fit_transform(proto_tensor)
            icd_strings = [self.id_to_icd.get(str(idx), f"C_{idx}") for idx in valid_classes]
            chapters = [c[0] if (c and c[0].isalpha()) else "Other" for c in icd_strings]

            df_l = pd.DataFrame({'Component 1': l_umap[:, 0], 'Component 2': l_umap[:, 1], 'ICD': icd_strings, 'Chapter': chapters})
            plt.figure(figsize=(10, 8))
            ax = sns.scatterplot(data=df_l, x='Component 1', y='Component 2', hue='Chapter', palette="tab20", alpha=0.85, edgecolor='black', linewidth=0.4, s=50)
            
            # Apply text annotations to the first 25 items cleanly
            for k, row in df_l.iterrows():
                if k < 25 and row['Chapter'] != 'Other':
                    ax.text(row['Component 1'] + 0.03, row['Component 2'] + 0.03, row['ICD'], fontsize=7, fontweight='semibold', alpha=0.8)
            
            plt.title(f"Model-Perceived Disease Prototype Topology Map (UMAP Space)\nClustered by Broad ICD-10 Taxonomy Chapters | Manifold Rank: {eff_rank:.2f}", fontweight='bold', fontsize=11, pad=12)
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="ICD-10 Chapter Roots", title_fontproperties={'weight': 'bold'})
            plt.tight_layout(); plt.savefig("./xai_exports/global_disease_prototype_topology.png", dpi=300); plt.close()

        # 📈 FIGURE 4: Population Counterfactual Spectrum Histogram
        plt.figure(figsize=(8, 4))
        sns.histplot(cf_deltas, kde=True, color="#e74c3c", bins=40, edgecolor='white', alpha=0.7)
        plt.axvline(0, color='black', linewidth=1.2, linestyle='--')
        plt.title("Population Counterfactual Risk Modulation Spectrum", fontweight='bold', fontsize=11)
        plt.xlabel("Risk Probability Delta Shift Shift (%)"); plt.ylabel("Patient Frequency Profile Record Count")
        plt.grid(True, linestyle=":", alpha=0.6); plt.tight_layout()
        plt.savefig("./xai_exports/population_counterfactual_spectrum.png", dpi=300); plt.close()

        # 📊 FIGURE 5: Linear Probe Structural Blueprint Heatmap
        plt.figure(figsize=(10, 4.2))
        sns.heatmap(blueprint, cmap="vlag", center=0, cbar_kws={'label': 'Parametric Weight Values'})
        plt.title(f"Linear Probe Parametric Weight Structural Blueprint (Track ID: {self.target_class})", fontweight='bold', fontsize=11)
        plt.xlabel("Latent Hidden Dimension Channels Axis"); plt.ylabel("Perceiver Bottleneck Slots Registry")
        plt.tight_layout(); plt.savefig("./xai_exports/probe_blueprint.png", dpi=300); plt.close()

        print("\n🎉 Comprehensive analytical evaluation complete. All high-fidelity assets exported cleanly to -> ./xai_exports/")

if __name__ == "__main__":
    engine = AdvancedClinicalAnalyticsEngine(CardioConfig())
    engine.execute_evaluation_loop()