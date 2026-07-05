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

# 🎛️ SYSTEM SETUP
cfg = CardioConfig()
device = cfg.device
os.makedirs("./xai_exports", exist_ok=True)
sns.set_theme(style="white")

# 📥 1. PIPELINE INITIALIZATION & WEIGHTS RESTORATION
print("🏭 Initializing Clinical Pipeline & Restoring Weights...")
pipeline = ClinicalPipeline(cfg, device)
decoder = ClinicalDecoder(cfg.codebook_json_path)
checkpoint_path = os.path.join(cfg.checkpoint_dir, "unified_jepa_and_probe.pt")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pipeline.context_encoder.load_state_dict(checkpoint["context_encoder_state"])
    pipeline.predictor.load_state_dict(checkpoint["predictor_state"])
    if "target_encoder_state" in checkpoint and pipeline.target_encoder is not None:
        pipeline.target_encoder.load_state_dict(checkpoint["target_encoder_state"])
    pipeline.linear_probe.load_state_dict(checkpoint["linear_probe_state"])
    print("🎯 Model weights fully synced.")
else:
    print("⚠️ Checkpoint missing. Running random initialization.")

# 📦 2. REAL DATASET PIPELINE LOADING
print("📖 Loading live validation records...")
val_loader = DataLoader(
    BVTDTimelineDataset(cfg.val_csv_path, max_seq_len=cfg.max_sequence_len, max_targets=cfg.max_targets), 
    batch_size=cfg.batch_size, shuffle=False
)

# Establish uniform chronological slicing across the global cohort configuration
split_idx = max(1, int(cfg.max_sequence_len * 0.70))
target_class = 16 

# ───────────────────────────────────────────────────────────────────────
# 🔬 3. POPULATION-WIDE MULTI-SAMPLE EVALUATION LOOP
# ───────────────────────────────────────────────────────────────────────
print("⚡ Executing unified attribution loops across entire validation cohort...")

# Explicit forward wrapper for Captum gradient graph tracking paths
def forward_wrapper(f, v, c, t, p):
    z = pipeline.context_encoder(f, v, c, t, p)
    return pipeline.linear_probe(pipeline.predictor(z))

# Instantiate Captum Layer Integrated Gradients engine
target_layer = list(pipeline.context_encoder.children())[0]
lig = LayerIntegratedGradients(forward_wrapper, target_layer)

# Core collection containers for multi-sample aggregation
z_list, y_list = [], []
cohort_counterfactual_deltas = []
cohort_ig_curves = []
cohort_attn_maps = []

# 🧬 Explicit Scope Attention Interceptor Configuration
attn_maps = []
def attn_hook(m, i, o): 
    if isinstance(o, tuple) and len(o) > 1 and o[1] is not None: 
        attn_maps.append(o[1].cpu().numpy())

def patch_multihead_attention(attn_module):
    orig_forward = attn_module.forward
    def wrapped_forward(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        return orig_forward(*args, **kwargs)
    attn_module.forward = wrapped_forward

for m in pipeline.context_encoder.modules():
    if isinstance(m, torch.nn.MultiheadAttention):
        patch_multihead_attention(m)

target_attn_block = pipeline.context_encoder.temporal_backbone.layers[0].self_attn

# Process all samples across the validation data loader space
for batch_idx, batch in enumerate(val_loader):
    print(f"  • Processing Batch {batch_idx + 1}/{len(val_loader)}...")
    
    # Pathway A: Extract Hidden Representation Coordinates for Global Manifold
    out = pipeline.process_batch(batch, device)
    z_list.append(out['z_hat_slots'].view(out['z_hat_slots'].size(0), -1).detach().cpu().numpy())
    y_list.append(out['multi_hot_targets'].cpu().numpy())
    
    # Extract structural timelines for current processing block
    f_ids_b = batch['feature_ids'][:, :split_idx].to(device)
    v_nums_b = batch['numeric_values'][:, :split_idx].to(device)
    c_ids_b = batch['cat_result_ids'][:, :split_idx].to(device)
    times_b = batch['timestamps'][:, :split_idx].to(device)
    p_mask_b = batch['padding_mask'][:, :split_idx].to(device)
    
    # Pathway B: Multi-Sample Batch Counterfactual Analysis (Fast Execution Track)
    with torch.no_grad():
        orig_probs = torch.sigmoid(forward_wrapper(f_ids_b, v_nums_b, c_ids_b, times_b, p_mask_b))[:, target_class].cpu().numpy()
        f_ids_mod = f_ids_b.clone()
        f_ids_mod[:, f_ids_mod.size(1)//2:] = 0
        mod_probs = torch.sigmoid(forward_wrapper(f_ids_mod, v_nums_b, c_ids_b, times_b, p_mask_b))[:, target_class].cpu().numpy()
    cohort_counterfactual_deltas.extend((mod_probs - orig_probs) * 100)

    # Pathway C: Micro-Slicing Patient-by-Patient for VRAM-Safe IG and Attention Map Extraction
    for p_idx in range(f_ids_b.size(0)):
        f_single = f_ids_b[p_idx:p_idx+1]
        v_single = v_nums_b[p_idx:p_idx+1]
        c_single = c_ids_b[p_idx:p_idx+1]
        t_single = times_b[p_idx:p_idx+1]
        p_single = p_mask_b[p_idx:p_idx+1]
        
        # Calculate individual patient Captum Layer Integrated Gradients
        ig_attr = lig.attribute(
            inputs=(f_single, v_single, c_single, t_single, p_single),
            target=target_class,
            n_steps=20,                 # Step scale optimized for group performance loops
            internal_batch_size=2      
        )[0].detach().cpu().numpy()
        
        cohort_ig_curves.append(np.sum(np.abs(ig_attr), axis=-1))
        
        # Intercept Multi-Head Self-Attention Matrix
        attn_maps.clear()
        hook_handle = target_attn_block.register_forward_hook(attn_hook)
        with torch.no_grad():
            _ = forward_wrapper(f_single, v_single, c_single, t_single, p_single)
        hook_handle.remove()
        
        if attn_maps:
            heatmap_data = attn_maps[0][0]
            if heatmap_data.ndim == 3:
                heatmap_data = np.mean(heatmap_data, axis=0)
            cohort_attn_maps.append(heatmap_data)

# Compile global arrays across all extracted list collections
z_cohort = np.concatenate(z_list, axis=0)
y_cohort = np.concatenate(y_list, axis=0)
mean_ig_timeline = np.mean(cohort_ig_curves, axis=0)
mean_attention_matrix = np.mean(cohort_attn_maps, axis=0)

# 🧬 PATHWAY 2: Linear Probe Parametric Weights Slicing
probe_linear_layer = next(m for m in pipeline.linear_probe.modules() if isinstance(m, torch.nn.Linear))
blueprint_weights = probe_linear_layer.weight[target_class].detach().cpu().numpy().reshape(cfg.num_slots, cfg.latent_dim)

# 🗺️ PATHWAY 4: Global Cohort UMAP Space Reductions
print("🗺️ Compiling global latent manifold space across all validation records...")
TOP_N_CONDITIONS = 8
class_frequencies = y_cohort.sum(axis=0)
top_class_indices = np.argsort(class_frequencies)[::-1][:TOP_N_CONDITIONS]
icd_code_mapping = {idx: decoder.id_to_icd.get(str(idx), f"ICD-{idx}") for idx in top_class_indices}

filtered_z, assigned_labels = [], []
for i in range(len(z_cohort)):
    patient_targets = y_cohort[i]
    active_top_classes = [idx for idx in top_class_indices if patient_targets[idx] == 1.0]
    if active_top_classes:
        filtered_z.append(z_cohort[i])
        assigned_labels.append(icd_code_mapping[active_top_classes[0]])

filtered_z = np.array(filtered_z)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
z_umap = reducer.fit_transform(filtered_z)

# ───────────────────────────────────────────────────────────────────────
# 🎨 4. COMPACT RENDERING TO PNG EXPORTS
# ───────────────────────────────────────────────────────────────────────
print("🖼️ Printing diagnostic population sheets to disk...")

# Export 1: Cohort-Average Integrated Gradients Matrix
plt.figure(figsize=(9, 3.5))
plt.fill_between(range(len(mean_ig_timeline)), mean_ig_timeline, color="#8e44ad", alpha=0.2)
plt.plot(mean_ig_timeline, color="#8e44ad", linewidth=1.5, marker='o', markersize=4)
plt.title(f"Cohort-Mean Captum Layer Integrated Gradients (Track ID: {target_class})", fontweight='bold')
plt.xlabel("Timeline Steps"); plt.ylabel("Mean Causal Weight"); plt.tight_layout(); plt.savefig("./xai_exports/captum_ig.png", dpi=300); plt.close()

# Export 2: Cohort-Average Attention Map Matrix
plt.figure(figsize=(7, 6))
sns.heatmap(mean_attention_matrix, cmap="crest")
plt.title("Cohort-Mean Self-Attention Matrix (Layer 0 Structural Flow)", fontweight='bold')
plt.xlabel("Key Timeline Axis"); plt.ylabel("Query Timeline Axis"); plt.tight_layout(); plt.savefig("./xai_exports/attention_matrix.png", dpi=300); plt.close()

# Export 3: Linear Probe Weights Matrix Blueprint
plt.figure(figsize=(10, 4))
sns.heatmap(blueprint_weights, cmap="vlag", center=0)
plt.title(f"Linear Probe Weight Structural Blueprint (Track ID: {target_class})", fontweight='bold')
plt.xlabel("Latent Dim Channels"); plt.ylabel("Perceiver Slots"); plt.tight_layout(); plt.savefig("./xai_exports/probe_blueprint.png", dpi=300); plt.close()

# Export 4: UMAP Space Clustering
plot_df = pd.DataFrame({'UMAP Component 1': z_umap[:, 0], 'UMAP Component 2': z_umap[:, 1], 'ICD Diagnosis': assigned_labels})
plot_df['ICD Diagnosis'] = pd.Categorical(plot_df['ICD Diagnosis'], categories=[icd_code_mapping[idx] for idx in top_class_indices], ordered=True)
plot_df = plot_df.sort_values('ICD Diagnosis')

plt.figure(figsize=(11, 7.5))
plt.grid(True, linestyle="--", alpha=0.5, color="#dcdde1", zorder=0)
sns.scatterplot(data=plot_df, x='UMAP Component 1', y='UMAP Component 2', hue='ICD Diagnosis', style='ICD Diagnosis', palette='Set1', s=55, alpha=0.85, edgecolor='w', linewidth=0.4, zorder=3)
plt.title(f"T-JEPA Cross-Modal Latent Space Geometry\n(Validation Patient Cohort - Top {TOP_N_CONDITIONS} Active Conditions)", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("UMAP Component 1", fontweight='bold'); plt.ylabel("UMAP Component 2", fontweight='bold')
plt.legend(title='ICD Diagnosis', title_fontproperties={'weight': 'bold'}, bbox_to_anchor=(1.03, 1), loc='upper left', frameon=True, facecolor='white', edgecolor='#bdc3c7')
plt.tight_layout(); plt.savefig("./xai_exports/global_umap_multiclass.png", dpi=300); plt.close()

# Export 5: Cohort Counterfactual Delta Distribution (Upgraded to Density Histogram)
plt.figure(figsize=(8, 4))
sns.histplot(cohort_counterfactual_deltas, kde=True, color="#e74c3c", bins=40, edgecolor='white', alpha=0.7)
plt.axvline(0, color='black', linewidth=1.2, linestyle='--')
plt.title(f"Population Counterfactual Risk Modulation Spectrum (ICD Track: {target_class})", fontweight='bold')
plt.xlabel("Risk Probability Delta Shift Shift (%)")
plt.ylabel("Patient Frequency Count")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("./xai_exports/counterfactual.png", dpi=300)
plt.close()

print("\n🎉 All real-world XAI pipeline population metrics exported safely to -> ./xai_exports/")

# ───────────────────────────────────────────────────────────────────────
# 🔍 CARDIO SYSTEM DIAGNOSTIC REPORT 
# ───────────────────────────────────────────────────────────────────────
active_tracks = np.where(y_cohort.sum(axis=0) > 0)[0].tolist()
print("\n🔬 CARDIO SYSTEM DIAGNOSTIC REPORT:")
print(f"• Successfully aggregated local attributions across {len(y_cohort)} validation samples.")
print(f"• Active track indices available in this validation cohort: {active_tracks}")
# ───────────────────────────────────────────────────────────────────────