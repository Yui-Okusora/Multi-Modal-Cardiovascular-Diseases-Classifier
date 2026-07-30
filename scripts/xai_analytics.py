# scripts/xai_analytics.py
r"""
====================================================================================================
CHRONOS-JEPA ADVANCED CLINICAL XAI & MANIFOLD ANALYTICS ENGINE (FULL RECOVERED & RECALIBRATED)
====================================================================================================
"""

import os
import json
import time
import sys
from pathlib import Path

# 🚀 Dynamically append project root (Cardio/) to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import umap
from torch.utils.data import DataLoader
from captum.attr import LayerIntegratedGradients

from config import CardioConfig
from src.TimelineDataset import BVTDFlattenedDataset  
from src.ModelModules import compute_comprehensive_manifold_diagnostics, ClinicalDecoder
from src.LoRAWrapper import defactorize_entire_architecture
from Pipeline import ClinicalPipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 
logging.getLogger('matplotlib').setLevel(logging.WARNING)


class AdvancedClinicalAnalyticsEngine:
    def __init__(self, cfg: CardioConfig):
        self.cfg = cfg
        self.device = cfg.device
        os.makedirs(cfg.xai_export_dir, exist_ok=True)
        sns.set_theme(style="ticks")
        
        with open(cfg.codebook_json_path, "r", encoding="utf-8") as f:
            self.codebooks = json.load(f)
        self.id_to_icd = self.codebooks.get("inverse_icd_codes", {})
        self.id_to_token = self.codebooks.get("inverse_maps", {})
        self.decoder = ClinicalDecoder(cfg.codebook_json_path)

        self.val_loader = DataLoader(
            BVTDFlattenedDataset(
                cfg.val_csv_path, 
                max_seq_len=cfg.max_sequence_len, 
                max_targets=cfg.max_targets,
                is_train=False
            ), 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )

    def run_footprint_audit(self, pipeline: ClinicalPipeline):
        """
        🖥️ DYNAMIC MODEL FOOTPRINT AUDIT:
        Queries loaded modules to present an absolute parameter ledger 
        (trainable vs. untrainable) across production boundaries.
        """
        print("\n" + "="*80)
        print("🔍 DETAILED MODEL FOOTPRINT AUDIT: TRAINING VS. PRODUCTION CONFIGURATION")
        print("="*80)
        
        ctx_enc   = getattr(pipeline, 'context_encoder', None)
        tgt_enc   = getattr(pipeline, 'target_encoder', None)
        predictor = getattr(pipeline, 'predictor', None)
        assembler = getattr(pipeline, 'assembler', None)
        projector = getattr(pipeline, 'context_projector', None)
        probe     = getattr(pipeline, 'probe', None)       
        cardinal  = getattr(pipeline, 'cardinal', None)    
        
        def get_stats(m):
            if m is None or not isinstance(m, nn.Module):
                return 0, 0, 0
            total = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            untrainable = total - trainable
            return total, trainable, untrainable

        modules_map = {
            "Context (Inference) Encoder": ctx_enc,
            "Target Encoder (Teacher)": tgt_enc,
            "Predictor Network": predictor,
            "Manifold Assembler": assembler,
            "Projector Layer": projector,
            "Label Probe": probe,
            "Cardinality Head": cardinal
        }
        
        print("📊 INDIVIDUAL MODEL MODULES AUDIT:")
        print("-" * 80)
        for name, mod in modules_map.items():
            if mod is not None:
                tot, tr, untr = get_stats(mod)
                print(f"   • {name:<30} │ Total: {tot:12,} │ Trainable: {tr:12,} │ Untrainable: {untr:12,}")
            else:
                print(f"   • {name:<30} │ [Module Sibling Not Loaded/Defined]")
        print("-" * 80)

        all_training_modules = [ctx_enc, tgt_enc, predictor, projector, probe, cardinal, assembler]
        train_tot = sum(get_stats(m)[0] for m in all_training_modules)
        train_tr = sum(get_stats(m)[1] for m in all_training_modules)
        train_untr = train_tot - train_tr

        inference_modules = [ctx_enc, probe, cardinal, assembler]
        inf_tot = sum(get_stats(m)[0] for m in inference_modules)
        inf_tr = sum(get_stats(m)[1] for m in inference_modules)
        inf_untr = inf_tot - inf_tr

        print(f"🏋️‍♀️ FULL TRAINING CONFIGURATION (All Loaded Architecture):")
        print(f"   • Total Parameter Count:       {train_tot:,}")
        print(f"   • Trainable Parameter Count:   {train_tr:,}")
        print(f"   • Untrainable Parameter Count: {train_untr:,}")
        print("-" * 80)

        print(f"🚀 PRODUCTION/INFERENCE CONFIGURATION (Encoder + Probes Only):")
        print(f"   • Total Parameter Count:       {inf_tot:,}")
        print(f"   • Trainable Parameter Count:   {inf_tr:,}")
        print(f"   • Untrainable Parameter Count: {inf_untr:,}")
        print("-" * 80)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            baseline_vram = torch.cuda.memory_allocated() / (1024 ** 2)
            audit_batch = next(iter(self.val_loader))
            start_time = time.perf_counter()
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=self.cfg.amp_dtype, enabled=self.cfg.use_amp):
                    _ = pipeline.process_batch(audit_batch, self.device, run_teacher=False)
            
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            dynamic_allocated = peak_vram - baseline_vram
            
            print(f"⚡ REAL-TIME GPU EXECUTION PROFILE (Batch Size: {self.val_loader.batch_size}):")
            print(f"   • Baseline VRAM Allocated:   {baseline_vram:.2f} MB")
            print(f"   • Peak Runtime VRAM Limit:   {peak_vram:.2f} MB")
            print(f"   • Dynamic Batch Overhead:    {dynamic_allocated:.2f} MB")
            print(f"   • Forward Pass Latency:      {elapsed_ms:.2f} ms")
        else:
            print("🖥️ CUDA platform is unavailable. Skipping active hardware profiling.")
        print("="*80 + "\n")

    def compute_cohort_attention_routing_matrix(self, pipeline: ClinicalPipeline) -> np.ndarray:
        pipeline.context_encoder.eval()
        seq_len = self.cfg.max_sequence_len
        accum_routing = np.zeros((seq_len, seq_len), dtype=np.float32)
        total_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                f_ids  = batch['feature_ids'].to(self.device)
                v_nums = batch['numeric_values'].to(self.device)
                c_ids  = batch['cat_result_ids'].to(self.device)
                times  = batch['timestamps'].to(self.device)
                s_mask = batch['student_mask'].to(self.device)

                tokenizer = pipeline.context_encoder.tokenizer
                tokens = tokenizer(f_ids, v_nums, c_ids, times)
                
                if hasattr(pipeline.context_encoder, 'pos_embedder'):
                    tokens = pipeline.context_encoder.pos_embedder(tokens)

                norm_tokens = F.normalize(tokens.float(), p=2, dim=-1)
                batch_routing = torch.bmm(norm_tokens, norm_tokens.transpose(1, 2)).cpu().numpy()
                
                # Mask out padding tokens so padded pairs are strictly 0.0
                valid_mask = (~s_mask.cpu()).float().numpy()[:, :, None]
                batch_routing = batch_routing * valid_mask * np.transpose(valid_mask, (0, 2, 1))
                
                accum_routing += batch_routing.mean(axis=0)
                total_batches += 1
                if total_batches >= 25: break

        cohort_mean_routing = accum_routing / max(1, total_batches)
        return np.clip(cohort_mean_routing, 0.0, 1.0)

    @torch.no_grad()
    def plot_multi_basin_hopfield_energy(self, probe_module, output_path: str):
        if not hasattr(probe_module, 'prototype_memory') or not hasattr(probe_module, 'k_proj'):
            return
            
        beta_vis = self.cfg.xai_hopfield_beta_vis
        with torch.no_grad():
            K_proto = F.normalize(probe_module.k_proj(probe_module.prototype_memory), p=2, dim=-1).cpu().numpy()
            
        M, D = K_proto.shape
        pca = PCA(n_components=2)
        K_2d = pca.fit_transform(K_proto)
        
        u_min, u_max = K_2d[:, 0].min() - 0.5, K_2d[:, 0].max() + 0.5
        v_min, v_max = K_2d[:, 1].min() - 0.5, K_2d[:, 1].max() + 0.5
        
        grid_u, grid_v = np.meshgrid(np.linspace(u_min, u_max, 100), np.linspace(v_min, v_max, 100))
        grid_2d = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1)
        grid_512 = pca.inverse_transform(grid_2d)
        grid_512_norm = grid_512 / (np.linalg.norm(grid_512, axis=1, keepdims=True) + 1e-8)
        
        dot_products = (grid_512_norm @ K_proto.T) * np.sqrt(D)
        max_dots = np.max(beta_vis * dot_products, axis=1, keepdims=True)
        lse = max_dots + np.log(np.sum(np.exp(beta_vis * dot_products - max_dots), axis=1, keepdims=True))
        
        pure_interaction_energy = -(1.0 / beta_vis) * lse.reshape(100, 100)
        pure_interaction_energy -= pure_interaction_energy.max()
        
        fig = plt.figure(figsize=(10, 8), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(grid_u, grid_v, pure_interaction_energy, cmap='viridis_r', alpha=0.85, edgecolor='none')
        
        proto_dots = (K_proto @ K_proto.T) * np.sqrt(D)
        p_max_dots = np.max(beta_vis * proto_dots, axis=1, keepdims=True)
        p_lse = p_max_dots + np.log(np.sum(np.exp(beta_vis * proto_dots - p_max_dots), axis=1, keepdims=True))
        proto_depths = -(1.0 / beta_vis) * p_lse.flatten()
        proto_depths -= pure_interaction_energy.max()
        
        ax.scatter(K_2d[:, 0], K_2d[:, 1], proto_depths, color='#e74c3c', s=35, zorder=10)
        ax.set_title(f"Probe Multi-Basin Hopfield Energy Landscape (Isolated $\\Delta E$ | $\\beta_{{vis}}={beta_vis}$)", fontweight='bold')
        ax.set_xlabel("Principal Archetype Axis 1 (u)")
        ax.set_ylabel("Principal Archetype Axis 2 (v)")
        ax.set_zlabel("Relative Attractor Depth $\\Delta E(q)$")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Attractor Energy Level $\\Delta E$")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def execute_evaluation_loop(self):
        pipeline = ClinicalPipeline(self.cfg, self.device)
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, self.cfg.unified_final_filename)
            
        pipeline.load_checkpoint(checkpoint_path=checkpoint_path)
        defactorize_entire_architecture(pipeline.context_encoder)
        for param in pipeline.context_encoder.parameters():
            param.requires_grad = True

        self.run_footprint_audit(pipeline)

        print("⚡ Processing population arrays and intercepting gradient pathways...")

        def captum_forward(f, v, c, t, s_mask, age, gender):
            curr_b = f.size(0)
            batch_dict = {
                'patient_session_id': ["anal_pat"] * curr_b,
                'feature_ids': f, 
                'numeric_values': v, 
                'cat_result_ids': c, 
                'timestamps': t, 
                'base_mask': s_mask,
                'student_mask': s_mask,
                'age': age, 
                'gender': gender,
                'icd_targets': torch.zeros(curr_b, 1, dtype=torch.long, device=f.device),
                'target_mask': torch.zeros(curr_b, 1, dtype=torch.bool, device=f.device)
            }
            out = pipeline.process_batch(batch_dict, f.device, run_teacher=False)
            return (out['predicted_cardinalities'] + torch.sigmoid(out['logits']).sum(dim=-1)).unsqueeze(-1)

        tokenizer_mod = pipeline.context_encoder.tokenizer
        target_layers = [
            tokenizer_mod.feature_embedding,      
            tokenizer_mod.numeric_encoder,           
            tokenizer_mod.text_encoder,   
            tokenizer_mod.time_embedder           
        ]
        
        lig = LayerIntegratedGradients(captum_forward, target_layers)

        z_slots_accum, y_accum = [], []
        cohort_counterfactual_deltas = []
        cohort_global_severity_scores = []

        seq_len = self.cfg.max_sequence_len
        chronological_stream_attributions = {
            "feature_id": np.zeros(seq_len), "numeric_value": np.zeros(seq_len),
            "categorical_result": np.zeros(seq_len), "timestamp": np.zeros(seq_len)
        }

        processed_samples = 0
        total_batches = len(self.val_loader)
        total_samples = len(self.val_loader.dataset)
        loop_start_time = time.perf_counter()

        print(f"🚀 Initializing Validation Stream ({total_samples} samples across {total_batches} batches)...")

        for batch_idx, batch in enumerate(self.val_loader):
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=self.cfg.amp_dtype, enabled=self.cfg.use_amp):
                    out = pipeline.process_batch(batch, self.device, run_teacher=False)
            
            z_slots_accum.append(out['z_c_slots'].detach().float().cpu())
            y_accum.append(out['multi_hot_targets'].cpu())

            f_ids_b  = batch['feature_ids'].to(self.device)
            v_nums_b = batch['numeric_values'].to(self.device)
            c_ids_b  = batch['cat_result_ids'].to(self.device)
            times_b  = batch['timestamps'].to(self.device)
            s_mask_b = batch['student_mask'].to(self.device)
            age_b    = batch['age'].to(self.device).float()
            gender_b = batch['gender'].to(self.device).long()

            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=self.cfg.amp_dtype, enabled=self.cfg.use_amp):
                    orig_severity = captum_forward(f_ids_b, v_nums_b, c_ids_b, times_b, s_mask_b, age_b, gender_b).float().cpu().numpy().flatten()
                    f_ids_mod = f_ids_b.clone()
                    f_ids_mod[:, f_ids_mod.size(1)//2:] = 0
                    mod_severity = captum_forward(f_ids_mod, v_nums_b, c_ids_b, times_b, s_mask_b, age_b, gender_b).float().cpu().numpy().flatten()
                    
            rel_delta = ((mod_severity - orig_severity) / (orig_severity + 1e-5)) * 100.0
            cohort_counterfactual_deltas.extend(rel_delta)
            cohort_global_severity_scores.extend(orig_severity)

            for p_idx in range(min(f_ids_b.size(0), 4)):
                f_s, v_s, c_s = f_ids_b[p_idx:p_idx+1], v_nums_b[p_idx:p_idx+1], c_ids_b[p_idx:p_idx+1]
                t_s, m_s = times_b[p_idx:p_idx+1], s_mask_b[p_idx:p_idx+1]
                age_s, gender_s = age_b[p_idx:p_idx+1], gender_b[p_idx:p_idx+1]

                try:
                    multi_attr = lig.attribute(
                        inputs=(f_s, v_s, c_s, t_s), 
                        target=0, 
                        additional_forward_args=(m_s, age_s, gender_s), 
                        n_steps=10, 
                        internal_batch_size=2
                    )
                    feat_stream = np.squeeze(np.sum(np.abs(multi_attr[0].detach().cpu().numpy()), axis=-1))
                    num_stream  = np.squeeze(np.sum(np.abs(multi_attr[1].detach().cpu().numpy()), axis=-1))
                    cat_stream  = np.squeeze(np.sum(np.abs(multi_attr[2].detach().cpu().numpy()), axis=-1))
                    time_stream = np.squeeze(np.sum(np.abs(multi_attr[3].detach().cpu().numpy()), axis=-1))
                    
                    patient_features = f_s[0].cpu().numpy()
                    for seq_idx in range(min(len(feat_stream), seq_len)):
                        if int(patient_features[seq_idx]) == 0: continue
                        chronological_stream_attributions["feature_id"][seq_idx] += feat_stream[seq_idx]
                        chronological_stream_attributions["numeric_value"][seq_idx] += num_stream[seq_idx]
                        chronological_stream_attributions["categorical_result"][seq_idx] += cat_stream[seq_idx]
                        chronological_stream_attributions["timestamp"][seq_idx] += time_stream[seq_idx]
                except Exception as e:
                    print(f"⚠️ [XAI ATTR WARNING] Batch {batch_idx} sample attribution failed: {e}")

            processed_samples += f_ids_b.size(0)
            
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                elapsed_time = time.perf_counter() - loop_start_time
                print(f"⏳ [XAI HARVEST] Batch {batch_idx:04d}/{total_batches:04d} │ Samples: {processed_samples:,}/{total_samples:,} │ Elapsed: {elapsed_time/60.:.1f}m")

            if processed_samples >= self.cfg.xai_max_samples: break

        z_slots = torch.cat(z_slots_accum, dim=0) 
        y_cohort = torch.cat(y_accum, dim=0).numpy() 
        z_flattened = z_slots.view(z_slots.size(0), -1) 
        z_mean_pooled = z_slots.mean(dim=1)             

        manifold_diag = compute_comprehensive_manifold_diagnostics(z_mean_pooled)

        print(f"\n📊 COHORT LATENT QUANTIZATION METRICS:")
        print(f"  • Manifold Effective Rank: {manifold_diag['effective_rank']:.2f} / {self.cfg.latent_dim}")
        print(f"  • Layer Representation Sparsity Index: {manifold_diag['sparsity_index']:.4f}")

        print("🧬 Compiling cohort-mean latent activation matrix...")
        actual_latent_matrix = z_slots.mean(dim=0).numpy()  
        mean_attn_map = self.compute_cohort_attention_routing_matrix(pipeline)

        self._render_all_exports(
            z_slots=z_slots, z_flat=z_flattened.numpy(), z_pooled=z_mean_pooled, 
            y_cohort=y_cohort, blueprint=actual_latent_matrix, 
            timeline_data=chronological_stream_attributions, mean_attn=mean_attn_map, 
            cf_deltas=cohort_counterfactual_deltas, eff_rank=manifold_diag["effective_rank"], 
            global_severity_scores=cohort_global_severity_scores
        )

        if pipeline.probe is not None:
            hopfield_path = os.path.join(self.cfg.xai_export_dir, "probe_multi_basin_hopfield_energy.png")
            self.plot_multi_basin_hopfield_energy(pipeline.probe, output_path=hopfield_path)

    def _render_all_exports(self, z_slots, z_flat, z_pooled, y_cohort, blueprint, timeline_data, mean_attn, cf_deltas, eff_rank, global_severity_scores):
        print("\n🖼️ Compiling presentation graphics to disk...")

        # 🚀 1. Dynamic Cohort Active Sequence Horizon Detection
        non_zero_indices = np.where(timeline_data["feature_id"] > 0)[0]
        if len(non_zero_indices) > 0:
            max_active_len = int(np.max(non_zero_indices)) + 1
        else:
            max_active_len = 50
        
        max_active_len = max(30, min(self.cfg.max_sequence_len, max_active_len + 5))
        print(f"🔍 [XAI ZOOM] Active clinical timeline window: 0 .. {max_active_len - 1} steps")

        # --- A. Timeline Stackplot ---
        x_axis = np.arange(max_active_len)
        feat_stream = timeline_data["feature_id"][:max_active_len]
        num_stream  = timeline_data["numeric_value"][:max_active_len]
        cat_stream  = timeline_data["categorical_result"][:max_active_len]
        time_stream = timeline_data["timestamp"][:max_active_len]

        plt.figure(figsize=(11, 5.5), dpi=300)
        plt.stackplot(
            x_axis, 
            feat_stream, 
            num_stream,
            cat_stream, 
            time_stream,
            labels=["Feature ID", "Numeric Value", "Categorical Result (BPE)", "Timestamp Delta"],
            colors=["#8e44ad", "#3498db", "#2ecc71", "#f1c40f"], 
            alpha=0.8
        )
        plt.title("Cohort-Mean Integrated Gradients Attribution Across Timeline Sequence Positions", fontsize=12, fontweight='bold', pad=12)
        plt.xlabel("Sequence Position (Chronological Timeline Step)", fontsize=10, labelpad=8)
        plt.ylabel("Attribution Mass", fontsize=10, labelpad=8)
        plt.xlim([0, max_active_len - 1])
        plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.xai_export_dir, "clinical_feature_importance.png"), dpi=300)
        plt.close()

        # 🚀 --- B. Attention Matrix (Inverted High-Contrast Colormap) ---
        active_attn = mean_attn[:max_active_len, :max_active_len]
        plt.figure(figsize=(10.5, 10.5), dpi=300)
        
        # Compute dynamic upper color limit for active non-zero attention cells
        non_zero_vals = active_attn[active_attn > 0]
        if len(non_zero_vals) > 0:
            vmax_target = float(np.percentile(non_zero_vals, 98))
        else:
            vmax_target = float(np.max(active_attn)) if np.max(active_attn) > 0 else 1.0
        vmax_target = max(vmax_target, 0.05)

        step_interval = max(1, max_active_len // 10)
        
        # 🚀 Using 'rocket_r' (inverted): background/0.0 is off-white, active attention is dark red/purple
        sns.heatmap(
            active_attn, 
            cmap="rocket_r", 
            vmin=0.0, 
            vmax=vmax_target,
            cbar_kws={'label': 'Attention Routing Strength'},
            xticklabels=step_interval,
            yticklabels=step_interval
        )
        plt.title(f"Cohort-Mean Layer 0 Token-to-Token Attention Routing Matrix [Active Horizon: {max_active_len} Tokens]", fontsize=12, fontweight='bold', pad=14)
        plt.xlabel("Key Sequence Position", fontsize=10, labelpad=8)
        plt.ylabel("Query Sequence Position", fontsize=10, labelpad=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.xai_export_dir, "high_res_attention_routing.png"), dpi=300)
        plt.close()

        # --- C. 2D & 🚀 3D UMAP Latent Patient Topology ---
        if z_slots.ndim == 3 and z_slots.shape[1] >= self.cfg.num_slots:
            z_pure = z_slots[:, :self.cfg.num_slots, :].mean(dim=1)
        else:
            z_pure = z_pooled if torch.is_tensor(z_pooled) else torch.tensor(z_pooled)

        z_norm = F.normalize(z_pure.float(), p=2, dim=-1).cpu().numpy()
        
        # 1) 2D UMAP Static Image
        p_reducer_2d = umap.UMAP(
            n_neighbors=self.cfg.xai_umap_n_neighbors, min_dist=self.cfg.xai_umap_min_dist, 
            n_components=2, metric=self.cfg.xai_umap_metric, random_state=self.cfg.random_seed
        )
        p_umap_2d = p_reducer_2d.fit_transform(z_norm)
        df_p_2d = pd.DataFrame({
            'UMAP 1': p_umap_2d[:, 0], 
            'UMAP 2': p_umap_2d[:, 1], 
            'Global Severity Load': np.array(global_severity_scores)[:len(p_umap_2d)]
        })

        plt.figure(figsize=(10, 8), dpi=300)
        sc = plt.scatter(data=df_p_2d, x='UMAP 1', y='UMAP 2', c='Global Severity Load', cmap='turbo', s=28, alpha=0.65)
        plt.colorbar(sc, pad=0.02).set_label("Joint Clinical Intensity Index (Expected Load Count)")
        plt.title("T-JEPA Latent Patient Topology mapped to Global Severity Load (2D)", fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.xai_export_dir, "global_patient_manifold.png"), dpi=300)
        plt.close()

        # 🚀 2) 3D UMAP Interactive HTML Export
        try:
            p_reducer_3d = umap.UMAP(
                n_neighbors=self.cfg.xai_umap_n_neighbors, min_dist=self.cfg.xai_umap_min_dist, 
                n_components=3, metric=self.cfg.xai_umap_metric, random_state=self.cfg.random_seed
            )
            p_umap_3d = p_reducer_3d.fit_transform(z_norm)
            
            import plotly.express as px
            df_3d = pd.DataFrame({
                'UMAP Axis 1': p_umap_3d[:, 0],
                'UMAP Axis 2': p_umap_3d[:, 1],
                'UMAP Axis 3': p_umap_3d[:, 2],
                'Global Severity Load': np.array(global_severity_scores)[:len(p_umap_3d)]
            })
            fig_3d = px.scatter_3d(
                df_3d, 
                x='UMAP Axis 1', 
                y='UMAP Axis 2', 
                z='UMAP Axis 3',
                color='Global Severity Load',
                color_continuous_scale='turbo',
                opacity=0.75,
                title="T-JEPA 3D Latent Patient Topology Mapped to Global Severity Load"
            )
            fig_3d.update_traces(marker=dict(size=3))
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='UMAP Axis 1',
                    yaxis_title='UMAP Axis 2',
                    zaxis_title='UMAP Axis 3'
                )
            )
            html_out_path = os.path.join(self.cfg.xai_export_dir, "global_patient_manifold_3d.html")
            fig_3d.write_html(html_out_path)
            print(f"🌐 [3D UMAP EXPORT] Saved interactive 3D manifold to -> {html_out_path}")
        except Exception as e:
            print(f"⚠️ [3D UMAP WARNING] Interactive Plotly HTML generation skipped: {e}")

        # --- D. Counterfactual Spectrum ---
        plt.figure(figsize=(10, 4.8), dpi=300)
        sns.histplot(np.clip(np.array(cf_deltas).flatten(), -100.0, 100.0), kde=True, color="#e74c3c", alpha=0.6, bins=40)
        plt.axvline(x=0.0, color='black', linestyle='--')
        plt.title("Population Counterfactual Risk Modulation Spectrum", fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.xai_export_dir, "population_counterfactual_spectrum.png"), dpi=300)
        plt.close()

        # --- E. Latent Blueprint ---
        plt.figure(figsize=(10, 4.2), dpi=300)
        sns.heatmap(blueprint, cmap="vlag", center=0)
        plt.title(f"Empirical Latent Activation Matrix [{blueprint.shape[0]} slots, 512 channels] (Centered Rank: {eff_rank:.2f})", fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.xai_export_dir, "probe_blueprint.png"), dpi=300)
        plt.close()
        
        cohort_prevalence = (y_cohort.sum() / y_cohort.size) * 100
        print(f"\n🎉 Analytical evaluation complete! Cohort overall target density: {cohort_prevalence:.3f}%")
        print(f"🚀 Saved exports to -> {self.cfg.xai_export_dir}")


if __name__ == "__main__":
    cfg = CardioConfig()
    engine = AdvancedClinicalAnalyticsEngine(cfg)
    engine.execute_evaluation_loop()