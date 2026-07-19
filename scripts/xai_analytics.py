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
from src.LoRAWrapper import *
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
        self.id_to_token = self.codebooks.get("inverse_maps", {})
        self.decoder = ClinicalDecoder(cfg.codebook_json_path)

        # ⚡ OPERATIONAL CHUNK: Step down batch size to avoid Captum sequential graph memory buildup
        self.val_loader = DataLoader(
            BVTDFlattenedDataset(cfg.val_csv_path, max_seq_len=cfg.max_sequence_len, max_targets=cfg.max_targets), 
            batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

    def run_footprint_audit(self, pipeline):
        """
        🖥️ DYNAMIC MODEL FOOTPRINT AUDIT:
        Queries loaded modules to present an absolute parameter ledger 
        (trainable vs. untrainable) across production boundaries.
        """
        print("\n" + "="*80)
        print("🔍 DETAILED MODEL FOOTPRINT AUDIT: TRAINING VS. PRODUCTION CONFIGURATION")
        print("="*80)
        
        # Resolve standalone architecture variables natively
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

        # Config A: FULL TRAINING CONFIGURATION (All potential slots)
        all_training_modules = [ctx_enc, tgt_enc, predictor, projector, probe, cardinal, assembler]
        train_tot = sum(get_stats(m)[0] for m in all_training_modules)
        train_tr = sum(get_stats(m)[1] for m in all_training_modules)
        train_untr = train_tot - train_tr

        # Config B: MINIMAL PRODUCTION/INFERENCE CONFIGURATION (Encoder + Assembler + Downstreams)
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

        # 2. Active VRAM performance metrics calculation pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            baseline_vram = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            audit_batch = next(iter(self.val_loader))
            start_time = time.perf_counter()
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _ = pipeline.process_batch(audit_batch, self.device, run_teacher=False)
            
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            dynamic_allocated = peak_vram - baseline_vram
            
            print(f"⚡ REAL-TIME GPU EXECUTION PROFILE (Batch Size: {self.val_loader.batch_size}):")
            print(f"   • Baseline VRAM Allocated:   {baseline_vram:.2f} MB")
            print(f"   • Peak Runtime VRAM Limit:   {peak_vram:.2f} MB")
            print(f"   • Dynamic Batch Overhead:    {dynamic_allocated:.2f} MB")
            print(f"   • Forward Pass Latency:      {elapsed_ms:.2f} ms")
        else:
            print("🖥️ CUDA platform is unavailable. Skipping active hardware profiling.")
        print("="*80 + "\n")

    def execute_evaluation_loop(self):
        pipeline = ClinicalPipeline(self.cfg, self.cfg.device)
        pipeline.load_checkpoint(checkpoint_path=os.path.join(self.cfg.checkpoint_dir, "unified_jepa_and_probe.pt"))
        
        # 🎛️ PRODUCTION RESET: Bake out lingering adapter wrappers and restore 100% trainable tracking states
        from src.LoRAWrapper import defactorize_entire_architecture
        defactorize_entire_architecture(pipeline.context_encoder)
        for param in pipeline.context_encoder.parameters():
            param.requires_grad = True

        self.run_footprint_audit(pipeline)
        print("⚡ Processing population arrays and intercepting gradient pathways...")

        def captum_forward(f, v, c, t, s_mask, age, gender):
            curr_b = f.size(0)
            batch_dict = {
                'feature_ids': f, 'numeric_values': v, 'cat_result_ids': c, 'timestamps': t, 'student_mask': s_mask,
                'age': age, 'gender': gender,
                'icd_targets': torch.zeros(curr_b, 1, dtype=torch.long, device=f.device),
                'target_mask': torch.zeros(curr_b, 1, dtype=torch.bool, device=f.device)
            }
            out = pipeline.process_batch(batch_dict, f.device, run_teacher=False)
            return (out['predicted_cardinalities'] + torch.sigmoid(out['logits']).sum(dim=-1)).unsqueeze(-1)

        # ─── RECONSTRUCTED MULTI-STREAM TARGET INFRASTRUCTURE ───
        tokenizer_mod = pipeline.context_encoder.tokenizer
        target_layers = [
            tokenizer_mod.feature_embedding,      # Stream 0: Variable Identity
            tokenizer_mod.numeric_norm,           # Stream 1: Continuous Volatility
            tokenizer_mod.cat_result_embedding,   # Stream 2: Categorical States
            tokenizer_mod.time_embedder           # Stream 3: Temporal Chronology
        ]
        
        # Instantiate LayerIntegratedGradients over the list of tracking modules
        lig = LayerIntegratedGradients(captum_forward, target_layers)

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
        cohort_attn_maps = []
        cohort_global_severity_scores = []

        # 🎯 CHRONOLOGICAL LEADERS: Initialize fixed arrays sized explicitly by sequence depth
        seq_len = self.cfg.max_sequence_len
        chronological_stream_attributions = {
            "feature_id": np.zeros(seq_len),
            "numeric_value": np.zeros(seq_len),
            "categorical_result": np.zeros(seq_len),
            "timestamp": np.zeros(seq_len)
        }

        total_batches = len(self.val_loader)
        total_samples = len(self.val_loader.dataset)
        processed_samples = 0
        loop_start_time = time.perf_counter()

        for batch_idx, batch in enumerate(self.val_loader):
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    orig_severity = captum_forward(f_ids_b, v_nums_b, c_ids_b, times_b, s_mask_b, age_b, gender_b).float().cpu().numpy().flatten()
                    
                    f_ids_mod = f_ids_b.clone()
                    f_ids_mod[:, f_ids_mod.size(1)//2:] = 0
                    
                    mod_severity = captum_forward(f_ids_mod, v_nums_b, c_ids_b, times_b, s_mask_b, age_b, gender_b).float().cpu().numpy().flatten()
                    
            cohort_counterfactual_deltas.extend(mod_severity - orig_severity)
            cohort_global_severity_scores.extend(orig_severity)

            for p_idx in range(f_ids_b.size(0)):
                f_s, v_s, c_s = f_ids_b[p_idx:p_idx+1], v_nums_b[p_idx:p_idx+1], c_ids_b[p_idx:p_idx+1]
                t_s, m_s = times_b[p_idx:p_idx+1], s_mask_b[p_idx:p_idx+1]
                age_s = age_b[p_idx:p_idx+1]
                gender_s = gender_b[p_idx:p_idx+1]

                multi_attr = lig.attribute(
                    inputs=(f_s, v_s, c_s, t_s, m_s), 
                    target=0, 
                    additional_forward_args=(age_s, gender_s),
                    n_steps=12, 
                    internal_batch_size=2
                )
                
                feat_stream = np.squeeze(np.sum(np.abs(multi_attr[0].detach().cpu().numpy()), axis=-1))
                num_stream  = np.squeeze(np.sum(np.abs(multi_attr[1].detach().cpu().numpy()), axis=-1))
                cat_stream  = np.squeeze(np.sum(np.abs(multi_attr[2].detach().cpu().numpy()), axis=-1))
                time_stream = np.squeeze(np.sum(np.abs(multi_attr[3].detach().cpu().numpy()), axis=-1))
                
                # 🎯 FIXED & DEFINED: Extract token identities for the current patient slice
                patient_features = f_s[0].cpu().numpy()
                
                # Accumulate gradient scores chronologically by step index position
                for seq_idx in range(min(len(feat_stream), seq_len)):
                    fid = int(patient_features[seq_idx])
                    
                    # 🛡️ PADDING SHIELD: Skip trailing pad tokens so they don't corrupt the chart baseline
                    if fid == 0:
                        continue
                        
                    chronological_stream_attributions["feature_id"][seq_idx] += feat_stream[seq_idx]
                    chronological_stream_attributions["numeric_value"][seq_idx] += num_stream[seq_idx]
                    chronological_stream_attributions["categorical_result"][seq_idx] += cat_stream[seq_idx]
                    chronological_stream_attributions["timestamp"][seq_idx] += time_stream[seq_idx]

                attn_maps.clear()
                handle = target_attn_block.register_forward_hook(attn_hook)
                with torch.no_grad(): 
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        _ = captum_forward(f_s, v_s, c_s, t_s, m_s, age_s, gender_s)
                handle.remove()
                if attn_maps:
                    heatmap_data = attn_maps[0][0]
                    if heatmap_data.ndim == 3: heatmap_data = np.mean(heatmap_data, axis=0)
                    cohort_attn_maps.append(heatmap_data)

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

        manifold_diag = compute_comprehensive_manifold_diagnostics(z_mean_pooled)
        print(f"\n📊 COHORT LATENT QUANTIZATION METRICS:")
        print(f"  • Manifold Effective Rank: {manifold_diag['effective_rank']:.2f} / {self.cfg.latent_dim}")
        print(f"  • Layer Representation Sparsity Index: {manifold_diag['sparsity_index']:.4f}")

        print("🧬 Compiling cohort-mean latent activation matrix...")
        actual_latent_matrix = z_slots.mean(dim=0).numpy()  
        
        self._render_all_exports(
            z_flat=z_flattened.numpy(), z_pooled=z_mean_pooled, y_cohort=y_cohort, blueprint=actual_latent_matrix, 
            timeline_data=chronological_stream_attributions, mean_attn=np.mean(cohort_attn_maps, axis=0), 
            cf_deltas=cohort_counterfactual_deltas, eff_rank=manifold_diag["effective_rank"], global_severity_scores=cohort_global_severity_scores
        )

    def _render_all_exports(self, z_flat, z_pooled, y_cohort, blueprint, timeline_data, mean_attn, cf_deltas, eff_rank, global_severity_scores):
        print("\n🖼️ Compiling consolidated presentation sheets to disk...")
        
        if z_pooled is None or len(z_pooled) == 0:
            print("⚠️ Warning: Pooled representations are empty. Manifold structural tracing compromised.")

        # ----------------------------------------------------------------------
        # 📊 CHART 1: CHRONOLOGICAL STACKED POSITION ATTRIBUTION TIMELINE
        # ----------------------------------------------------------------------
        print("📈 Generating position-sorted stacked area timeline...")
        seq_len = len(timeline_data["feature_id"])
        positions = np.arange(seq_len)
        
        plt.figure(figsize=(11, 5.5), dpi=300)
        
        # Build chronological stacked areas using the position leader arrays
        plt.stackplot(
            positions,
            timeline_data["feature_id"],
            timeline_data["numeric_value"],
            timeline_data["categorical_result"],
            timeline_data["timestamp"],
            labels=[
                "Stream 0: Variable Identity (Feature ID)", 
                "Stream 1: Value Magnitude (Numeric Projection)", 
                "Stream 2: Categorical Finding (Result ID)", 
                "Stream 3: Temporal Recency (Timestamp Delta)"
            ],
            colors=["#8e44ad", "#3498db", "#2ecc71", "#f1c40f"],
            alpha=0.8
        )
        
        plt.title("Cohort-Mean Integrated Gradients Attribution Across Timeline Sequence Positions", fontsize=12, fontweight='bold', pad=12)
        plt.xlabel("Chronological Sequence Index Position (Old Historical Records ───► Acute Bedside Entry)", fontsize=10, labelpad=8)
        plt.ylabel("Cumulative Absolute Gradient Importance Mass", fontsize=10, labelpad=8)
        plt.xlim(0, seq_len - 1)
        plt.grid(True, linestyle=":", alpha=0.3)
        plt.legend(loc="upper left", frameon=True, fontsize=9)
        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig("./xai_exports/clinical_feature_importance.png", dpi=300)
        plt.close()

        # ----------------------------------------------------------------------
        # 🗺️ CHART 2: HIGH-RESOLUTION STANDALONE ATTENTION ROUTING MATRIX
        # ----------------------------------------------------------------------
        print("🗺️ Exporting high-fidelity standalone Attention Routing matrix...")
        plt.figure(figsize=(10.5, 10.5), dpi=300) 
        
        sns.heatmap(
            mean_attn, 
            cmap="crest", 
            cbar_kws={'label': 'Routing Co-Prevalence Weight', 'pad': 0.02},
            xticklabels=20, yticklabels=20 
        )
        
        plt.title("Cohort-Mean Layer 0 Token-to-Token Attention Routing Matrix", fontsize=12, fontweight='bold', pad=14)
        plt.xlabel("Key Token Timeline Sequence Index", fontsize=10, labelpad=8)
        plt.ylabel("Query Token Timeline Sequence Index", fontsize=10, labelpad=8)
        plt.tight_layout()
        plt.savefig("./xai_exports/high_res_attention_routing.png", dpi=300)
        plt.close()

        # ----------------------------------------------------------------------
        # 🧬 CHART 3: GLOBAL POPULATION MANIFOLD REDUCTION
        # ----------------------------------------------------------------------
        print("🧬 Running global population manifold reduction...")
        p_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        p_umap = p_reducer.fit_transform(z_flat)
        
        df_p = pd.DataFrame({
            'UMAP 1': p_umap[:, 0], 
            'UMAP 2': p_umap[:, 1], 
            'Global Severity Load': np.array(global_severity_scores)[:len(p_umap)]
        })

        plt.figure(figsize=(10, 8), dpi=300)
        sc = plt.scatter(data=df_p, x='UMAP 1', y='UMAP 2', c='Global Severity Load', cmap='turbo', s=25, alpha=0.6, edgecolor='none')
        
        cbar = plt.colorbar(sc, pad=0.02)
        cbar.set_label("Joint Clinical Intensity Index (Expected Load Count)", fontsize=10, labelpad=8)
        cbar.ax.tick_params(labelsize=9)
        
        plt.title("T-JEPA Latent Patient Topology mapped to Global Severity Load", fontsize=12, fontweight='bold', pad=12)
        plt.xlabel("UMAP Coordinate 1", fontsize=10); plt.ylabel("UMAP Coordinate 2", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.4); sns.despine(trim=True)
        plt.tight_layout(); plt.savefig("./xai_exports/global_patient_manifold.png", dpi=300); plt.close()

        # ----------------------------------------------------------------------
        # 📈 CHART 4: POPULATION COUNTERFACTUAL RISK MODULATION SPECTRUM
        # ----------------------------------------------------------------------
        print("📊 Compiling Counterfactual Risk Modulation Spectrum...")
        plt.figure(figsize=(10, 4.8), dpi=300)
        sns.histplot(cf_deltas, kde=True, color="#e74c3c", alpha=0.6, edgecolor='white', bins=40)
        plt.axvline(x=0.0, color='black', linestyle='--', linewidth=1.2)
        
        plt.title("Population Counterfactual Risk Modulation Spectrum", fontsize=12, fontweight='bold', pad=12)
        plt.xlabel("Risk Probability Delta Shift (%)", fontsize=10, labelpad=8)
        plt.ylabel("Count", fontsize=10, labelpad=8)
        plt.grid(True, linestyle=":", alpha=0.4)
        sns.despine(trim=True)
        plt.tight_layout(); plt.savefig("./xai_exports/population_counterfactual_spectrum.png", dpi=300); plt.close()

        # ----------------------------------------------------------------------
        # 🖼️ CHART 5: LATENT ACTIVATION BLUEPRINT HEATMAP
        # ----------------------------------------------------------------------
        plt.figure(figsize=(10, 4.2), dpi=300)
        sns.heatmap(blueprint, cmap="vlag", center=0)
        plt.title(f"Empirical Latent Activation Matrix [24 slots, 512 channels] (Centered Rank: {eff_rank:.2f})", fontweight='bold')
        plt.xlabel("Latent Channel Dimension (512)"); plt.ylabel("Temporal Context Slot (24)")
        plt.tight_layout(); plt.savefig("./xai_exports/probe_blueprint.png", dpi=300); plt.close()
        
        cohort_prevalence = (y_cohort.sum() / y_cohort.size) * 100
        print(f"\n🎉 Analytical evaluation complete. Cohort overall target density: {cohort_prevalence:.3f}%")
        print("🚀 Asset path -> ./xai_exports/")

if __name__ == "__main__":
    cfg = CardioConfig()
    cfg.val_csv_path = "val_patient_flattened.csv"
    cfg.batch_size = 64  
    
    engine = AdvancedClinicalAnalyticsEngine(cfg)
    engine.execute_evaluation_loop()