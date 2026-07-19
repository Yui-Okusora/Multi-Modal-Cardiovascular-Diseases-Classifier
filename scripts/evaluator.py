import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')  # Guard rail for remote headless environments
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ingest the newly optimized pre-flattened trajectory dataset loader
from src.TimelineDataset import BVTDFlattenedDataset
from src.ModelModules import *
from src.LoRAWrapper import *
from config import CardioConfig
from Pipeline import *
from src.ModelModules import execute_clinical_audit

import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def extract_probe_predictions(pipeline, data_loader, device):
    """🚀 HIGH-SPEED REFACTORED STANDALONE EXTRACTION PASSTHROUGH"""
    all_probs, all_targets, all_cardinalities = [], [], []
    
    with torch.no_grad():
        for batch in data_loader:
            out = pipeline.process_batch(batch, device, run_teacher=False)
            
            all_probs.append(torch.sigmoid(out['logits']).cpu().numpy())
            all_targets.append(out['multi_hot_targets'].cpu().numpy())
            all_cardinalities.append(out['predicted_cardinalities'].cpu().numpy())
            
    return (
        np.concatenate(all_probs, axis=0), 
        np.concatenate(all_targets, axis=0), 
        np.concatenate(all_cardinalities, axis=0)
    )

def generate_and_save_macro_pr_curve(targets, probabilities, output_path="./xai_exports/macro_pr_curve.png", min_positive_prevalence=2):
    """
    📈 MULTI-LABEL MACRO INTERPOLATION CORE:
    Standardizes precision tracks across a static 100-point recall grid 
    to generate a clean, publication-grade Macro Precision-Recall Curve.
    """
    num_samples, num_classes = targets.shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    recall_grid = np.linspace(0.0, 1.0, 100)
    interpolated_precisions = []
    
    for c_idx in range(num_classes):
        pos_count = targets[:, c_idx].sum()
        if pos_count >= min_positive_prevalence and pos_count < num_samples:
            prec, rec, _ = precision_recall_curve(targets[:, c_idx], probabilities[:, c_idx])
            rev_rec = rec[::-1]
            rev_prec = prec[::-1]
            interp_prec = np.interp(recall_grid, rev_rec, rev_prec)
            interpolated_precisions.append(interp_prec)
            
    macro_precision = np.mean(interpolated_precisions, axis=0)
    macro_auc_pr = auc(recall_grid, macro_precision) * 100

    sns.set_theme(style="ticks")
    plt.figure(figsize=(7.5, 6), dpi=300)
    
    plt.fill_between(recall_grid, macro_precision, color="#2980b9", alpha=0.15, label="Manifold Area Volume")
    plt.plot(recall_grid, macro_precision, color="#2980b9", linewidth=2.5, 
             label=f"Macro-Average PR Curve (AUC = {macro_auc_pr:.2f}%)")
    
    baseline_prevalence = targets.sum() / (num_samples * num_classes)
    plt.axhline(y=baseline_prevalence, color="#e74c3c", linestyle="--", linewidth=1.2, 
                label=f"Random Prevalence Baseline ({baseline_prevalence * 100:.2f}%)")
    
    plt.title("T-JEPA Macro-Averaged Clinical Precision-Recall Curve", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Recall (Sensitivity / True Positive Rate)", fontsize=10, labelpad=8)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=10, labelpad=8)
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=9, facecolor="white", edgecolor="none")
    sns.despine(trim=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📌 [PR CURVE EXPORTED] High-resolution evaluation plot saved cleanly to -> {output_path}")

def generate_and_save_separated_threshold_curves(targets, probabilities, output_path="./xai_exports/separated_pr_threshold_curves.png", min_positive_prevalence=2):
    """
    🎛️ SEPARATED THRESHOLD METRIC GRID SWEEP:
    Sweeps the decision threshold space from 0.01 to 0.99, calculating Macro Precision 
    and Macro Recall independently to chart their cross-over behavior.
    """
    num_samples, num_classes = targets.shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    active_class_indices = [
        c for c in range(num_classes) 
        if targets[:, c].sum() >= min_positive_prevalence and targets[:, c].sum() < num_samples
    ]
    
    threshold_grid = np.linspace(0.01, 0.99, 100)
    macro_precisions = []
    macro_recalls = []
    
    print("⏳ Running parallel threshold grid sweep across the 456 clinical tracks...")
    for t in threshold_grid:
        preds_at_t = (probabilities[:, active_class_indices] > t).astype(float)
        macro_p, macro_r, _, _ = precision_recall_fscore_support(
            targets[:, active_class_indices], 
            preds_at_t, 
            average='macro', 
            zero_division=0
        )
        macro_precisions.append(macro_p * 100)
        macro_recalls.append(macro_r * 100)
        
    precision_array = np.array(macro_precisions)
    recall_array = np.array(macro_recalls)
    active_operating_mask = (precision_array > 0.0) & (recall_array > 0.0)
    
    if np.any(active_operating_mask):
        absolute_deltas = np.abs(precision_array - recall_array)
        absolute_deltas[~active_operating_mask] = float('inf')
        breakeven_idx = np.argmin(absolute_deltas)
        optimal_threshold = threshold_grid[breakeven_idx]
        breakeven_score = (precision_array[breakeven_idx] + recall_array[breakeven_idx]) / 2.0
    else:
        breakeven_idx = np.argmin(np.abs(precision_array - recall_array))
        optimal_threshold = threshold_grid[breakeven_idx]
        breakeven_score = precision_array[breakeven_idx]

    sns.set_theme(style="ticks")
    plt.figure(figsize=(8.5, 5.5), dpi=300)
    
    plt.plot(threshold_grid, macro_precisions, color="#2980b9", linewidth=2.5, label="Macro Precision (Positive Predictive Value)")
    plt.plot(threshold_grid, macro_recalls, color="#af7ac5", linewidth=2.5, label="Macro Recall (Sensitivity / TPR)")
    
    plt.axvline(x=optimal_threshold, color="#2c3e50", linestyle=":", linewidth=1.2)
    plt.scatter(optimal_threshold, breakeven_score, color="#e74c3c", s=60, zorder=5,
                label=f"Breakeven Point (Thresh: {optimal_threshold:.2f} | Score: {breakeven_score:.2f}%)")
    
    plt.title("T-JEPA Separated Precision & Recall Threshold Spectrum", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Classification Decision Threshold Boundary (τ)", fontsize=10, labelpad=8)
    plt.ylabel("Macro Population Metric Score (%)", fontsize=10, labelpad=8)
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-2.0, 102.0])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=9, facecolor="white", edgecolor="none")
    sns.despine(trim=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📌 [THRESHOLD PLOT EXPORTED] High-resolution separation lines saved cleanly to -> {output_path}")

class CleanReadonlyEvaluator:
    def __init__(self, cfg: CardioConfig):
        self.cfg = cfg
        self.device = cfg.device
        
        self.val_loader = DataLoader(
            BVTDFlattenedDataset(cfg.val_csv_path, max_seq_len=cfg.max_sequence_len, max_targets=cfg.max_targets), 
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )

    def evaluate_pre_fit_checkpoint(self, checkpoint_name="unified_jepa_and_probe.pt"):
        print(f"\n🏥 Loading Production Artifact For Immediate Evaluation: {checkpoint_name}")
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint missing at: {checkpoint_path}")
            return

        pipeline = ClinicalPipeline(self.cfg, self.device)
        pipeline.load_checkpoint(checkpoint_path)
        
        # 🛡️ SYSTEM INTEGRITY SHIELD: Explicitly load cardinal parameters if absent or stored under legacy keys
        weights = torch.load(checkpoint_path, map_location=self.device)
        for key in ['cardinal_state', 'ensemble_cardinals_state', 'cardinal_head_state', 'cardinality_head_state']:
            if key in weights and weights[key] is not None:
                # Standalone backward compatibility map to cleanly drop prefix indices
                sd = {k.replace("0.", ""): v for k, v in weights[key].items()}
                if pipeline.cardinal is None:
                    pipeline.inject_phase2_infrastructure()
                pipeline.cardinal.load_state_dict(sd, strict=False)
                print(f"✨ Manually restored standalone cardinal head parameters from dictionary key: '{key}'")
                break
                
        # Force evaluation mode across the standalone downstream modules
        if pipeline.probe is not None:
            pipeline.probe.eval()
        if pipeline.cardinal is not None:
            pipeline.cardinal.eval()
        
        # 1. Extract raw predictions over frozen backbone features
        probs, targets, pred_cards = extract_probe_predictions(pipeline, self.val_loader, self.device)
        
        # 2. Run Baseline Multi-Label Evaluation (Standard global flat threshold of 0.15)
        print("\n📊 Running Baseline Audit (Fixed Anchor τ = 0.15 with Auxiliary Cardinality)...")
        flat_thresholds = np.ones(pipeline.num_icd_classes) * 0.15
        execute_clinical_audit(
            targets, probs, 
            predicted_cardinalities=pred_cards, 
            thresholds=flat_thresholds, 
            calibrate_per_class=False
        )
        
        # 3. Run Built-In Clinical Safety Auto-Calibration (3-Tier Barrier Stratification)
        print("\n🌀 Running Built-In Clinical Safety Auto-Calibration (with Auxiliary Cardinality)...")
        clinical_audit = execute_clinical_audit(
            targets, probs, 
            predicted_cardinalities=pred_cards, 
            thresholds=None, 
            calibrate_per_class=True
        )
        calibrated_thresholds = clinical_audit["calibrated_thresholds"]
        
        # 4. Persist Clinically Calibrated Threshold Vector to Checkpoint Directory
        threshold_save_path = os.path.join(self.cfg.checkpoint_dir, "calibrated_thresholds.json")
        thresholds_dict = {str(i): float(t) for i, t in enumerate(calibrated_thresholds)}
        with open(threshold_save_path, "w") as f:
            json.dump(thresholds_dict, f, indent=4)
        print(f"💾 [EXPORT COMPLETE] Saved clinically calibrated thresholds to -> {threshold_save_path}")
        
        # 5. Save Analytical Visual Plots
        generate_and_save_macro_pr_curve(targets, probs, output_path="./xai_exports/macro_precision_recall_curve.png")
        generate_and_save_separated_threshold_curves(targets, probs, output_path="./xai_exports/separated_pr_threshold_curves.png")

if __name__ == "__main__":
    cfg = CardioConfig()
    evaluator = CleanReadonlyEvaluator(cfg)
    evaluator.evaluate_pre_fit_checkpoint()