# evaluator.py
import os
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
from config import CardioConfig
from Pipeline import *
from src.ModelModules import execute_clinical_audit

import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def extract_probe_predictions(pipeline, data_loader, device):
    """🛠️ EXTRACTS SEQUENTIAL RISK MATRICES ACROSS PRE-FLATTENED SLICES"""
    all_probs, all_targets = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            # Single-pass forward execution pass over pre-flattened [B, T] tensors
            out = pipeline.process_batch(batch, device, run_teacher=False)
            logits = pipeline.linear_probe(out['z_hat_slots'])
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(out['multi_hot_targets'].cpu().numpy())
            
    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)

def generate_and_save_macro_pr_curve(targets, probabilities, output_path="./xai_exports/macro_pr_curve.png", min_positive_prevalence=2):
    """
    📈 MULTI-LABEL MACRO INTERPOLATION CORE:
    Standardizes precision tracks across a static 100-point recall grid 
    to generate a clean, publication-grade Macro Precision-Recall Curve.
    """
    num_samples, num_classes = targets.shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define a uniform 100-point grid for standardizing recall axes
    recall_grid = np.linspace(0.0, 1.0, 100)
    interpolated_precisions = []
    
    for c_idx in range(num_classes):
        pos_count = targets[:, c_idx].sum()
        # Evaluate metrics only on active, long-tailed sparse classes present in the cohort
        if pos_count >= min_positive_prevalence and pos_count < num_samples:
            prec, rec, _ = precision_recall_curve(targets[:, c_idx], probabilities[:, c_idx])
            
            # Reverse arrays because precision_recall_curve outputs recall in descending order
            # np.interp requires the x-axis (recall) to be strictly increasing
            rev_rec = rec[::-1]
            rev_prec = prec[::-1]
            
            # Interpolate precision values across the uniform grid
            interp_prec = np.interp(recall_grid, rev_rec, rev_prec)
            interpolated_precisions.append(interp_prec)
            
    # Compute the true macro average across all valid category tracks
    macro_precision = np.mean(interpolated_precisions, axis=0)
    macro_auc_pr = auc(recall_grid, macro_precision) * 100

    # 🎨 RENDERING HIERARCHY (Matches your other thesis figure styles)
    sns.set_theme(style="ticks")
    plt.figure(figsize=(7.5, 6), dpi=300)
    
    # Fill area under the curve to visualize performance capacity
    plt.fill_between(recall_grid, macro_precision, color="#2980b9", alpha=0.15, label="Manifold Area Volume")
    plt.plot(recall_grid, macro_precision, color="#2980b9", linewidth=2.5, 
             label=f"Macro-Average PR Curve (AUC = {macro_auc_pr:.2f}%)")
    
    # Plot baseline reference line (the average density of positive elements across the label space)
    baseline_prevalence = targets.sum() / (num_samples * num_classes)
    plt.axhline(y=baseline_prevalence, color="#e74c3c", linestyle="--", linewidth=1.2, 
                label=f"Random Prevalence Baseline ({baseline_prevalence * 100:.2f}%)")
    
    # Plot configuration layout boundaries
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
    
    # 1. Filter out rare classes with near-zero positive tokens to prevent metric noise
    active_class_indices = [
        c for c in range(num_classes) 
        if targets[:, c].sum() >= min_positive_prevalence and targets[:, c].sum() < num_samples
    ]
    
    # 2. Establish a fine-grained 100-point threshold sweep grid
    threshold_grid = np.linspace(0.01, 0.99, 100)
    macro_precisions = []
    macro_recalls = []
    
    print("⏳ Running parallel threshold grid sweep across the 456 clinical tracks...")
    for t in threshold_grid:
        # Binarize output matrices at the current threshold level slice
        preds_at_t = (probabilities[:, active_class_indices] > t).astype(float)
        
        # Calculate macro metrics across the uncollapsed diagnostic columns
        macro_p, macro_r, _, _ = precision_recall_fscore_support(
            targets[:, active_class_indices], 
            preds_at_t, 
            average='macro', 
            zero_division=0
        )
        macro_precisions.append(macro_p * 100)
        macro_recalls.append(macro_r * 100)
        
    # Find the empirical breakeven intersection index (closest point where Precision == Recall)
    precision_array = np.array(macro_precisions)
    recall_array = np.array(macro_recalls)
    breakeven_idx = np.argmin(np.abs(precision_array - recall_array))
    optimal_threshold = threshold_grid[breakeven_idx]
    breakeven_score = precision_array[breakeven_idx]

    # 🎨 RENDERING HIERARCHY
    sns.set_theme(style="ticks")
    plt.figure(figsize=(8.5, 5.5), dpi=300)
    
    # Plot curves with contrasting clinical palette colors
    plt.plot(threshold_grid, macro_precisions, color="#2980b9", linewidth=2.5, label="Macro Precision (Positive Predictive Value)")
    plt.plot(threshold_grid, macro_recalls, color="#af7ac5", linewidth=2.5, label="Macro Recall (Sensitivity / TPR)")
    
    # Mark the empirical crossover intersection point
    plt.axvline(x=optimal_threshold, color="#2c3e50", linestyle=":", linewidth=1.2)
    plt.scatter(optimal_threshold, breakeven_score, color="#e74c3c", s=60, zorder=5,
                label=f"Breakeven Point (Thresh: {optimal_threshold:.2f} | Score: {breakeven_score:.2f}%)")
    
    # Plot configuration layout boundaries
    plt.title("T-JEPA Separated Precision & Recall Threshold Spectrum", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Classification Decision Threshold Boundary (τ)", fontsize=10, labelpad=8)
    plt.ylabel("Macro Population Metric Score (%)", fontsize=10, labelpad=8)
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-2.0, 102.0])
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # 🎯 FIX: Swapped out 'bbox_to_transform' for the valid Matplotlib parameter 'bbox_transform'
    plt.legend(
        loc="upper right", 
        frameon=True, 
        fontsize=9, 
        facecolor="white", 
        edgecolor="none"
    )
    sns.despine(trim=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📌 [THRESHOLD PLOT EXPORTED] High-resolution separation lines saved cleanly to -> {output_path}")
    
class CleanReadonlyEvaluator:
    def __init__(self, cfg: CardioConfig):
        self.cfg = cfg
        self.device = cfg.device
        
        # Link the dataloader channel directly to the pre-flattened validation file
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

        pipeline = ClinicalPipeline(cfg, cfg.device)
        pipeline.load_checkpoint("./checkpoints/unified_jepa_and_probe.pt")
        probs, targets = extract_probe_predictions(pipeline, self.val_loader, self.device)
        
        # Route directly to your global clinical audit generator
        execute_clinical_audit(targets, probs)
        generate_and_save_macro_pr_curve(targets, probs, output_path="./xai_exports/macro_precision_recall_curve.png")
        generate_and_save_separated_threshold_curves(targets, probs, output_path="./xai_exports/separated_pr_threshold_curves.png")

if __name__ == "__main__":
    cfg = CardioConfig()
    
    evaluator = CleanReadonlyEvaluator(cfg)
    evaluator.evaluate_pre_fit_checkpoint()