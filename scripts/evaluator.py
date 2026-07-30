# scripts/evaluator.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc, precision_recall_fscore_support
from typing import Tuple, Optional

from src.TimelineDataset import BVTDFlattenedDataset
from src.ModelModules import execute_clinical_audit
from config import CardioConfig
from Pipeline import ClinicalPipeline

import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def extract_probe_predictions(
    pipeline: ClinicalPipeline, 
    data_loader: DataLoader, 
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_probs, all_targets, all_cardinalities = [], [], []
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=pipeline.cfg.amp_dtype, enabled=pipeline.cfg.use_amp):
            for batch in data_loader:
                out = pipeline.process_batch(batch, device, run_teacher=False)
                probs = torch.sigmoid(out['logits'])

                all_probs.append(probs.float().cpu().numpy())
                all_targets.append(out['multi_hot_targets'].cpu().numpy())
                all_cardinalities.append(out['predicted_cardinalities'].float().cpu().numpy())
            
    return (
        np.concatenate(all_probs, axis=0), 
        np.concatenate(all_targets, axis=0), 
        np.concatenate(all_cardinalities, axis=0)
    )


def generate_and_save_macro_pr_curve(
    targets: np.ndarray, 
    probabilities: np.ndarray, 
    output_path: str, 
    min_positive_prevalence: int = 2
):
    num_samples, num_classes = targets.shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    recall_grid = np.linspace(0.0, 1.0, 100)
    interpolated_precisions = []
    
    for c_idx in range(num_classes):
        pos_count = targets[:, c_idx].sum()
        if pos_count >= min_positive_prevalence and pos_count < num_samples:
            prec, rec, _ = precision_recall_curve(targets[:, c_idx], probabilities[:, c_idx])
            interp_prec = np.interp(recall_grid, rec[::-1], prec[::-1])
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
    plt.xlim([-0.02, 1.02]); plt.ylim([-0.02, 1.02])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=9, facecolor="white", edgecolor="none")
    sns.despine(trim=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📌 [PR CURVE EXPORTED] Saved cleanly to -> {output_path}")


def generate_and_save_separated_threshold_curves(
    targets: np.ndarray, 
    probabilities: np.ndarray, 
    output_path: str, 
    min_positive_prevalence: int = 2
):
    num_samples, num_classes = targets.shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    active_class_indices = [
        c for c in range(num_classes) 
        if targets[:, c].sum() >= min_positive_prevalence and targets[:, c].sum() < num_samples
    ]
    
    threshold_grid = np.linspace(0.01, 0.99, 100)
    macro_precisions, macro_recalls = [], []
    
    for t in threshold_grid:
        preds_at_t = (probabilities[:, active_class_indices] > t).astype(float)
        macro_p, macro_r, _, _ = precision_recall_fscore_support(
            targets[:, active_class_indices], preds_at_t, average='macro', zero_division=0
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
    else:
        breakeven_idx = np.argmin(np.abs(precision_array - recall_array))
        
    optimal_threshold = threshold_grid[breakeven_idx]
    breakeven_score = (precision_array[breakeven_idx] + recall_array[breakeven_idx]) / 2.0

    sns.set_theme(style="ticks")
    plt.figure(figsize=(8.5, 5.5), dpi=300)
    
    plt.plot(threshold_grid, macro_precisions, color="#2980b9", linewidth=2.5, label="Macro Precision (PPV)")
    plt.plot(threshold_grid, macro_recalls, color="#af7ac5", linewidth=2.5, label="Macro Recall (Sensitivity)")
    plt.axvline(x=optimal_threshold, color="#2c3e50", linestyle=":", linewidth=1.2)
    plt.scatter(optimal_threshold, breakeven_score, color="#e74c3c", s=60, zorder=5,
                label=f"Breakeven Point (Thresh: {optimal_threshold:.2f} | Score: {breakeven_score:.2f}%)")
    
    plt.title("T-JEPA Precision & Recall Threshold Spectrum", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Classification Decision Threshold (tau)", fontsize=10, labelpad=8)
    plt.ylabel("Macro Metric Score (%)", fontsize=10, labelpad=8)
    plt.xlim([-0.02, 1.02]); plt.ylim([-2.0, 102.0])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=9, facecolor="white")
    sns.despine(trim=True); plt.tight_layout()
    plt.savefig(output_path, dpi=300); plt.close()
    print(f"📌 [THRESHOLD PLOT EXPORTED] Saved cleanly to -> {output_path}")


class CleanReadonlyEvaluator:
    def __init__(self, cfg: CardioConfig):
        self.cfg = cfg
        self.device = cfg.device
        
        self.val_loader = DataLoader(
            BVTDFlattenedDataset(
                preprocessed_csv_path=cfg.val_csv_path, 
                max_seq_len=cfg.max_sequence_len, 
                max_subwords=cfg.max_subwords,  # 🚀 FIXED: Explicit keyword argument
                max_targets=cfg.max_targets,    # 🚀 FIXED: Explicit keyword argument
                is_train=False
            ), 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )

    def evaluate_checkpoint(self, checkpoint_name: Optional[str] = None):
        if checkpoint_name is None:
            intermediary_path = os.path.join(self.cfg.checkpoint_dir, self.cfg.unified_intermediary_filename)
            final_path = os.path.join(self.cfg.checkpoint_dir, self.cfg.unified_final_filename)
            
            if os.path.exists(intermediary_path):
                checkpoint_name = self.cfg.unified_intermediary_filename
            elif os.path.exists(final_path):
                checkpoint_name = self.cfg.unified_final_filename
            else:
                print(f"❌ No checkpoint artifacts located in: {self.cfg.checkpoint_dir}")
                return

        print(f"\n🏥 Loading Production Artifact For Immediate Evaluation: {checkpoint_name}")
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, checkpoint_name)

        pipeline = ClinicalPipeline(self.cfg, self.device)
        pipeline.load_checkpoint(checkpoint_path)
                
        pipeline.context_encoder.eval()
        if pipeline.probe is not None: pipeline.probe.eval()
        if pipeline.cardinal is not None: pipeline.cardinal.eval()
        
        probs, targets, pred_cards = extract_probe_predictions(pipeline, self.val_loader, self.device)

        # 🚀 CARDINALITY DIAGNOSTIC AUDIT
        rounded_k = np.maximum(1, np.round(pred_cards.squeeze()))
        unique_k, k_counts = np.unique(rounded_k, return_counts=True)
        dist_str = ", ".join([f"K={int(k)}: {c}" for k, c in zip(unique_k, k_counts)])
        print(f"\n📊 [CARDINALITY AUDIT] Mean Pred K: {pred_cards.mean():.2f} (Min: {pred_cards.min():.2f}, Max: {pred_cards.max():.2f})")
        print(f"   ↳ Dynamic K Distribution: [{dist_str}]")
        
        print(f"\n📊 Running Baseline Audit (Fixed Anchor tau = {self.cfg.eval_flat_threshold} with Auxiliary Cardinality)...")
        flat_thresholds = np.ones(pipeline.num_icd_classes) * self.cfg.eval_flat_threshold
        execute_clinical_audit(
            targets, probs, predicted_cardinalities=pred_cards, 
            thresholds=flat_thresholds, calibrate_per_class=False, temp_alpha=self.cfg.eval_temp_alpha
        )
        
        print("\n🌀 Running Built-In Clinical Safety Auto-Calibration (with Auxiliary Cardinality)...")
        clinical_audit = execute_clinical_audit(
            targets, probs, predicted_cardinalities=pred_cards, 
            thresholds=None, calibrate_per_class=True, temp_alpha=self.cfg.eval_temp_alpha
        )
        calibrated_thresholds = clinical_audit["calibrated_thresholds"]
        
        threshold_save_path = os.path.join(self.cfg.checkpoint_dir, self.cfg.calibrated_thresholds_filename)
        thresholds_dict = {str(i): float(t) for i, t in enumerate(calibrated_thresholds)}
        with open(threshold_save_path, "w") as f:
            json.dump(thresholds_dict, f, indent=4)
        print(f"💾 [EXPORT COMPLETE] Saved calibrated thresholds to -> {threshold_save_path}")
        
        pr_path = os.path.join(self.cfg.xai_export_dir, "macro_precision_recall_curve.png")
        thresh_path = os.path.join(self.cfg.xai_export_dir, "separated_pr_threshold_curves.png")
        generate_and_save_macro_pr_curve(targets, probs, output_path=pr_path, min_positive_prevalence=self.cfg.min_positive_prevalence)
        generate_and_save_separated_threshold_curves(targets, probs, output_path=thresh_path, min_positive_prevalence=self.cfg.min_positive_prevalence)


if __name__ == "__main__":
    cfg = CardioConfig()
    evaluator = CleanReadonlyEvaluator(cfg)
    evaluator.evaluate_checkpoint(cfg.unified_final_filename)