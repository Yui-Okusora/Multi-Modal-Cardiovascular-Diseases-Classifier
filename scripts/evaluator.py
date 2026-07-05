# evaluator.py
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')  # Guard rail for remote headless servers (prevents display crashes)
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.TimelineDataset import BVTDTimelineDataset
from src.ModelModules import *
from config import CardioConfig
from Pipeline import *

from ModelModules import execute_clinical_audit

import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning) # Clean block for scikit-learn 1.8+
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def load_unified_evaluation_pipeline(cfg, meta, checkpoint_path, device):
    """
    🎯 Single Responsibility: Re-construct your model architecture footprint 
    and pull the fully pre-fit probe head straight out of the unified checkpoint.
    """
    pipeline = ClinicalPipeline(cfg, device)
    
    # Load the unified production artifact containing all weights
    weights = torch.load(checkpoint_path, map_location=device)
    pipeline.context_encoder.load_state_dict(weights['context_encoder_state'])
    pipeline.predictor.load_state_dict(weights['predictor_state'])
    pipeline.linear_probe.load_state_dict(weights['linear_probe_state']) # 🎯 Pulled directly from Phase 2!
    
    pipeline.context_encoder.eval()
    pipeline.predictor.eval()
    pipeline.linear_probe.eval()
    
    
    return pipeline, pipeline.linear_probe

def extract_probe_predictions(probe, pipeline, data_loader, mode, device):
    """🎯 Single Responsibility: Process a single evaluation pass to gather probabilities and targets."""
    all_probs, all_targets = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            out = pipeline.process_batch(batch, device)
            # Route through the pre-fit probe using the target representation track
            logits = probe(out[mode])
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(out['multi_hot_targets'].cpu().numpy())
            
    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)

class CleanReadonlyEvaluator:
    def __init__(self, cfg: CardioConfig):
        self.cfg = cfg
        self.device = cfg.device
        with open(cfg.codebook_json_path, "r", encoding="utf-8") as f:
            self.meta = __import__('json').load(f)["metadata"]
        
        # 🎯 Clean Isolation: We ONLY need the validation set loader now!
        self.val_loader = DataLoader(
            BVTDTimelineDataset(cfg.val_csv_path, max_seq_len=cfg.max_sequence_len, max_targets=cfg.max_targets), 
            batch_size=cfg.batch_size, shuffle=False
        )

    def evaluate_pre_fit_checkpoint(self, checkpoint_name="unified_jepa_and_probe.pt"):
        print(f"\n🌀 Loading Production Artifact For Immediate Evaluation: {checkpoint_name}")
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Checkpoint missing at: {checkpoint_path}")
            return

        # 1. Pipeline Assembly (Loads pre-fit models instantly)
        pipeline, probe = load_unified_evaluation_pipeline(self.cfg, self.meta, checkpoint_path, self.device)

        # 2. Immediate Run (Evaluating the predictor space track 'z_hat_slots' as configured in trainer Phase 2)
        mode = 'z_hat_slots'
        probs, targets = extract_probe_predictions(probe, pipeline, self.val_loader, mode, self.device)
        execute_clinical_audit(targets, probs)


if __name__ == "__main__":
    cfg = CardioConfig()
    evaluator = CleanReadonlyEvaluator(cfg)
    evaluator.evaluate_pre_fit_checkpoint()