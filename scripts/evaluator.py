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

def load_unified_evaluation_pipeline(cfg, checkpoint_path, device):
    """🛠️ RECONSTRUCTS ARTIFACT AND SYNCS POLYSYMMETRIC WEIGHTS"""
    pipeline = ClinicalPipeline(cfg, device)
    weights = torch.load(checkpoint_path, map_location=device)
    pipeline.context_encoder.load_state_dict(weights['context_encoder_state'])
    pipeline.predictor.load_state_dict(weights['predictor_state'])
    pipeline.linear_probe.load_state_dict(weights['linear_probe_state'])
    
    pipeline.context_encoder.eval()
    pipeline.predictor.eval()
    pipeline.linear_probe.eval()
    return pipeline

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

        pipeline = load_unified_evaluation_pipeline(self.cfg, checkpoint_path, self.device)
        probs, targets = extract_probe_predictions(pipeline, self.val_loader, self.device)
        
        # Route directly to your global clinical audit generator
        execute_clinical_audit(targets, probs)

if __name__ == "__main__":
    cfg = CardioConfig()
    
    evaluator = CleanReadonlyEvaluator(cfg)
    evaluator.evaluate_pre_fit_checkpoint()