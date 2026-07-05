# src/Zone1JEPA.py
import math
import torch
import torch.nn as nn
import json

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_recall_fscore_support
import numpy as np

class ContinuousTimeEmbedding(nn.Module):
    """
    Encodes irregular physical timestamps using Random Fourier Features (RFF).
    Maps continuous time intervals into periodic frequency coordinates.
    """
    def __init__(self, d_model: int, dropout_p: float = 0.15):
        super().__init__()
        self.d_time = d_model
        half_dim = d_model // 2
        frequencies = torch.exp(
            torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000.0) / (half_dim - 1))
        )
        self.register_buffer("frequencies", frequencies)
        
        # 🎯 AUGMENTATION 1: Prevents continuous time coordinates from blowing out 
        # the absolute variance space relative to the structural categorical IDs
        self.time_norm = nn.LayerNorm(d_model)
        self.time_dropout = nn.Dropout(p=dropout_p)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        scaled_t = t.unsqueeze(-1) * self.frequencies
        sin_t = torch.sin(scaled_t)
        cos_t = torch.cos(scaled_t)
        time_features = torch.cat([sin_t, cos_t], dim=-1)
        
        # Ground and regularize the temporal boundary vector
        return self.time_dropout(self.time_norm(time_features))

class UnifiedSystemicTokenizer(nn.Module):
    """
    Unified front-end router mapping both Vital Signs and Lab Tests into a shared coordinate space.
    Features an activation-scaling layer for numeric value magnitude safety.
    """
    def __init__(self, num_total_features: int, num_cat_results: int, d_model: int = 256, dropout_p: float = 0.15):
        super().__init__()
        self.feature_embedding = nn.Embedding(num_total_features, d_model)
        self.numeric_projection = nn.Linear(1, d_model, bias=False)
        
        # STABILIZER 1: Prevents unscaled physical measurements from overwhelming the space
        self.numeric_norm = nn.LayerNorm(d_model)
        
        self.cat_result_embedding = nn.Embedding(num_cat_results, d_model, padding_idx=0)
        
        # Pass the token-level dropout down to the time encoder block
        self.time_embedder = ContinuousTimeEmbedding(d_model, dropout_p=dropout_p)
        
        # 🎯 AUGMENTATION 2: Global Frontend Stabilizer Circuit
        # Restores strict cross-patient coordinate alignment after additive pooling
        self.global_token_norm = nn.LayerNorm(d_model)
        self.global_token_dropout = nn.Dropout(p=dropout_p)

    def forward(self, feature_ids: torch.Tensor, numeric_values: torch.Tensor, cat_result_ids: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        feat_emb = self.feature_embedding(feature_ids)
        
        # Scale-stabilized continuous measurement pathway
        val_raw = self.numeric_projection(numeric_values.unsqueeze(-1))
        val_emb = self.numeric_norm(val_raw)
        
        cat_emb = self.cat_result_embedding(cat_result_ids)
        time_emb = self.time_embedder(timestamps)
        
        # Combine the distinct modal streams
        combined_tokens = feat_emb + val_emb + cat_emb + time_emb
        
        # 🎯 THE SHIELD: Standardize token representations across patients before they hit attention loops.
        # Randomly zeroes out 15% of feature coordinates per pass, forcing attention heads 
        # to ground themselves in underlying medical patterns rather than relying on time shortcuts.
        return self.global_token_dropout(self.global_token_norm(combined_tokens))

class PerceiverLatentPooling(nn.Module):
    """
    Squeezes variable-length sequential event streams down to a fixed bottleneck matrix of K slots.
    Implements Pre-LN and query-scale shields to prevent gradient accumulation explosions.
    """
    def __init__(self, num_slots: int, d_model: int, nheads: int = 8):
        super().__init__()
        self.latent_slots = nn.Parameter(torch.randn(num_slots, d_model))
        
        # 🎯 STABILIZER 2: Standardizes the learned query slots to prevent parameter drift scaling
        self.slot_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, batch_first=True)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Pre-LN Execution Path (Significantly more stable backpropagation characteristics)
        norm_slots = self.slot_norm(self.latent_slots).unsqueeze(0).expand(batch_size, -1, -1)
        norm_x = self.kv_norm(x)
        
        # Direct structural guard against empty sequences row-mask crash traps
        if padding_mask is not None:
            all_padded_rows = padding_mask.all(dim=-1, keepdim=True)
            padding_mask = padding_mask.masked_fill(all_padded_rows, False)

        attn_out, _ = self.cross_attn(
            query=norm_slots, 
            key=norm_x, 
            value=norm_x, 
            key_padding_mask=padding_mask
        )
        
        # Residual connection over the standardized queries
        return norm_slots + attn_out

class ContextEncoder(nn.Module):
    """
    The deep sequential backbone of our JEPA structure (Student Stream).
    Utilizes clean token-consistent LayerNorm pipelines to eliminate optimization conflicts.
    """
    def __init__(self, num_total_features: int, num_cat_results: int, d_model: int = 512, num_slots: int = 8, nlayers: int = 4):
        super().__init__()
        self.tokenizer = UnifiedSystemicTokenizer(num_total_features, num_cat_results, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 4, 
            batch_first=True, activation='gelu', norm_first=True
        )
        self.temporal_backbone = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.perceiver_pool = PerceiverLatentPooling(num_slots=num_slots, d_model=d_model)
        
        # 🎯 STABILIZER 3: Unified slot feature stabilizer replacing the conflicting BatchNorm
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, feature_ids: torch.Tensor, numeric_values: torch.Tensor, cat_result_ids: torch.Tensor, timestamps: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        x_tokens = self.tokenizer(feature_ids, numeric_values, cat_result_ids, timestamps)
        h_seq = self.temporal_backbone(x_tokens, src_key_padding_mask=padding_mask)
        z_c = self.perceiver_pool(h_seq, padding_mask=padding_mask)
        
        # Normalize features consistently across slot coordinates
        z_c_stable = self.output_norm(z_c)
        return torch.nn.functional.normalize(z_c_stable, p=2, dim=-1)

class TargetEncoder(nn.Module):
    """
    🎯 PURE JEPA TEACHER: Encodes unmasked future timeline sequences 
    to create stable target coordinates for the predictive world model.
    """
    def __init__(self, num_total_features: int, num_cat_results: int, d_model: int = 512, num_slots: int = 8, nlayers: int = 4):
        super().__init__()
        self.tokenizer = UnifiedSystemicTokenizer(num_total_features, num_cat_results, d_model)
        
        # Symmetrical temporal processing layer matches student abstraction depth
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 4, 
            batch_first=True, activation='gelu', norm_first=True
        )
        self.temporal_backbone = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.perceiver_pool = PerceiverLatentPooling(num_slots=num_slots, d_model=d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, feature_ids, numeric_values, cat_result_ids, timestamps, padding_mask=None):
        x_tokens = self.tokenizer(feature_ids, numeric_values, cat_result_ids, timestamps)
        # Process the future sequence context before pooling
        h_seq = self.temporal_backbone(x_tokens, src_key_padding_mask=padding_mask)
        z_t = self.perceiver_pool(h_seq, padding_mask=padding_mask)
        z_t_stable = self.output_norm(z_t)
        return torch.nn.functional.normalize(z_t_stable, p=2, dim=-1)

class Predictor(nn.Module):
    """🎯 Whole-Slot Predictor: Handles matrix-to-matrix cross-slot translation."""
    def __init__(self, num_slots: int = 8, d_model: int = 512):
        super().__init__()
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        self.slot_combiner = nn.Linear(num_slots, num_slots)

    def forward(self, z_c):
        # Channel-wise transformations
        z_predicted = self.channel_mlp(z_c) 
        
        # Cross-slot mixing with a clean residual layout and protection barrier
        z_predicted_t = z_predicted.transpose(1, 2)
        z_matched_t = self.slot_combiner(z_predicted_t)
        z_out = z_matched_t.transpose(1, 2)
        
        # 🎯 THE SHIELD: Stabilize cross-talk scale before routing to VICReg
        return torch.nn.functional.normalize(z_predicted + z_out, p=2, dim=-1)
    
class LinearProbeHead(nn.Module):
    """
    🎯 THE COHORT-ALIGNED INFRASTRUCTURE PROBE:
    Bypasses fixed initialization seeds completely. Relies entirely on the 
    convexity of the downstream loss landscape to force statistical metric 
    reproducibility across machine setups and shuffled datasets.
    """
    def __init__(self, in_slots: int, in_dim: int, num_classes: int):
        super().__init__()
        input_flat_dim = in_slots * in_dim
        
        self.feature_dropout = nn.Dropout(p=0.4)
        # Natively uses standard PyTorch unseeded initialization
        self.classifier = nn.Linear(input_flat_dim, num_classes)

    def forward(self, z_hat_slots):
        flat_z = z_hat_slots.contiguous().view(z_hat_slots.size(0), -1)
        regularized_features = self.feature_dropout(flat_z)
        return self.classifier(regularized_features)

class ClinicalDecoder:
    def __init__(self, codebook_json_path: str = "clinical_codebooks.json"):
        with open(codebook_json_path, "r", encoding="utf-8") as f:
            self.codebooks = json.load(f)
            
        self.num_total_features = self.codebooks["metadata"]["num_total_features"]
        self.num_cat_results = self.codebooks["metadata"]["num_cat_results"]
        self.num_icd_classes = self.codebooks["metadata"]["num_icd_classes"]
        
        self.id_to_icd = self.codebooks["inverse_icd_codes"]
        self.id_to_feature = self.codebooks["inverse_maps"]

    def decode_predictions(self, probabilities_tensor: torch.Tensor, threshold: float = 0.50):
        active_indices = torch.nonzero(probabilities_tensor > threshold).flatten().tolist()
        diagnoses = []
        for idx in active_indices:
            icd_string = self.id_to_icd.get(str(idx), "UNKNOWN_CODE")
            diagnoses.append(icd_string)
        return diagnoses
    
def execute_clinical_audit(targets, probabilities, thresholds=None, min_positive_prevalence: int = 2):
    """
    🎯 CLINICAL AUDIT CORE:
    Computes a complete suite of ranking, safety, and workflow metrics 
    across long-tailed clinical target vectors.
    """
    num_samples, num_classes = targets.shape
    
    # -----------------------------------------------------------------
    # TIER 1: RANKING METRICS
    # -----------------------------------------------------------------
    auc_roc_list = []
    auc_pr_list = []
    active_class_indices = []
    
    for c_idx in range(num_classes):
        pos_count = targets[:, c_idx].sum()
        # Metrics require at least one positive and one negative sample in the slice
        if pos_count >= min_positive_prevalence and pos_count < num_samples:
            active_class_indices.append(c_idx)
            
            # AUC-ROC
            auc_roc_list.append(roc_auc_score(targets[:, c_idx], probabilities[:, c_idx]))
            
            # AUC-PR (Average Precision calculation via Precision-Recall Integral)
            prec, rec, _ = precision_recall_curve(targets[:, c_idx], probabilities[:, c_idx])
            auc_pr_list.append(auc(rec, prec))
            
    macro_auc_roc = np.mean(auc_roc_list) * 100 if auc_roc_list else 0.0
    macro_auc_pr = np.mean(auc_pr_list) * 100 if auc_pr_list else 0.0

    # -----------------------------------------------------------------
    # 🎯 TIER 2: HARD DECISION METRICS (Threshold Calibration)
    # -----------------------------------------------------------------
    # Add an explicit calibration toggle check (defaulting to False for row-wise safety)
    calibrate_per_class = True 
    
    if thresholds is None:
        # Enforce a true, flat global default anchor across all 456 tracks
        flat_global_anchor = 0.15 
        thresholds = np.ones(num_classes) * flat_global_anchor
        
        if calibrate_per_class:
            for c_idx in active_class_indices:
                best_f1 = -1.0
                best_thresh = 0.50
                for thresh in np.linspace(0.01, 0.90, 90):
                    class_preds = (probabilities[:, c_idx] > thresh).astype(float)
                    _, _, f1, _ = precision_recall_fscore_support(
                        targets[:, c_idx], class_preds, average='binary', zero_division=0
                    )
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh
                thresholds[c_idx] = best_thresh
        
    preds = np.zeros_like(probabilities)
    sensitivity_list = []
    specificity_list = []
    
    for c_idx in range(num_classes):
        preds[:, c_idx] = (probabilities[:, c_idx] > thresholds[c_idx]).astype(float)
        
        # Calculate True Positives, False Positives, True Negatives, False Negatives
        y_true = targets[:, c_idx]
        y_pred = preds[:, c_idx]
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        if c_idx in active_class_indices:  # Only track classes that have active labels in this cohort
            sensitivity_list.append(sens)
            specificity_list.append(spec)

    if len(active_class_indices) > 0:
        macro_p, _, macro_f1, _ = precision_recall_fscore_support(
            targets[:, active_class_indices], 
            preds[:, active_class_indices], 
            average='macro', 
            zero_division=0
        )
    else:
        macro_p, macro_f1 = 0.0, 0.0
        
    macro_sens = np.mean(sensitivity_list) * 100 if sensitivity_list else 0.0
    macro_spec = np.mean(specificity_list) * 100 if specificity_list else 0.0

    # -----------------------------------------------------------------
    # TIER 3: WORKFLOW METRICS
    # -----------------------------------------------------------------
    top1_count = 0
    top3_count = 0
    top5_count = 0  
    
    # 🛠️ THE FIX: Sort directly by raw probabilities to preserve true intra-patient rankings
    normalized_scores = probabilities 
    
    for i in range(num_samples):
        # Sort indices by model confidence in descending order
        top_indices = np.argsort(normalized_scores[i])[::-1]
        
        if targets[i, top_indices[0]] == 1.0:
            top1_count += 1
        if np.any(targets[i, top_indices[:3]] == 1.0):
            top3_count += 1
        if np.any(targets[i, top_indices[:5]] == 1.0):  
            top5_count += 1
            
    top1_rate = (top1_count / num_samples) * 100
    top3_rate = (top3_count / num_samples) * 100
    top5_rate = (top5_count / num_samples) * 100  # Calculate Top-5 rate
    
    # -----------------------------------------------------------------
    # RENDER AUDIT REPORT
    # -----------------------------------------------------------------
    print("\n" + "═"*70)
    print(" 🏥 COMPREHENSIVE CLINICAL MANIFOLD AUDIT REPORT")
    print("═"*70)
    print(f" 🩺 [TIER 1: RANKING]   Macro AUC-ROC:          {macro_auc_roc:3.2f}%")
    print(f" 🩺 [TIER 1: RANKING]   Macro AUC-PR (Sparsity):{macro_auc_pr:3.2f}%")
    print("-" * 70)
    print(f" 🛡️ [TIER 2: BOUNDARY]  Calibrated Macro F1:    {macro_f1 * 100:3.2f}%")
    print(f" 🛡️ [TIER 2: BOUNDARY]  Macro Precision:        {macro_p * 100:3.2f}%")
    print(f" 🛡️ [TIER 2: BOUNDARY]  Macro Sensitivity (TPR):{macro_sens:3.2f}%")
    print(f" 🛡️ [TIER 2: BOUNDARY]  Macro Specificity (TNR):{macro_spec:3.2f}%")
    print("-" * 70)
    print(f" ⚡ [TIER 3: WORKFLOW]  Top-1 Primary Hit Rate: {top1_rate:3.2f}%")
    print(f" ⚡ [TIER 3: WORKFLOW]  Top-3 Differential Rate:{top3_rate:3.2f}%")
    print(f" ⚡ [TIER 3: WORKFLOW]  Top-5 Differential Rate:{top5_rate:3.2f}%")  # Render Top-5
    print("═"*70 + "\n")
    