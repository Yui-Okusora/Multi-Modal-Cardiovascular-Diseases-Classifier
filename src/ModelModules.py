# src/Zone1JEPA.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_recall_fscore_support
import numpy as np

class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, d_model: int): # Removed dropout_p
        super().__init__()
        self.d_time = d_model
        half_dim = d_model // 2
        frequencies = torch.exp(
            torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000.0) / (half_dim - 1))
        )
        self.register_buffer("frequencies", frequencies)
        
        self.time_norm = nn.LayerNorm(d_model)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        scaled_t = t.unsqueeze(-1) * self.frequencies
        sin_t = torch.sin(scaled_t)
        cos_t = torch.cos(scaled_t)
        time_features = torch.cat([sin_t, cos_t], dim=-1)
        
        # Ground and return clean coordinates
        return self.time_norm(time_features)

class UnifiedSystemicTokenizer(nn.Module):
    def __init__(self, num_total_features: int, num_cat_results: int, d_model: int = 256):
        super().__init__()
        self.feature_embedding = nn.Embedding(num_total_features, d_model)
        self.numeric_projection = nn.Linear(1, d_model, bias=False)
        self.numeric_norm = nn.LayerNorm(d_model)
        self.cat_result_embedding = nn.Embedding(num_cat_results, d_model, padding_idx=0)
        
        # Clean time embedder without stochastic channels
        self.time_embedder = ContinuousTimeEmbedding(d_model)
        
        self.global_token_norm = nn.LayerNorm(d_model)

    def forward(self, feature_ids, numeric_values, cat_result_ids, timestamps):
        feat_emb = self.feature_embedding(feature_ids)
        cat_emb = self.cat_result_embedding(cat_result_ids)
        time_emb = self.time_embedder(timestamps)
        
        # ─── FEATURE LINEAR MODULATION (FiLM) GATING ───
        # Project the continuous values and map them to a [0, 1] scaling gate
        val_projected = self.numeric_projection(numeric_values.unsqueeze(-1))
        val_gate = torch.sigmoid(self.numeric_norm(val_projected))
        
        # Multiply instead of adding! The numeric magnitude now directly modulates the token energy
        modulated_features = feat_emb * val_gate
        
        combined_tokens = modulated_features + cat_emb + time_emb
        return self.global_token_norm(combined_tokens)

class PerceiverLatentPooling(nn.Module):
    def __init__(self, num_slots: int, d_model: int, nheads: int = 8):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        
        # 1. Base Latent Slots
        self.latent_slots = nn.Parameter(torch.empty(num_slots, d_model))
        self.slot_pos_embeddings = nn.Parameter(torch.empty(num_slots, d_model))
        
        self._reset_parameters()
        
        self.slot_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, batch_first=True)
        self.latent_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model)
        
        # ─── PIPELINE STEP C: LATENT EXPRESSION FFN (NO DROPOUT) ───
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def _reset_parameters(self):
        nn.init.orthogonal_(self.latent_slots, gain=1.0)
        nn.init.orthogonal_(self.slot_pos_embeddings, gain=0.1)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = x.size(0)
        
        ordered_slots = self.latent_slots + self.slot_pos_embeddings
        norm_slots = self.slot_norm(ordered_slots).unsqueeze(0).expand(batch_size, -1, -1)
        norm_x = self.kv_norm(x)
        
        if padding_mask is not None:
            all_padded_rows = padding_mask.all(dim=-1, keepdim=True)
            padding_mask = padding_mask.masked_fill(all_padded_rows, False)

        # Step A: Pull temporal data into ordered slots
        attn_out, _ = self.cross_attn(
            query=norm_slots, 
            key=norm_x, 
            key_padding_mask=padding_mask,
            value=norm_x
        )
        slots = norm_slots + attn_out
        
        # Step B: Latent Self-Attention (Slots communicate to unmix features)
        norm_slots_self = self.self_attn_norm(slots)
        self_attn_out, _ = self.latent_self_attn(
            query=norm_slots_self,
            key=norm_slots_self,
            value=norm_slots_self
        )
        slots = slots + self_attn_out
        
        # Step C: Post-attention non-linear coordinate projection (100% deterministic)
        slots = slots + self.feed_forward(self.ff_norm(slots))
        
        return slots

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
        self.is_teacher = True

    def forward(self, feature_ids, numeric_values, cat_result_ids, timestamps, padding_mask=None):
        x_tokens = self.tokenizer(feature_ids, numeric_values, cat_result_ids, timestamps)
        # Process the future sequence context before pooling
        h_seq = self.temporal_backbone(x_tokens, src_key_padding_mask=padding_mask)
        z_t = self.perceiver_pool(h_seq, padding_mask=padding_mask)
        z_t_stable = self.output_norm(z_t)
        return torch.nn.functional.normalize(z_t_stable, p=2, dim=-1)

class Predictor(nn.Module):
    """🎯 Whole-Slot Predictor: Handles matrix-to-matrix cross-slot translation.
    
    Upgraded to use a Permutation-Invariant Self-Attention layout, forcing
    slots to route context based on content affinity rather than rigid indices.
    """
    def __init__(self, num_slots: int = 24, d_model: int = 512, nhead: int = 8):
        super().__init__()
        # Retain your original channel-wise transformations for element capacity
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 🌀 Content-driven slot mixing block
        # norm_first=True is highly recommended to stabilize bfloat16 mixed precision
        self.slot_mixer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        
        # 🛡️ THE NEW SHIELD: Stabilizes hidden coordinate variance across channels.
        # Replaces F.normalize to preserve vector magnitudes for downstream probes.
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, z_c):
        # z_c input shape: [Batch, Num_Slots, d_model]
        
        # Step 1: Element-wise feature translation
        z_predicted = self.channel_mlp(z_c) 
        
        # Step 2: Permutation-invariant attention sweep across the slot axis
        # Matrix commutes flawlessly with any random slot indexing changes
        z_out = self.slot_mixer(z_predicted)
        
        # Step 3: Clean residual aggregation + scale stabilization
        # Preserves the true mathematical norm (entropy) of your clinical markers
        return self.final_norm(z_predicted + z_out)

class PatientManifoldAssembler(nn.Module):
    """
    🧬 PATIENT MANIFOLD ASSEMBLER (SCALE-STABILIZED)
    Stitches static covariates safely without drowning out the timeline norm.
    """
    def __init__(self, num_cat_results: int, latent_dim: int = 512):
        super().__init__()
        self.age_projector = nn.Linear(in_features=1, out_features=latent_dim)
        self.gender_embed = nn.Embedding(num_embeddings=num_cat_results, embedding_dim=latent_dim)
        
        # 🛡️ Add scale stabilizers to match the L2 magnitude of the encoder slots
        self.age_norm = nn.LayerNorm(latent_dim)
        self.gender_norm = nn.LayerNorm(latent_dim)

    def forward(self, z_c_raw: torch.Tensor, age: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
        # z_c_raw shape: [Batch, 24, 512] (Strict L2 norm = 1.0)
        
        # Project and strictly normalize demographics to match the timeline slot scale
        z_age = self.age_norm(self.age_projector(age.unsqueeze(-1))).unsqueeze(1)
        z_gender = self.gender_norm(self.gender_embed(gender)).unsqueeze(1)
        
        # Optionally apply a scaling factor (e.g., 0.1) if you want demographics to act purely as a subtle prior
        z_age = torch.nn.functional.normalize(z_age, p=2, dim=-1) * 0.40
        z_gender = torch.nn.functional.normalize(z_gender, p=2, dim=-1) * 0.40
        
        # Stitch them cleanly -> Output Shape: [Batch, 26, 512]
        return torch.cat([z_c_raw, z_age, z_gender], dim=1)

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
        self.feature_dropout = nn.Dropout(p=0.15)
        self.classifier = nn.Linear(input_flat_dim, num_classes)

    def forward(self, z_hat_slots: torch.Tensor) -> torch.Tensor:
        flat_z = z_hat_slots.contiguous().view(z_hat_slots.size(0), -1)
        return self.classifier(self.feature_dropout(flat_z))
    
class LabelAttentiveSlotProbe(nn.Module):
    """
    🎯 SOTA LAAT-INSPIRED PROBE HEAD FOR PERCEIVER SLOTS
    Each class queries the 24 slots individually to extract its own tailored representation.
    """
    def __init__(self, in_slots: int = 24, in_dim: int = 512, num_classes: int = 456):
        super().__init__()
        self.in_slots = in_slots
        self.in_dim = in_dim
        self.num_classes = num_classes
        
        self.class_embeddings = nn.Parameter(torch.empty(num_classes, in_dim))
        self.query_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.weight_class = nn.Parameter(torch.empty(num_classes, in_dim))
        self.bias_class = nn.Parameter(torch.empty(num_classes))
        
        nn.init.xavier_uniform_(self.class_embeddings)
        nn.init.xavier_uniform_(self.weight_class)
        nn.init.zeros_(self.bias_class)
        self.dropout = nn.Dropout(p=0.20)

    def forward(self, z_hat_slots: torch.Tensor) -> torch.Tensor:
        batch_size = z_hat_slots.size(0)
        queries = self.query_proj(self.class_embeddings)
        
        Q = queries.unsqueeze(0).expand(batch_size, -1, -1)
        K, V = z_hat_slots, z_hat_slots
        
        class_specific_contexts = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False, scale=1.0 
        )
        
        return torch.sum(class_specific_contexts * self.weight_class.unsqueeze(0), dim=-1) + self.bias_class.unsqueeze(0)
    
class AuxiliaryCardinalityHead(nn.Module):
    """
    🎯 ATTENTIVE SLOT-AGGREGATION POOLING HEAD:
    Eliminates brute-force flattening bottlenecks. Treats Perceiver slots as 
    a permutation-invariant set, tracking clinical density signatures 
    natively with a microscopic parameter footprint.
    """
    def __init__(self, in_slots: int = 24, in_dim: int = 512, hidden_dim: int = 64):
        super().__init__()
        self.in_slots = in_slots
        self.in_dim = in_dim
        
        self.slot_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.attention_pool = nn.Linear(hidden_dim, 1, bias=False)
        self.output_projector = nn.Linear(hidden_dim, 1)

    def forward(self, z_hat_slots: torch.Tensor) -> torch.Tensor:
        h_slots = self.slot_net(z_hat_slots)
        attn_logits = self.attention_pool(h_slots) 
        attn_weights = F.softmax(attn_logits, dim=1) 
        pooled_context = torch.sum(h_slots * attn_weights, dim=1)
        return self.output_projector(pooled_context).squeeze(-1)

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

class ClassAwareASLWithLogitAdjustment(nn.Module):
    def __init__(self, class_frequencies, tau=1.0, gamma_pos=0.0, gamma_neg_base=4.0, beta_neg=2.5):
        super().__init__()
        self.tau = tau
        self.gamma_pos = gamma_pos
        
        # 🚀 NEW PRECISION TUNING FACTOR: Weights the cost of false positives
        # Setting this above 1.0 forces backprop to prioritize Precision/PR-AUC
        self.beta_neg = beta_neg
        
        # Binary log-odds ratio prior adjustment vector
        log_odds = torch.log(class_frequencies / (1.0 - class_frequencies + 1e-7))
        self.register_buffer('prior_offset', self.tau * log_odds)
        
        # Smooth inverse frequency scaling for negative focusing
        self.register_buffer('gamma_neg', gamma_neg_base + (1.0 - class_frequencies) * 2.0)
        
    def forward(self, logits, targets, is_training=True):
        if is_training:
            logits = logits - self.prior_offset
            
        # FP32 numerical precision protection layer
        probs = torch.sigmoid(logits).float()
        targets = targets.float()
        
        loss_pos = targets * torch.log(probs + 1e-7) * (1.0 - probs).pow(self.gamma_pos)
        loss_neg = (1.0 - targets) * torch.log(1.0 - probs + 1e-7) * probs.pow(self.gamma_neg)
        
        # 🚀 THE MODEL-LEVEL CURE: Scale up the penalty for false positives
        # This directly forces backpropagation to clean up the PR-AUC curve
        batch_loss = - (loss_pos + (self.beta_neg * loss_neg))
        
        return batch_loss.sum(dim=-1).mean()

@torch.no_grad()
def compute_comprehensive_manifold_diagnostics(z: torch.Tensor):
    """
    🚀 UNIFIED MANIFOLD ANALYTICS ENGINE:
    Combines structural slot health audits with centered, shielded 
    Effective Rank calculations. Accepts 3D [B, K, D] or 2D [B, D] tensors.
    """
    z = z.float()

    # 🛡️ SVD PLATFORM PROTECTION SHIELD
    if not torch.isfinite(z).all():
        z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
        
    # Parse inputs based on active tensor dimensionality
    if z.dim() == 3:
        B, K, D = z.shape
        # 1. Compute average batch standard deviation
        var_per_dim = z.var(dim=0)
        mean_batch_std = torch.sqrt(var_per_dim + 1e-6).mean().item()
        
        # 2. Monitor cross-slot similarity correlations
        z_norm = F.normalize(z, p=2, dim=-1)
        slot_sim = torch.bmm(z_norm, z_norm.transpose(1, 2))
        triu_indices = torch.triu_indices(K, K, offset=1, device=z.device)
        mean_slot_cross_talk = slot_sim[:, triu_indices[0], triu_indices[1]].mean().item()
        
        # Unroll sequential dimensions for global manifold matrix checks
        matrix_for_svd = z.contiguous().view(B * K, D)
        z_for_sparsity = z.mean(dim=1)  # Mean-pooled row for Hoyer index calculation
    else:
        # Handle 2D population inputs gracefully (e.g., downstream pooled states)
        mean_batch_std = z.std(dim=0).mean().item()
        mean_slot_cross_talk = 0.0
        matrix_for_svd = z
        z_for_sparsity = z
        
    # 3. Hoyer Sparsity Index Calculation
    sparsity_index = torch.mean(
        torch.norm(z_for_sparsity, p=1, dim=1) / 
        (torch.norm(z_for_sparsity, p=2, dim=1) + 1e-8)
    ).item()

    # 4. Centered, Spectral Covariance Effective Rank
    if matrix_for_svd.size(0) <= 1:
        return {"batch_std": mean_batch_std, "slot_cross_talk": mean_slot_cross_talk, "effective_rank": float(matrix_for_svd.size(-1)), "sparsity_index": sparsity_index}
        
    # Force mean subtraction to compute true variance orientations
    z_centered = matrix_for_svd - matrix_for_svd.mean(dim=0, keepdim=True)

    try:
        _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
        p = S / (S.sum() + 1e-10)
        effective_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-10))).item()
    except Exception:
        effective_rank = float('nan')
        
    return {
        "batch_std": mean_batch_std,
        "slot_cross_talk": mean_slot_cross_talk,
        "effective_rank": effective_rank,
        "sparsity_index": sparsity_index
    }

def execute_clinical_audit(targets, probabilities, predicted_cardinalities=None, thresholds=None, min_positive_prevalence: int = 2, calibrate_per_class: bool = True, silent: bool = False):
    """
    🎯 CLINICAL AUDIT CORE (DYNAMIC ADAPTIVE ENGINE):
    Computes ranking, safety, and workflow metrics across long-tailed targets.
    Fully upgraded to accept a continuous stream of predicted patient diagnostic counts
    to calculate dynamic, patient-specific Adaptive-K workflow metrics.
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
        if pos_count >= min_positive_prevalence and pos_count < num_samples:
            active_class_indices.append(c_idx)
            auc_roc_list.append(roc_auc_score(targets[:, c_idx], probabilities[:, c_idx]))
            prec, rec, _ = precision_recall_curve(targets[:, c_idx], probabilities[:, c_idx])
            auc_pr_list.append(auc(rec, prec))
            
    macro_auc_roc = np.mean(auc_roc_list) * 100 if auc_roc_list else 0.0
    macro_auc_pr = np.mean(auc_pr_list) * 100 if auc_pr_list else 0.0

    if len(active_class_indices) > 0:
        micro_auc_roc = roc_auc_score(
            targets[:, active_class_indices], 
            probabilities[:, active_class_indices], 
            average='micro'
        ) * 100
    else:
        micro_auc_roc = 0.0

    # High-speed breakout bypass path for early-epoch pretraining validation checks
    if silent and not calibrate_per_class and predicted_cardinalities is None:
        return {
            "macro_auc_roc": macro_auc_roc,
            "macro_auc_pr": macro_auc_pr,
            "micro_auc_roc": micro_auc_roc
        }

    # -----------------------------------------------------------------
    # 🎯 TIER 2: HARD DECISION METRICS (Target-Bounded Threshold Calibration)
    # -----------------------------------------------------------------
    if thresholds is None:
        flat_global_anchor = 0.15 
        thresholds = np.ones(num_classes) * flat_global_anchor
        
        if calibrate_per_class:
            for c_idx in active_class_indices:
                best_score = -1.0
                best_thresh = 0.50
                y_true = targets[:, c_idx]
                
                # Scan precision-recall space across the valid clinical logit distribution
                for thresh in np.linspace(0.0001, 0.99, 200): # Expanded resolution up to 0.99
                    class_preds = (probabilities[:, c_idx] > thresh).astype(float)
                    
                    tp = np.sum((y_true == 1) & (class_preds == 1))
                    fp = np.sum((y_true == 0) & (class_preds == 1))
                    tn = np.sum((y_true == 0) & (class_preds == 0))
                    fn = np.sum((y_true == 1) & (class_preds == 0))
                    
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    
                    # ─── 🚀 PREVALENCE-INDEPENDENT BARRIER STRATIFICATION ───
                    
                    # Tier 1: The Holy Grail (All constraints cleared)
                    if sens >= 0.85 and prec >= 0.85 and spec >= 0.95:
                        # Tie-breaker prioritizes tight convergence grouping
                        variance_penalty = np.var([sens, prec, spec])
                        score = 100.0 + (sens + prec + spec) - variance_penalty
                        
                    # Tier 2: Precision Rescue (Specificity is high enough to save PPV)
                    elif spec >= 0.990 and sens >= 0.70:
                        score = 50.0 + (sens * 2.0) + prec
                        
                    # Tier 3: Geometric Balancing Fallback
                    else:
                        # Maximizes Bookmaker Informedness (Informedness = Sens + Spec - 1)
                        # Multiplied by precision to keep background noise tightly locked
                        informedness = sens + spec - 1.0
                        score = max(0.0, informedness) * (prec + 1e-5)
                        
                    if score > best_score:
                        best_score = score
                        best_thresh = thresh
                        
                thresholds[c_idx] = best_thresh
        
    preds = np.zeros_like(probabilities)
    sensitivity_list = []
    specificity_list = []
    
    for c_idx in range(num_classes):
        preds[:, c_idx] = (probabilities[:, c_idx] > thresholds[c_idx]).astype(float)
        y_true = targets[:, c_idx]
        y_pred = preds[:, c_idx]
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        if c_idx in active_class_indices:
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
    # ⚡ TIER 3: DUAL-PARADIGM WORKFLOW METRICS (FIXED + ADAPTIVE)
    # -----------------------------------------------------------------
    hit1_count, hit3_count, hit5_count, hit8_count = 0, 0, 0, 0
    p1_patient_scores, p3_patient_scores, p5_patient_scores, p8_patient_scores = [], [], [], []
    
    adaptive_hit_count = 0
    adaptive_precision_scores = []
    
    for i in range(num_samples):
        top_indices = np.argsort(probabilities[i])[::-1]
        
        if targets[i, top_indices[0]] == 1.0: hit1_count += 1
        if np.any(targets[i, top_indices[:3]] == 1.0): hit3_count += 1
        if np.any(targets[i, top_indices[:5]] == 1.0): hit5_count += 1
        if np.any(targets[i, top_indices[:8]] == 1.0): hit8_count += 1
            
        p1_patient_scores.append(np.sum(targets[i, top_indices[:1]]) / 1.0)
        p3_patient_scores.append(np.sum(targets[i, top_indices[:3]]) / 3.0)
        p5_patient_scores.append(np.sum(targets[i, top_indices[:5]]) / 5.0)
        p8_patient_scores.append(np.sum(targets[i, top_indices[:8]]) / 8.0)
        
        if predicted_cardinalities is not None:
            k_adaptive = max(1, int(np.round(predicted_cardinalities[i])))
        else:
            k_adaptive = max(1, int(np.sum(probabilities[i] > thresholds)))
            
        adaptive_slice = top_indices[:k_adaptive]
        
        if np.any(targets[i, adaptive_slice] == 1.0):
            adaptive_hit_count += 1
            
        adaptive_precision_scores.append(np.sum(targets[i, adaptive_slice]) / float(k_adaptive))
            
    top1_rate = (hit1_count / num_samples) * 100
    top3_rate = (hit3_count / num_samples) * 100
    top5_rate = (hit5_count / num_samples) * 100
    top8_rate = (hit8_count / num_samples) * 100
    
    pat1 = np.mean(p1_patient_scores) * 100
    pat3 = np.mean(p3_patient_scores) * 100
    pat5 = np.mean(p5_patient_scores) * 100
    pat8 = np.mean(p8_patient_scores) * 100
    
    adaptive_hit_rate = (adaptive_hit_count / num_samples) * 100
    adaptive_precision = np.mean(adaptive_precision_scores) * 100
    
    metrics_summary = {
        "macro_auc_roc": macro_auc_roc,
        "micro_auc_roc": micro_auc_roc,
        "macro_auc_pr": macro_auc_pr,
        "macro_f1": macro_f1 * 100,
        "macro_precision": macro_p * 100,
        "macro_sensitivity": macro_sens,
        "macro_specificity": macro_spec,
        "top1_rate": top1_rate,
        "top3_rate": top3_rate,
        "top5_rate": top5_rate,
        "top8_rate": top8_rate,
        "precision_at_1": pat1,
        "precision_at_3": pat3,
        "precision_at_5": pat5,
        "precision_at_8": pat8,
        "adaptive_hit_rate": adaptive_hit_rate,
        "adaptive_precision": adaptive_precision,
        "calibrated_thresholds": thresholds
    }

    if not silent:
        print("\n" + "═"*75)
        print(" 🏥 COMPREHENSIVE CLINICAL MANIFOLD AUDIT REPORT")
        print("═"*75)
        print(f" 🩺 [TIER 1: RANKING]   Macro AUC-ROC:          {macro_auc_roc:3.2f}%")
        print(f" 🩺 [TIER 1: RANKING]   Micro AUC-ROC:          {micro_auc_roc:3.2f}%")
        print(f" 🩺 [TIER 1: RANKING]   Macro AUC-PR (Sparsity):{macro_auc_pr:3.2f}%")
        print("-" * 75)
        print(f" 🛡️ [TIER 2: BOUNDARY]  Calibrated Macro F1:    {metrics_summary['macro_f1']:3.2f}%")
        print(f" 🛡️ [TIER 2: BOUNDARY]  Macro Precision:        {metrics_summary['macro_precision']:3.2f}%")
        print(f" 🛡️ [TIER 2: BOUNDARY]  Macro Sensitivity (TPR):{macro_sens:3.2f}%")
        print(f" 🛡️ [TIER 2: BOUNDARY]  Macro Specificity (TNR):{macro_spec:3.2f}%")
        print("-" * 75)
        print(f" 🚀 [ADAPTIVE HORIZON]  Hit Rate (Safety Net):  {adaptive_hit_rate:3.2f}%")
        print(f" 🚀 [ADAPTIVE HORIZON]  Precision (Density):    {adaptive_precision:3.2f}%")
        print("-" * 75)
        print(f" ⚡ [TIER 3: FIXED K]   Top-1 Primary Hit Rate: {top1_rate:3.2f}% │ Precision@1: {pat1:3.2f}%")
        print(f" ⚡ [TIER 3: FIXED K]   Top-3 Differential Rate:{top3_rate:3.2f}% │ Precision@3: {pat3:3.2f}%")
        print(f" ⚡ [TIER 3: FIXED K]   Top-5 Differential Rate:{top5_rate:3.2f}% │ Precision@5: {pat5:3.2f}%")
        print(f" ⚡ [TIER 3: FIXED K]   Top-8 Differential Rate:{top8_rate:3.2f}% │ Precision@8: {pat8:3.2f}%")
        print("═"*75 + "\n")
        
    return metrics_summary
    