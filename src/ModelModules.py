# src/ModelModules.py
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_recall_fscore_support
from typing import Tuple, Dict, Any, List, Optional, Union


# ==================================================================================================
# 1. CONTINUOUS TIME EMBEDDING MODULE
# ==================================================================================================
class ContinuousTimeEmbedding(nn.Module):
    r"""
    DESCRIPTION:
    ------------
    Maps continuous-time lookback deltas (e.g., hours elapsed since an event) into a dense, 
    high-dimensional Fourier feature vector.

    TECHNOLOGY & MATHEMATICAL FOUNDATION:
    --------------------------------------
    Extends Vaswani-style sinusoidal positional encodings to non-discrete, continuous scalar domain t:
        PE(t, 2i)   = sin(t * \omega_i)
        PE(t, 2i+1) = cos(t * \omega_i)
    where frequencies \omega_i = exp(- (2i / d_model) * log(10000)). 
    This preserves exact temporal distances across irregular longitudinal clinical check-ins.
    A LayerNorm layer stabilizes feature variance across time frequencies.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_time = d_model
        half_dim = d_model // 2
        
        # Exponentially decaying frequency spectrum
        frequencies = torch.exp(
            torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000.0) / (half_dim - 1))
        )
        self.register_buffer("frequencies", frequencies)
        self.time_norm = nn.LayerNorm(d_model)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of shape [B, L] containing continuous temporal lookback deltas.
        Returns:
            Tensor of shape [B, L, d_model] containing normalized time features.
        """
        scaled_t = t.unsqueeze(-1) * self.frequencies
        sin_t = torch.sin(scaled_t)
        cos_t = torch.cos(scaled_t)
        time_features = torch.cat([sin_t, cos_t], dim=-1)
        return self.time_norm(time_features)


class PeriodicNumericalEmbedding(nn.Module):
    """
    Expands normalized scalar v into a multi-frequency Fourier representation
    to match the high-rank space of embedding lookup tables.
    """
    def __init__(self, d_model: int = 512, num_frequencies: int = 32):
        super().__init__()
        frequencies = torch.exp(
            torch.arange(num_frequencies, dtype=torch.float32) * -(math.log(10.0) / max(1, num_frequencies - 1))
        )
        self.register_buffer("frequencies", frequencies)
        
        self.mlp = nn.Sequential(
            nn.Linear(num_frequencies * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v_scaled = v.unsqueeze(-1) * self.frequencies
        fourier_feat = torch.cat([torch.sin(v_scaled), torch.cos(v_scaled)], dim=-1)
        return self.mlp(fourier_feat)


# ==================================================================================================
# 2. UNIFIED SYSTEMIC TOKENIZER
# ==================================================================================================
class UnifiedSystemicTokenizer(nn.Module):
    """
    DESCRIPTION:
    ------------
    Hybrid UnifiedSystemicTokenizer combining High-Rank Periodic Fourier numeric 
    encoding with Feature-wise Linear Modulation (FiLM) conditioned on clinical concept IDs.
    """
    def __init__(self, num_total_features: int, num_cat_results: int, d_model: int = 512, num_frequencies: int = 32):
        super().__init__()
        self.feature_embedding = nn.Embedding(num_total_features, d_model)
        self.numeric_encoder = PeriodicNumericalEmbedding(d_model=d_model, num_frequencies=num_frequencies)
        self.film_gamma = nn.Linear(d_model, d_model)
        self.cat_result_embedding = nn.Embedding(num_cat_results, d_model, padding_idx=0)
        self.time_embedder = ContinuousTimeEmbedding(d_model)
        self.global_token_norm = nn.LayerNorm(d_model)

    def forward(
        self, 
        feature_ids: torch.Tensor, 
        numeric_values: torch.Tensor, 
        cat_result_ids: torch.Tensor, 
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        feat_emb = self.feature_embedding(feature_ids)
        cat_emb  = self.cat_result_embedding(cat_result_ids)
        time_emb = self.time_embedder(timestamps)
        
        num_emb  = self.numeric_encoder(numeric_values)
        gamma = torch.sigmoid(self.film_gamma(feat_emb))
        modulated_num_emb = gamma * num_emb
        
        has_numeric_mask = (cat_result_ids == 0).unsqueeze(-1).float()
        modulated_num_emb = modulated_num_emb * has_numeric_mask
        
        combined_tokens = feat_emb + modulated_num_emb + cat_emb + time_emb
        return self.global_token_norm(combined_tokens)


# ==================================================================================================
# 3. PERCEIVER LATENT POOLING MODULE
# ==================================================================================================
class PerceiverLatentPooling(nn.Module):
    """
    DESCRIPTION:
    ------------
    Asymmetric latent bottleneck layer that pools dynamic-length clinical timelines into 
    a fixed-size latent matrix [B, num_slots, d_model] (typically 24 slots of dimension 512).

    TECHNOLOGY & MATHEMATICAL FOUNDATION:
    --------------------------------------
    Based on the Perceiver Architecture (Jaegle et al., 2021):
      1. Cross-Attention: 24 learned latent query slots query the variable-length sequence [B, L, d_model].
      2. Latent Self-Attention: Slots communicate among themselves to unmix independent clinical themes.
      3. Latent Feed-Forward (FFN): Deterministic non-linear coordinate projection.
    """
    def __init__(self, num_slots: int, d_model: int, nheads: int = 8):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        
        self.latent_slots = nn.Parameter(torch.empty(num_slots, d_model))
        self.slot_pos_embeddings = nn.Parameter(torch.empty(num_slots, d_model))
        self._reset_parameters()
        
        self.slot_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, batch_first=True)
        self.latent_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model)
        
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

        attn_out, _ = self.cross_attn(
            query=norm_slots, key=norm_x, value=norm_x, key_padding_mask=padding_mask
        )
        slots = norm_slots + attn_out
        
        norm_slots_self = self.self_attn_norm(slots)
        self_attn_out, _ = self.latent_self_attn(
            query=norm_slots_self, key=norm_slots_self, value=norm_slots_self
        )
        slots = slots + self_attn_out
        slots = slots + self.feed_forward(self.ff_norm(slots))
        return slots


# ==================================================================================================
# 4. CONTEXT ENCODER (STUDENT STREAM)
# ==================================================================================================
class ContextEncoder(nn.Module):
    r"""
    DESCRIPTION:
    ------------
    The deep sequential backbone of the JEPA student stream processing sliding event windows.
    """
    def __init__(
        self, 
        num_total_features: int, 
        num_cat_results: int, 
        d_model: int = 512, 
        num_slots: int = 24, 
        nlayers: int = 6
    ):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = UnifiedSystemicTokenizer(num_total_features, num_cat_results, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 4, 
            batch_first=True, activation='gelu', norm_first=True
        )
        self.temporal_backbone = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.perceiver_pool = PerceiverLatentPooling(num_slots=num_slots, d_model=d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self, 
        feature_ids: torch.Tensor, 
        numeric_values: torch.Tensor, 
        cat_result_ids: torch.Tensor, 
        timestamps: torch.Tensor, 
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        x_events = self.tokenizer(feature_ids, numeric_values, cat_result_ids, timestamps)
        h_seq = self.temporal_backbone(x_events, src_key_padding_mask=padding_mask)
        z_c = self.perceiver_pool(h_seq, padding_mask=padding_mask)
        z_c_stable = self.output_norm(z_c)
        z_c_normalized = F.normalize(z_c_stable, p=2, dim=-1) * math.sqrt(self.d_model)
        return z_c_normalized


TargetEncoder = ContextEncoder


# ==================================================================================================
# 5. WORLD-MODEL PREDICTOR
# ==================================================================================================
class Predictor(nn.Module):
    r"""
    DESCRIPTION:
    ------------
    The world-model transition function in the JEPA architecture. Maps current student latent slots 
    to predicted future target representations.
    """
    def __init__(self, num_slots: int = 24, d_model: int = 512, nhead: int = 8):
        super().__init__()
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        self.slot_mixer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.0, activation=F.gelu, batch_first=True, norm_first=True
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, z_c: torch.Tensor) -> torch.Tensor:
        z_predicted = self.channel_mlp(z_c)
        z_out = self.slot_mixer(z_predicted)
        return self.final_norm(z_predicted + z_out)


# ==================================================================================================
# 6. PATIENT MANIFOLD ASSEMBLER
# ==================================================================================================
class PatientManifoldAssembler(nn.Module):
    """
    DESCRIPTION:
    ------------
    Stitches static patient covariates (Age, Gender) onto dynamic clinical timeline slots to 
    construct the complete **26-token Patient State Manifold**.
    """
    def __init__(self, num_cat_results: int, latent_dim: int = 512, covariate_scale: float = 0.50):
        super().__init__()
        self.latent_dim = latent_dim
        self.covariate_scale = covariate_scale
        
        self.age_projector = nn.Linear(in_features=1, out_features=latent_dim)
        self.gender_embed = nn.Embedding(num_embeddings=num_cat_results, embedding_dim=latent_dim)
        
        self.age_norm = nn.LayerNorm(latent_dim)
        self.gender_norm = nn.LayerNorm(latent_dim)

    def forward(self, z_c_raw: torch.Tensor, age: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
        z_age = self.age_norm(self.age_projector(age.unsqueeze(-1))).unsqueeze(1)
        z_gender = self.gender_norm(self.gender_embed(gender)).unsqueeze(1)
        
        scale = math.sqrt(self.latent_dim)
        z_age = F.normalize(z_age, p=2, dim=-1) * (scale * self.covariate_scale)
        z_gender = F.normalize(z_gender, p=2, dim=-1) * (scale * self.covariate_scale)
        
        return torch.cat([z_c_raw, z_age, z_gender], dim=1)


# ==================================================================================================
# 7. LINEAR PROBE HEAD
# ==================================================================================================
class LinearProbeHead(nn.Module):
    """
    DESCRIPTION:
    ------------
    Baseline linear classification head for multi-label ICD disease scoring.
    """
    def __init__(self, in_slots: int, in_dim: int, num_classes: int, dropout_p: float = 0.15):
        super().__init__()
        input_flat_dim = in_slots * in_dim
        self.feature_dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(input_flat_dim, num_classes)

    def forward(self, z_hat_slots: torch.Tensor) -> torch.Tensor:
        flat_z = z_hat_slots.contiguous().view(z_hat_slots.size(0), -1)
        return self.classifier(self.feature_dropout(flat_z))


# ==================================================================================================
# 8. LABEL-ATTENTIVE SLOT PROBE (LAAT + MODERN HOPFIELD)
# ==================================================================================================
class LabelAttentiveSlotProbe(nn.Module):
    r"""
    DESCRIPTION:
    ------------
    Advanced Label-Attentive Probe Head (LAAT-inspired) integrated with a Multi-Head 
    Continuous Modern Hopfield Associative Memory layer based on Ramsauer et al. (2020).
    """
    def __init__(
        self, 
        in_slots: int = 26, 
        in_dim: int = 512, 
        num_classes: int = 456, 
        num_prototypes: int = 64,
        num_hopfield_heads: int = 8,
        dropout_p: float = 0.20
    ):
        super().__init__()
        self.in_slots = in_slots
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.num_heads = num_hopfield_heads
        self.head_dim = in_dim // num_hopfield_heads

        self.prototype_memory = nn.Parameter(torch.empty(num_prototypes, in_dim))
        
        self.q_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.out_proj = nn.Linear(in_dim, in_dim, bias=False)

        self.hopfield_beta = nn.Parameter(torch.full((num_hopfield_heads, 1, 1), 1.0 / math.sqrt(self.head_dim)))
        self.hopfield_gate = nn.Parameter(torch.tensor(0.2))

        self.class_embeddings = nn.Parameter(torch.empty(num_classes, in_dim))
        self.query_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.weight_class = nn.Parameter(torch.empty(num_classes, in_dim))
        self.bias_class = nn.Parameter(torch.empty(num_classes))
        self.class_log_temp = nn.Parameter(torch.full((num_classes, 1, 1), math.log(1.0 / math.sqrt(in_dim))))
        
        self.slot_norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.prototype_memory)
        nn.init.xavier_uniform_(self.class_embeddings)
        nn.init.xavier_uniform_(self.weight_class)
        nn.init.zeros_(self.bias_class)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, z_hat_slots: torch.Tensor) -> torch.Tensor:
        B, L, D = z_hat_slots.shape
        z_norm = self.slot_norm(z_hat_slots)
        
        Q = self.q_proj(z_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(self.prototype_memory).view(self.num_prototypes, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(self.prototype_memory).view(self.num_prototypes, self.num_heads, self.head_dim).transpose(0, 1)

        Q_scaled = F.normalize(Q, p=2, dim=-1)
        K_scaled = F.normalize(K, p=2, dim=-1)

        beta_clamped = torch.clamp(self.hopfield_beta, min=0.1, max=10.0)
        energy_sim = torch.matmul(Q_scaled, K_scaled.transpose(-2, -1)) * beta_clamped
        attn_weights = F.softmax(energy_sim, dim=-1)
        
        retrieved_heads = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, L, D)
        hopfield_out = self.out_proj(retrieved_heads)
        z_refined = z_hat_slots + torch.sigmoid(self.hopfield_gate) * hopfield_out

        queries = self.query_proj(self.class_embeddings)
        Q_laat = queries.unsqueeze(0).expand(B, -1, -1)
        K_laat, V_laat = z_refined, z_refined

        temp = torch.exp(self.class_log_temp).clamp(min=0.01, max=5.0).view(1, -1, 1)
        Q_scaled = Q_laat * temp

        class_specific_contexts = F.scaled_dot_product_attention(
            Q_scaled, K_laat, V_laat, attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False, scale=1.0
        )

        return torch.sum(class_specific_contexts * self.weight_class.unsqueeze(0), dim=-1) + self.bias_class.unsqueeze(0)


# ==================================================================================================
# 9. AUXILIARY CARDINALITY HEAD
# ==================================================================================================
class AuxiliaryCardinalityHead(nn.Module):
    """
    DESCRIPTION:
    ------------
    Auxiliary regression head that predicts total active disease count (cardinality) for a patient.
    """
    def __init__(self, in_slots: int = 26, in_dim: int = 512, hidden_dim: int = 64):
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


# ==================================================================================================
# 10. CLINICAL DECODER UTILITY
# ==================================================================================================
class ClinicalDecoder:
    """
    DESCRIPTION:
    ------------
    Utility class for decoding model prediction tensors back into human-readable ICD string codes.
    """
    def __init__(self, codebook_json_path: str = "clinical_codebooks.json"):
        with open(codebook_json_path, "r", encoding="utf-8") as f:
            self.codebooks = json.load(f)
            
        self.num_total_features = self.codebooks["metadata"]["num_total_features"]
        self.num_cat_results = self.codebooks["metadata"]["num_cat_results"]
        self.num_icd_classes = self.codebooks["metadata"]["num_icd_classes"]
        self.id_to_icd = self.codebooks["inverse_icd_codes"]
        self.id_to_feature = self.codebooks["inverse_maps"]

    def decode_predictions(self, probabilities_tensor: torch.Tensor, threshold: float = 0.50) -> List[str]:
        active_indices = torch.nonzero(probabilities_tensor > threshold).flatten().tolist()
        return [self.id_to_icd.get(str(idx), "UNKNOWN_CODE") for idx in active_indices]


# ==================================================================================================
# 11. CLASS-AWARE ASYMMETRIC LOSS
# ==================================================================================================
class ClassAwareASL(nn.Module):
    def __init__(
        self, 
        class_frequencies: Union[torch.Tensor, np.ndarray, list], 
        gamma_pos: float = 0.0, 
        gamma_neg_base: float = 4.5, 
        beta_neg_base: float = 2.5,
        delta_beta: float = 1.0
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        
        if not isinstance(class_frequencies, torch.Tensor):
            class_frequencies = torch.tensor(class_frequencies, dtype=torch.float32)
        else:
            class_frequencies = class_frequencies.float()
            
        self.register_buffer('gamma_neg', gamma_neg_base + (1.0 - class_frequencies) * 2.0)
        self.register_buffer('beta_neg', beta_neg_base + delta_beta * (1.0 - class_frequencies))
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).float()
        targets = targets.float()
        
        loss_pos = targets * torch.log(probs + 1e-7) * (1.0 - probs).pow(self.gamma_pos)
        loss_neg = (1.0 - targets) * torch.log(1.0 - probs + 1e-7) * probs.pow(self.gamma_neg)
        
        batch_loss = - (loss_pos + (self.beta_neg * loss_neg))
        return batch_loss.sum(dim=-1).mean()


# ==================================================================================================
# 12. SPECTRAL MANIFOLD DIAGNOSTICS ENGINE
# ==================================================================================================
@torch.no_grad()
def compute_comprehensive_manifold_diagnostics(z: torch.Tensor) -> Dict[str, float]:
    r"""
    DESCRIPTION:
    ------------
    Evaluates representation health across latent space representations to detect collapse or cross-talk.
    """
    z = z.float()
    if not torch.isfinite(z).all():
        z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
        
    if z.dim() == 3:
        B, K, D = z.shape
        var_per_dim = z.var(dim=0)
        mean_batch_std = torch.sqrt(var_per_dim + 1e-6).mean().item()
        
        z_norm = F.normalize(z, p=2, dim=-1)
        slot_sim = torch.bmm(z_norm, z_norm.transpose(1, 2))
        triu_indices = torch.triu_indices(K, K, offset=1, device=z.device)
        mean_slot_cross_talk = slot_sim[:, triu_indices[0], triu_indices[1]].mean().item()
        
        matrix_for_svd = z.contiguous().view(B * K, D)
        z_for_sparsity = z.mean(dim=1)
    else:
        mean_batch_std = z.std(dim=0).mean().item()
        mean_slot_cross_talk = 0.0
        matrix_for_svd = z
        z_for_sparsity = z
        
    sparsity_index = torch.mean(
        torch.norm(z_for_sparsity, p=1, dim=1) / (torch.norm(z_for_sparsity, p=2, dim=1) + 1e-8)
    ).item()

    if matrix_for_svd.size(0) <= 1:
        return {
            "batch_std": mean_batch_std, 
            "slot_cross_talk": mean_slot_cross_talk, 
            "effective_rank": float(matrix_for_svd.size(-1)), 
            "sparsity_index": sparsity_index
        }
        
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


# ==================================================================================================
# 13. DYNAMIC CLINICAL AUDIT ENGINE
# ==================================================================================================
def execute_clinical_audit(
    targets: np.ndarray, 
    probabilities: np.ndarray, 
    predicted_cardinalities: Optional[np.ndarray] = None, 
    thresholds: Optional[np.ndarray] = None, 
    min_positive_prevalence: int = 2, 
    calibrate_per_class: bool = True, 
    temp_alpha: float = 0.15,
    silent: bool = False
) -> Dict[str, Any]:
    num_samples, num_classes = targets.shape

    if temp_alpha > 0.0:
        f_c = targets.sum(axis=0)
        max_f = np.max(f_c)
        T_c = 1.0 + temp_alpha * np.log((max_f + 1.0) / (f_c + 1.0))
        
        clipped_probs = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        raw_logits = np.log(clipped_probs / (1.0 - clipped_probs))
        scaled_logits = raw_logits / T_c
        probabilities = 1.0 / (1.0 + np.exp(-scaled_logits))

    auc_roc_list, auc_pr_list, active_class_indices = [], [], []
    for c_idx in range(num_classes):
        pos_count = targets[:, c_idx].sum()
        if pos_count >= min_positive_prevalence and pos_count < num_samples:
            active_class_indices.append(c_idx)
            auc_roc_list.append(roc_auc_score(targets[:, c_idx], probabilities[:, c_idx]))
            prec, rec, _ = precision_recall_curve(targets[:, c_idx], probabilities[:, c_idx])
            auc_pr_list.append(auc(rec, prec))
            
    macro_auc_roc = np.mean(auc_roc_list) * 100 if auc_roc_list else 0.0
    macro_auc_pr = np.mean(auc_pr_list) * 100 if auc_pr_list else 0.0
    
    micro_auc_roc = roc_auc_score(
        targets[:, active_class_indices], probabilities[:, active_class_indices], average='micro'
    ) * 100 if active_class_indices else 0.0

    if silent and not calibrate_per_class and predicted_cardinalities is None:
        return {"macro_auc_roc": macro_auc_roc, "macro_auc_pr": macro_auc_pr, "micro_auc_roc": micro_auc_roc}

    if thresholds is None:
        thresholds = np.ones(num_classes) * 0.15
        if calibrate_per_class:
            for c_idx in active_class_indices:
                best_score, best_thresh = -1.0, 0.50
                y_true = targets[:, c_idx]
                for thresh in np.linspace(0.0001, 0.99, 200):
                    class_preds = (probabilities[:, c_idx] > thresh).astype(float)
                    tp = np.sum((y_true == 1) & (class_preds == 1))
                    fp = np.sum((y_true == 0) & (class_preds == 1))
                    tn = np.sum((y_true == 0) & (class_preds == 0))
                    fn = np.sum((y_true == 1) & (class_preds == 0))
                    
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    
                    if sens >= 0.85 and prec >= 0.85 and spec >= 0.95:
                        score = 100.0 + (sens + prec + spec) - np.var([sens, prec, spec])
                    elif spec >= 0.990 and sens >= 0.70:
                        score = 50.0 + (sens * 2.0) + prec
                    else:
                        score = max(0.0, sens + spec - 1.0) * (prec + 1e-5)
                        
                    if score > best_score:
                        best_score, best_thresh = score, thresh
                thresholds[c_idx] = best_thresh
        
    preds = np.zeros_like(probabilities)
    sensitivity_list, specificity_list = [], []
    for c_idx in range(num_classes):
        preds[:, c_idx] = (probabilities[:, c_idx] > thresholds[c_idx]).astype(float)
        y_true, y_pred = targets[:, c_idx], preds[:, c_idx]
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if c_idx in active_class_indices:
            sensitivity_list.append(sens)
            specificity_list.append(spec)

    macro_p, _, macro_f1, _ = precision_recall_fscore_support(
        targets[:, active_class_indices], preds[:, active_class_indices], average='macro', zero_division=0
    ) if active_class_indices else (0.0, 0.0, 0.0, 0.0)
    
    macro_sens = np.mean(sensitivity_list) * 100 if sensitivity_list else 0.0
    macro_spec = np.mean(specificity_list) * 100 if specificity_list else 0.0

    hit1, hit3, hit5, hit8 = 0, 0, 0, 0
    p1_scores, p3_scores, p5_scores, p8_scores = [], [], [], []
    adaptive_hit_count = 0
    adaptive_precision_scores = []
    
    for i in range(num_samples):
        top_indices = np.argsort(probabilities[i])[::-1]
        if targets[i, top_indices[0]] == 1.0: hit1 += 1
        if np.any(targets[i, top_indices[:3]] == 1.0): hit3 += 1
        if np.any(targets[i, top_indices[:5]] == 1.0): hit5 += 1
        if np.any(targets[i, top_indices[:8]] == 1.0): hit8 += 1
            
        p1_scores.append(np.sum(targets[i, top_indices[:1]]) / 1.0)
        p3_scores.append(np.sum(targets[i, top_indices[:3]]) / 3.0)
        p5_scores.append(np.sum(targets[i, top_indices[:5]]) / 5.0)
        p8_scores.append(np.sum(targets[i, top_indices[:8]]) / 8.0)
        
        k_adaptive = max(1, int(np.round(predicted_cardinalities[i]))) if predicted_cardinalities is not None else max(1, int(np.sum(probabilities[i] > thresholds)))
        adaptive_slice = top_indices[:k_adaptive]
        if np.any(targets[i, adaptive_slice] == 1.0): adaptive_hit_count += 1
        adaptive_precision_scores.append(np.sum(targets[i, adaptive_slice]) / float(k_adaptive))
            
    metrics_summary = {
        "macro_auc_roc": macro_auc_roc,
        "micro_auc_roc": micro_auc_roc,
        "macro_auc_pr": macro_auc_pr,
        "macro_f1": macro_f1 * 100,
        "macro_precision": macro_p * 100,
        "macro_sensitivity": macro_sens,
        "macro_specificity": macro_spec,
        "top1_rate": (hit1 / num_samples) * 100,
        "top3_rate": (hit3 / num_samples) * 100,
        "top5_rate": (hit5 / num_samples) * 100,
        "top8_rate": (hit8 / num_samples) * 100,
        "precision_at_1": np.mean(p1_scores) * 100,
        "precision_at_3": np.mean(p3_scores) * 100,
        "precision_at_5": np.mean(p5_scores) * 100,
        "precision_at_8": np.mean(p8_scores) * 100,
        "adaptive_hit_rate": (adaptive_hit_count / num_samples) * 100,
        "adaptive_precision": np.mean(adaptive_precision_scores) * 100,
        "calibrated_thresholds": thresholds
    }

    if not silent:
        print("\n" + "═"*75)
        print(" 🏥 COMPREHENSIVE CLINICAL MANIFOLD AUDIT REPORT")
        print("═"*75)
        if temp_alpha > 0.0:
            print(f" 🧪 [CALIBRATION] Temperature Scaling Active (alpha = {temp_alpha})")
            print("-" * 75)
        print(f" 🩺 [TIER 1: RANKING]   Macro AUC-ROC:          {macro_auc_roc:3.2f}%")
        print(f" 🩺 [TIER 1: RANKING]   Micro AUC-ROC:          {micro_auc_roc:3.2f}%")
        print(f" 🩺 [TIER 1: RANKING]   Macro AUC-PR (Sparsity):{macro_auc_pr:3.2f}%")
        print("-" * 75)
        print(f" 🛡️ [TIER 2: BOUNDARY]  Calibrated Macro F1:    {metrics_summary['macro_f1']:3.2f}%")
        print(f" 🛡️ [TIER 2: BOUNDARY]  Macro Precision:        {metrics_summary['macro_precision']:3.2f}%")
        print(f" 🛡️ [TIER 2: BOUNDARY]  Macro Sensitivity (TPR):{macro_sens:3.2f}%")
        print(f" 🛡️ [TIER 2: BOUNDARY]  Macro Specificity (TNR):{macro_spec:3.2f}%")
        print("-" * 75)
        print(f" 🚀 [ADAPTIVE HORIZON]  Hit Rate (Safety Net):  {metrics_summary['adaptive_hit_rate']:3.2f}%")
        print(f" 🚀 [ADAPTIVE HORIZON]  Precision (Density):    {metrics_summary['adaptive_precision']:3.2f}%")
        print("═"*75 + "\n")
        print(f" ⚡ [TIER 3: FIXED K]   Top-1 Primary Hit Rate: {metrics_summary['top1_rate']:3.2f}% │ Precision@1: {metrics_summary['precision_at_1']:3.2f}%")
        print(f" ⚡ [TIER 3: FIXED K]   Top-3 Differential Rate:{metrics_summary['top3_rate']:3.2f}% │ Precision@3: {metrics_summary['precision_at_3']:3.2f}%")
        print(f" ⚡ [TIER 3: FIXED K]   Top-5 Differential Rate:{metrics_summary['top5_rate']:3.2f}% │ Precision@5: {metrics_summary['precision_at_5']:3.2f}%")
        print(f" ⚡ [TIER 3: FIXED K]   Top-8 Differential Rate:{metrics_summary['top8_rate']:3.2f}% │ Precision@8: {metrics_summary['precision_at_8']:3.2f}%")
        print("═"*75 + "\n")
        
    return metrics_summary
