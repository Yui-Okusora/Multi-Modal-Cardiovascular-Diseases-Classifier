# src/ModelModules.py
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
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


class LocalFactorizedTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 1500, d_model: int = 512, bottleneck_dim: int = 64):
        super().__init__()
        self.subword_embed = nn.Embedding(vocab_size, bottleneck_dim, padding_idx=0)
        self.projection = nn.Linear(bottleneck_dim, d_model)
        self.attn_query = nn.Linear(d_model, 1, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def _pool_impl(self, embs: torch.Tensor, flat_ids: torch.Tensor) -> torch.Tensor:
        attn_logits = self.attn_query(embs)
        
        pad_mask = (flat_ids == 0).unsqueeze(-1)
        all_padded = pad_mask.all(dim=1, keepdim=True)
        pad_mask = pad_mask.masked_fill(all_padded, False)
        
        attn_logits = attn_logits.masked_fill(pad_mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=1)
        
        pooled_text = torch.sum(embs * attn_weights, dim=1)
        return self.layer_norm(pooled_text)

    def forward(self, cat_result_ids: torch.Tensor) -> torch.Tensor:
        B, L_seq, L_text = cat_result_ids.shape
        flat_ids = cat_result_ids.view(B * L_seq, L_text)
        
        embs = self.projection(self.subword_embed(flat_ids))  # [B*L_seq, 16, 512]
        
        if self.training and embs.requires_grad:
            pooled = checkpoint(self._pool_impl, embs, flat_ids, use_reentrant=False)
        else:
            pooled = self._pool_impl(embs, flat_ids)
            
        return pooled.view(B, L_seq, -1)

# ==================================================================================================
# 2. UNIFIED SYSTEMIC TOKENIZER
# ==================================================================================================
class UnifiedSystemicEmbedder(nn.Module):
    def __init__(self, num_total_features: int, bpe_vocab_size: int = 1500, d_model: int = 512, num_frequencies: int = 32):
        super().__init__()
        self.feature_embedding = nn.Embedding(num_total_features, d_model)
        self.numeric_encoder = PeriodicNumericalEmbedding(d_model=d_model, num_frequencies=num_frequencies)
        self.film_gamma = nn.Linear(d_model, d_model)
        
        self.text_encoder = LocalFactorizedTextEncoder(vocab_size=bpe_vocab_size, d_model=d_model, bottleneck_dim=64)
        
        self.time_embedder = ContinuousTimeEmbedding(d_model)
        self.global_token_norm = nn.LayerNorm(d_model)

    def forward(
        self, 
        feature_ids: torch.Tensor,       # [B, L_seq]
        numeric_values: torch.Tensor,    # [B, L_seq]
        cat_result_ids: torch.Tensor,    # [B, L_seq, 16] (3D Tensor)
        timestamps: torch.Tensor         # [B, L_seq]
    ) -> torch.Tensor:
        feat_emb = self.feature_embedding(feature_ids)
        time_emb = self.time_embedder(timestamps)
        
        # 1. Process continuous numeric features with FiLM
        num_emb = self.numeric_encoder(numeric_values)
        gamma = torch.sigmoid(self.film_gamma(feat_emb))
        modulated_num_emb = gamma * num_emb
        
        # Mask numeric embedding if event has text (checks if first subword != 0)
        has_numeric_mask = (cat_result_ids[:, :, 0] == 0).unsqueeze(-1).float()
        modulated_num_emb = modulated_num_emb * has_numeric_mask
        
        # 2. Local Factorized BPE Attention Pooling -> [B, L_seq, 512]
        cat_emb = self.text_encoder(cat_result_ids)
        
        # 3. Positionally Locked Addition across all 4 streams
        combined_tokens = feat_emb + modulated_num_emb + cat_emb + time_emb
        return self.global_token_norm(combined_tokens)


class PerceiverLatentPooling(nn.Module):
    def __init__(self, num_slots: int = 24, d_model: int = 512, nheads: int = 8, self_attn_layers: int = 2):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        
        # Single parameter matrix for slots
        self.learned_slots = nn.Parameter(torch.empty(num_slots, d_model))
        nn.init.orthogonal_(self.learned_slots, gain=1.0)
        
        self.slot_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, batch_first=True)
        
        # Multi-layer self-attention stack with explicit final LayerNorm
        latent_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nheads, dim_feedforward=d_model * 2,
            dropout=0.05, activation='gelu', batch_first=True, norm_first=True
        )
        # 🚀 Fix 4: Pass final_norm to TransformerEncoder so output is normalized
        final_norm = nn.LayerNorm(d_model)
        self.slot_mixer = nn.TransformerEncoder(latent_layer, num_layers=self_attn_layers, norm=final_norm)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = x.size(0)
        
        norm_slots = self.slot_norm(self.learned_slots).unsqueeze(0).expand(batch_size, -1, -1)
        norm_x = self.kv_norm(x)
        
        # Robust padding mask handling
        if padding_mask is not None:
            all_padded_rows = padding_mask.all(dim=-1, keepdim=True)
            padding_mask = padding_mask.masked_fill(all_padded_rows, False)

        # Cross-Attention: Compress sequence length L -> 24 slots
        attn_out, _ = self.cross_attn(
            query=norm_slots, key=norm_x, value=norm_x, key_padding_mask=padding_mask
        )
        slots = norm_slots + attn_out
        
        # Deep Latent Slot Mixing
        return self.slot_mixer(slots)


# ==================================================================================================
# 4. OPTIMIZED CONTEXT ENCODER (STUDENT STREAM)
# ==================================================================================================
class ContextEncoder(nn.Module):
    """
    Streamlined Context Encoder with $L_2$ Hypersphere Output Projection.
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
        
        self.tokenizer = UnifiedSystemicEmbedder(num_total_features, num_cat_results, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=8, 
            dim_feedforward=d_model * 4, 
            batch_first=True, 
            activation='gelu', 
            norm_first=True
        )
        self.temporal_backbone = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.perceiver_pool = PerceiverLatentPooling(num_slots=num_slots, d_model=d_model, self_attn_layers=2)

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
        
        # L2 Hypersphere Projection (Scale = sqrt(d_model))
        return F.normalize(z_c, p=2, dim=-1) * math.sqrt(self.d_model)


TargetEncoder = ContextEncoder


# ==================================================================================================
# 5. WORLD-MODEL PREDICTOR
# ==================================================================================================
class Predictor(nn.Module):
    """
    Cross-Attention World-Model Predictor:
    1. Internal learned target queries (Q) attend to past latents (K, V).
    2. Lightweight inter-slot self-attention pass allows predicted slots to unmix.
    3. L2 Hypersphere projection at exit matches TargetEncoder metric space.
    """
    def __init__(self, num_slots: int = 24, d_model: int = 512, nhead: int = 8):
        super().__init__()
        self.num_future_slots = num_slots
        self.d_model = d_model
        
        # Target queries owned internally as learned parameters
        self.target_queries = nn.Parameter(torch.empty(num_slots, d_model))
        nn.init.orthogonal_(self.target_queries, gain=1.0)
        
        # Pre-LN Cross Attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # 🚀 Fix 2: Lightweight Inter-Slot Self-Attention block so predicted slots can coordinate
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.slot_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # Feed-Forward Block
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, z_past: torch.Tensor, future_time_emb: torch.Tensor = None) -> torch.Tensor:
        batch_size = z_past.size(0)
        
        # Expand internal target queries across the batch: [B, 24, 512]
        q = self.target_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 🚀 Fix 3: Robust shape broadcast guard for 2D vs 3D future_time_emb
        if future_time_emb is not None:
            if future_time_emb.dim() == 2:
                future_time_emb = future_time_emb.unsqueeze(1)
            q = q + future_time_emb
            
        norm_q = self.norm_q(q)
        norm_kv = self.norm_kv(z_past)
        
        # 1. Cross-Attend: Target Queries (Q) look up Past Context (K, V)
        cross_out, _ = self.cross_attn(query=norm_q, key=norm_kv, value=norm_kv)
        h_cross = q + cross_out
        
        # 2. Inter-Slot Self-Attention: Allow predicted future slots to unmix
        norm_h = self.self_attn_norm(h_cross)
        self_out, _ = self.slot_self_attn(query=norm_h, key=norm_h, value=norm_h)
        h_self = h_cross + self_out
        
        # 3. FFN Refinement
        z_hat_raw = h_self + self.ffn(h_self)
        
        # 🚀 Fix 1: Matching L2 Hypersphere Projection (Scale = sqrt(d_model))
        return F.normalize(z_hat_raw, p=2, dim=-1) * math.sqrt(self.d_model)


# ==================================================================================================
# 6. PATIENT MANIFOLD ASSEMBLER
# ==================================================================================================
class PatientManifoldAssembler(nn.Module):
    """
    Optimized Patient Manifold Assembler:
    1. Direct L2 projection without redundant LayerNorm pre-scaling.
    2. Compact embedding table for gender (4 slots instead of BPE vocab size).
    3. Robust shape formatting for age and gender inputs.
    """
    def __init__(self, num_cat_results: int, num_gender_cats: int = 4, latent_dim: int = 512, covariate_scale: float = 0.50):
        super().__init__()
        self.latent_dim = latent_dim

        covariate_scale = max(0.01, min(0.99, covariate_scale))
        raw_init = math.log(covariate_scale / (1.0 - covariate_scale))
        self.raw_covariate_scale = nn.Parameter(torch.tensor(raw_init))
        
        self.age_projector = nn.Linear(in_features=1, out_features=latent_dim)
        # 🚀 Fix: Compact embedding table for categorical demographic features
        self.gender_embed = nn.Embedding(num_embeddings=num_gender_cats, embedding_dim=latent_dim)

    @property
    def covariate_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_covariate_scale)

    def forward(self, z_c_raw: torch.Tensor, age: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
        age_2d = age.view(-1, 1).float()
        gender_1d = gender.view(-1).long()

        z_age = self.age_projector(age_2d).unsqueeze(1)
        z_gender = self.gender_embed(gender_1d).unsqueeze(1)
        
        scale = math.sqrt(self.latent_dim)
        c_scale = self.covariate_scale
        
        # 🚀 Fix: Clean direct L2 normalization without redundant LayerNorm
        z_age = F.normalize(z_age, p=2, dim=-1) * (scale * c_scale)
        z_gender = F.normalize(z_gender, p=2, dim=-1) * (scale * c_scale)
        
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
# 8A. CONTINUOUS HOPFIELD ARCHETYPE INJECTOR
# ==================================================================================================
class ContinuousHopfieldMemory(nn.Module):
    r"""
    Standalone Associative Memory layer. Retrieves textbook clinical archetypes 
    and overlays them onto the patient's latent slots for noise-cancellation.
    """
    def __init__(self, in_dim: int = 512, num_prototypes: int = 128, num_heads: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.num_prototypes = num_prototypes
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        self.prototype_memory = nn.Parameter(torch.empty(num_prototypes, in_dim))
        
        self.q_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.out_proj = nn.Linear(in_dim, in_dim, bias=False)

        self.hopfield_beta = nn.Parameter(torch.full((num_heads, 1, 1), math.sqrt(self.head_dim)))
        self.hopfield_gate = nn.Parameter(torch.tensor(1.0))
        self.slot_norm = nn.LayerNorm(in_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.prototype_memory)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)  # 🚀 Fix preserved: Opens gradient gates instantly

    def forward(self, z_slots: torch.Tensor) -> torch.Tensor:
        B, L, D = z_slots.shape
        z_norm = self.slot_norm(z_slots)
        
        Q = self.q_proj(z_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(self.prototype_memory).view(self.num_prototypes, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(self.prototype_memory).view(self.num_prototypes, self.num_heads, self.head_dim).transpose(0, 1)

        Q_scaled = F.normalize(Q, p=2, dim=-1)
        K_scaled = F.normalize(K, p=2, dim=-1)

        beta_clamped = torch.clamp(self.hopfield_beta, min=1.0, max=32.0)
        energy_sim = torch.matmul(Q_scaled, K_scaled.transpose(-2, -1)) * beta_clamped
        attn_weights = F.softmax(energy_sim, dim=-1)
        
        retrieved_heads = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, L, D)
        hopfield_out = self.out_proj(retrieved_heads)
        
        return z_slots + torch.sigmoid(self.hopfield_gate) * hopfield_out


# ==================================================================================================
# 8B. LABEL-ATTENTIVE PROBE HEAD (LAAT)
# ==================================================================================================
class LabelAttentiveProbe(nn.Module):
    r"""
    A switchable decisive probe that uses cross-attention to route specific 
    physiological slots to specific disease probabilities.
    """
    def __init__(self, in_dim: int = 512, num_classes: int = 456, dropout_p: float = 0.20):
        super().__init__()
        self.class_queries = nn.Parameter(torch.empty(num_classes, in_dim))
        self.weight_class = nn.Parameter(torch.empty(num_classes, in_dim))
        self.bias_class = nn.Parameter(torch.empty(num_classes))
        
        # 🚀 Fix preserved: Temperature log initialized to 0.0 for sharp routing
        self.class_log_temp = nn.Parameter(torch.zeros(num_classes, 1, 1))
        
        self.dropout = nn.Dropout(p=dropout_p)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.class_queries)
        nn.init.xavier_uniform_(self.weight_class)
        nn.init.zeros_(self.bias_class)

    def forward(self, z_refined: torch.Tensor) -> torch.Tensor:
        B, L, D = z_refined.shape
        Q_laat = self.class_queries.unsqueeze(0).expand(B, -1, -1)
        K_laat, V_laat = z_refined, z_refined

        # 🚀 Fix preserved: Clamp up to 10.0 allows dynamic sharpening
        temp = torch.exp(self.class_log_temp).clamp(min=0.1, max=10.0).view(1, -1, 1)
        Q_scaled_laat = Q_laat * temp

        class_specific_contexts = F.scaled_dot_product_attention(
            Q_scaled_laat, K_laat, V_laat, attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False, scale=1.0 / math.sqrt(D)
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
        self.gate_pool = nn.Linear(hidden_dim, 1)
        self.output_projector = nn.Linear(hidden_dim, 1)

    def forward(self, z_hat_slots: torch.Tensor) -> torch.Tensor:
        h_slots = self.slot_net(z_hat_slots)
        
        # 🚀 Fix: Additive Sigmoid Gating allows disease count aggregation across active slots
        gate_weights = torch.sigmoid(self.gate_pool(h_slots))  # [B, 26, 1] in (0, 1)
        pooled_context = torch.sum(h_slots * gate_weights, dim=1)  # Additive sum
        
        raw_score = self.output_projector(pooled_context).squeeze(-1)
        
        # Smooth physical floor: Guarantees output K >= 1.0
        return 1.0 + F.softplus(raw_score)


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
        gamma_neg: float = 4.0, 
        gamma_pos: float = 0.0, 
        clip: float = 0.05, 
        eps: float = 1e-8
    ):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Compute Probabilities safely
        p_pos = torch.sigmoid(logits)
        p_neg = 1.0 - p_pos

        # 2. Asymmetric Probability Shifting (Margin m = clip)
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)

        # 3. Compute Focal Weights IN the autograd graph (Crucial for ASL)
        pt_pos = p_pos * targets
        pt_neg = p_neg * (1.0 - targets)
        
        # 4. Cross Entropy
        loss_pos = -targets * torch.log(p_pos.clamp(min=self.eps))
        loss_neg = -(1.0 - targets) * torch.log(p_neg.clamp(min=self.eps))

        # 5. Apply Gamma Modulation dynamically
        if self.gamma_pos > 0:
            loss_pos = loss_pos * torch.pow(1.0 - pt_pos, self.gamma_pos)
        if self.gamma_neg > 0:
            loss_neg = loss_neg * torch.pow(1.0 - pt_neg, self.gamma_neg)

        return (loss_pos + loss_neg).sum(dim=-1).mean()

class KendallMultiTaskLoss(nn.Module):
    """
    🎯 KENDALL MULTI-TASK UNCERTAINTY LOSS WRAPPER
    Seamlessly integrates homoscedastic uncertainty weighting into BaseExecutionEngine
    by receiving and returning a dictionary of [weight, raw_loss] pairs.
    """
    def __init__(self, task_keys: list = None, init_log_var: float = 0.0):
        super().__init__()
        self.log_vars = nn.ParameterDict()
        if task_keys is not None:
            for key in task_keys:
                self.log_vars[key] = nn.Parameter(torch.tensor(init_log_var, dtype=torch.float32))

    def forward(self, loss_dict: dict) -> dict:
        out_dict = {}
        barrier_penalty = 0.0

        for key, val in loss_dict.items():
            base_weight, raw_loss = val if isinstance(val, (list, tuple)) else (1.0, val)
            
            # Dynamic fallback registration if key was not pre-registered in __init__
            if key not in self.log_vars:
                dev = raw_loss.device if torch.is_tensor(raw_loss) else torch.device("cpu")
                self.log_vars[key] = nn.Parameter(torch.tensor(0.0, device=dev, dtype=torch.float32))

            log_var = self.log_vars[key]
            
            # Kendall Precision Weight: 0.5 * exp(-log_var)
            precision = 0.5 * torch.exp(-log_var)
            effective_weight = precision * (base_weight if base_weight is not None else 1.0)
            
            out_dict[key] = [effective_weight, raw_loss]
            
            # Accumulate homoscedastic barrier penalty: 0.5 * log_var
            barrier_penalty = barrier_penalty + (0.5 * log_var)

        # Added as a component so BaseExecutionEngine automatically sums it into total_loss
        out_dict["loss_kendall_barrier"] = [1.0, barrier_penalty]
        return out_dict

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
def print_clinical_audit_report(
    metrics_summary: Dict[str, Any], 
    temp_alpha: float = 0.15, 
    calibration_beta: float = 1.0
) -> None:
    """
    Renders a formatted, comprehensive clinical audit scorecard strictly using values from metrics_summary.
    """
    print("\n" + "═"*75)
    print(" 🏥 COMPREHENSIVE CLINICAL MANIFOLD AUDIT REPORT")
    print("═"*75)
    
    if temp_alpha > 0.0:
        print(f" 🧪 [CALIBRATION] Temperature Scaling Active (alpha = {temp_alpha}) | Calibration Beta = {calibration_beta}")
        print("-" * 75)
        
    print(f" 🩺 [TIER 1: RANKING]   Macro AUC-ROC:          {metrics_summary.get('macro_auc_roc', 0.0):3.2f}%")
    print(f" 🩺 [TIER 1: RANKING]   Micro AUC-ROC:          {metrics_summary.get('micro_auc_roc', 0.0):3.2f}%")
    print(f" 🩺 [TIER 1: RANKING]   Macro AUC-PR (Sparsity):{metrics_summary.get('macro_auc_pr', 0.0):3.2f}%")
    print("-" * 75)
    
    print(f" 🛡️ [TIER 2: BOUNDARY - GLOBAL MICRO METRICS]")
    print(f"    • Micro F1 Score:           {metrics_summary.get('micro_f1', 0.0):3.2f}%")
    print(f"    • Micro Precision:          {metrics_summary.get('micro_precision', 0.0):3.2f}%")
    print(f"    • Micro Sensitivity (Recall):{metrics_summary.get('micro_sensitivity', 0.0):3.2f}%")
    print("-" * 75)
    
    print(f" 🛡️ [TIER 2: BOUNDARY - UNWEIGHTED MACRO]")
    print(f"    • Unweighted Macro F1:      {metrics_summary.get('macro_f1', 0.0):3.2f}%")
    print(f"    • Unweighted Macro Precision: {metrics_summary.get('macro_precision', 0.0):3.2f}%")
    print(f"    • Unweighted Sensitivity:   {metrics_summary.get('macro_sensitivity', 0.0):3.2f}%")
    print(f"    • Unweighted Specificity:   {metrics_summary.get('macro_specificity', 0.0):3.2f}%")
    print("-" * 75)
    
    print(f" 📊 [TIER 2: PREVALENCE-WEIGHTED MACRO]")
    print(f"    • Weighted Macro F1:        {metrics_summary.get('weighted_macro_f1', 0.0):3.2f}%")
    print(f"    • Weighted Macro Precision: {metrics_summary.get('weighted_macro_precision', 0.0):3.2f}%")
    print(f"    • Weighted Sensitivity:     {metrics_summary.get('weighted_macro_sensitivity', 0.0):3.2f}%")
    print(f"    • Weighted Specificity:     {metrics_summary.get('weighted_macro_specificity', 0.0):3.2f}%")
    print("-" * 75)
    
    print(f" 🩺 [TIER 2: TOP-50 FREQUENT DISEASES BENCHMARK]")
    print(f"    • Top-50 Macro F1:          {metrics_summary.get('top50_macro_f1', 0.0):3.2f}%")
    print(f"    • Top-50 Precision:         {metrics_summary.get('top50_macro_precision', 0.0):3.2f}%")
    print(f"    • Top-50 Sensitivity:       {metrics_summary.get('top50_macro_sensitivity', 0.0):3.2f}%")
    print(f"    • Top-50 Specificity:       {metrics_summary.get('top50_macro_specificity', 0.0):3.2f}%")
    print("-" * 75)
    
    print(f" 🚀 [ADAPTIVE HORIZON]  Hit Rate (Safety Net):  {metrics_summary.get('adaptive_hit_rate', 0.0):3.2f}%")
    print(f" 🚀 [ADAPTIVE HORIZON]  Precision (Density):    {metrics_summary.get('adaptive_precision', 0.0):3.2f}%")
    print("═"*75 + "\n")
    
    print(f" ⚡ [TIER 3: FIXED K]   Top-1 Primary Hit Rate: {metrics_summary.get('top1_rate', 0.0):3.2f}% │ Precision@1: {metrics_summary.get('precision_at_1', 0.0):3.2f}%")
    print(f" ⚡ [TIER 3: FIXED K]   Top-3 Differential Rate:{metrics_summary.get('top3_rate', 0.0):3.2f}% │ Precision@3: {metrics_summary.get('precision_at_3', 0.0):3.2f}%")
    print(f" ⚡ [TIER 3: FIXED K]   Top-5 Differential Rate:{metrics_summary.get('top5_rate', 0.0):3.2f}% │ Precision@5: {metrics_summary.get('precision_at_5', 0.0):3.2f}%")
    print(f" ⚡ [TIER 3: FIXED K]   Top-8 Differential Rate:{metrics_summary.get('top8_rate', 0.0):3.2f}% │ Precision@8: {metrics_summary.get('precision_at_8', 0.0):3.2f}%")
    print("═"*75 + "\n")

def execute_clinical_audit(
    targets: np.ndarray, 
    probabilities: np.ndarray, 
    predicted_cardinalities: Optional[np.ndarray] = None, 
    thresholds: Optional[np.ndarray] = None, 
    min_positive_prevalence: int = 2, 
    calibrate_per_class: bool = True, 
    calibration_beta: float = 1.0,  # 🚀 Added: beta > 1.0 (e.g., 2.0) prioritizes Recall > 60%
    temp_alpha: float = 0.15,
    silent: bool = False
) -> Dict[str, Any]:
    num_samples, num_classes = targets.shape

    # 1. Temperature Scaling Calibration (Frequency-Aware)
    if temp_alpha > 0.0:
        f_c = targets.sum(axis=0)
        max_f = np.max(f_c)
        T_c = 1.0 + temp_alpha * np.log((max_f + 1.0) / (f_c + 1.0))
        
        clipped_probs = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        raw_logits = np.log(clipped_probs / (1.0 - clipped_probs))
        scaled_logits = raw_logits / T_c
        probabilities = 1.0 / (1.0 + np.exp(-scaled_logits))

    # 2. Ranking Metrics Calculation (ROC-AUC & PR-AUC)
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

    # 3. Direct Per-Class F_beta Threshold Calibration
    if thresholds is None:
        thresholds = np.ones(num_classes) * 0.15
        if calibrate_per_class:
            for c_idx in active_class_indices:
                y_true = targets[:, c_idx]
                y_prob = probabilities[:, c_idx]
                
                if np.sum(y_true) == 0:
                    thresholds[c_idx] = 0.50
                    continue

                precisions, recalls, thresh_grid = precision_recall_curve(y_true, y_prob)
                
                # 🚀 F_beta calculation (calibration_beta = 2.0 optimizes for Recall > 60%)
                beta_sq = calibration_beta ** 2
                fbeta_scores = (1 + beta_sq) * (precisions * recalls) / ((beta_sq * precisions) + recalls + 1e-8)
                
                best_idx = np.argmax(fbeta_scores)
                if best_idx < len(thresh_grid):
                    thresholds[c_idx] = thresh_grid[best_idx]
                else:
                    thresholds[c_idx] = 0.50
        
    # 4. Binary Decision Thresholding & Granular Class Audit
    preds = np.zeros_like(probabilities)
    sensitivity_list, specificity_list = [], []
    per_class_p, per_class_r, per_class_f1, per_class_counts = [], [], [], []

    for c_idx in range(num_classes):
        preds[:, c_idx] = (probabilities[:, c_idx] > thresholds[c_idx]).astype(float)
        y_true, y_pred = targets[:, c_idx], preds[:, c_idx]
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_c = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

        if c_idx in active_class_indices:
            sensitivity_list.append(sens)
            specificity_list.append(spec)
            per_class_p.append(prec)
            per_class_r.append(sens)
            per_class_f1.append(f1_c)
            per_class_counts.append(targets[:, c_idx].sum())

    # Convert to arrays for vectorized macro weighting
    p_arr = np.array(per_class_p)
    r_arr = np.array(per_class_r)
    spec_arr = np.array(specificity_list)
    f1_arr = np.array(per_class_f1)
    counts_arr = np.array(per_class_counts)

    # 4a. Unweighted Macro Metrics
    macro_p = np.mean(p_arr) if len(p_arr) > 0 else 0.0
    macro_f1 = np.mean(f1_arr) if len(f1_arr) > 0 else 0.0
    macro_sens = np.mean(sensitivity_list) * 100 if sensitivity_list else 0.0
    macro_spec = np.mean(specificity_list) * 100 if specificity_list else 0.0

    # 🚀 4b. Missing Global Micro Metrics (Micro F1, Micro Precision, Micro Recall)
    active_targets = targets[:, active_class_indices]
    active_preds = preds[:, active_class_indices]
    
    global_tp = np.sum((active_targets == 1) & (active_preds == 1))
    global_fp = np.sum((active_targets == 0) & (active_preds == 1))
    global_fn = np.sum((active_targets == 1) & (active_preds == 0))
    
    micro_p = (global_tp / (global_tp + global_fp)) * 100 if (global_tp + global_fp) > 0 else 0.0
    micro_r = (global_tp / (global_tp + global_fn)) * 100 if (global_tp + global_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    # 4c. Prevalence-Weighted Macro Metrics
    total_positives = np.sum(counts_arr)
    if total_positives > 0:
        class_weights = counts_arr / total_positives
        weighted_macro_p = np.sum(p_arr * class_weights) * 100
        weighted_macro_sens = np.sum(r_arr * class_weights) * 100
        weighted_macro_spec = np.sum(spec_arr * class_weights) * 100  # 🚀 Added: Weighted Specificity
        weighted_macro_f1 = np.sum(f1_arr * class_weights) * 100
    else:
        weighted_macro_p, weighted_macro_sens, weighted_macro_spec, weighted_macro_f1 = 0.0, 0.0, 0.0, 0.0

    # 4d. Top-50 Most Frequent Diseases Benchmark
    if len(counts_arr) > 0:
        top50_k = min(50, len(counts_arr))
        top50_indices = np.argsort(counts_arr)[::-1][:top50_k]
        top50_macro_p = np.mean(p_arr[top50_indices]) * 100
        top50_macro_sens = np.mean(r_arr[top50_indices]) * 100
        top50_macro_spec = np.mean(spec_arr[top50_indices]) * 100  # 🚀 Added: Top-50 Specificity
        top50_macro_f1 = np.mean(f1_arr[top50_indices]) * 100
    else:
        top50_macro_p, top50_macro_sens, top50_macro_spec, top50_macro_f1 = 0.0, 0.0, 0.0, 0.0

    # 🚀 4e. Top-100 Most Frequent Diseases Benchmark
    if len(counts_arr) > 0:
        top100_k = min(100, len(counts_arr))
        top100_indices = np.argsort(counts_arr)[::-1][:top100_k]
        top100_macro_p = np.mean(p_arr[top100_indices]) * 100
        top100_macro_sens = np.mean(r_arr[top100_indices]) * 100
        top100_macro_spec = np.mean(spec_arr[top100_indices]) * 100
        top100_macro_f1 = np.mean(f1_arr[top100_indices]) * 100
    else:
        top100_macro_p, top100_macro_sens, top100_macro_spec, top100_macro_f1 = 0.0, 0.0, 0.0, 0.0

    # 5. Top-K Hit Rates & Precision@K
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
        "micro_f1": micro_f1,
        "micro_precision": micro_p,
        "micro_sensitivity": micro_r,
        "weighted_macro_f1": weighted_macro_f1,
        "weighted_macro_precision": weighted_macro_p,
        "weighted_macro_sensitivity": weighted_macro_sens,
        "weighted_macro_specificity": weighted_macro_spec,
        "top50_macro_f1": top50_macro_f1,
        "top50_macro_precision": top50_macro_p,
        "top50_macro_sensitivity": top50_macro_sens,
        "top50_macro_specificity": top50_macro_spec,
        "top100_macro_f1": top100_macro_f1,
        "top100_macro_precision": top100_macro_p,
        "top100_macro_sensitivity": top100_macro_sens,
        "top100_macro_specificity": top100_macro_spec,
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
        print_clinical_audit_report(
            metrics_summary=metrics_summary, 
            temp_alpha=temp_alpha, 
            calibration_beta=calibration_beta
        )
        
    return metrics_summary