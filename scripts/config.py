# scripts/config.py
import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Tuple, Set, List

@dataclass
class CardioConfig:
    # ─── 1. STORAGE AND ARTIFACT PATHS ───
    sql_file_path: str = "./bvtd_db_20251208.sql"
    master_cdha_csv: str = "master_cdha_cleaned.csv"
    master_xn_csv: str = "master_xn_cleaned.csv"
    train_csv_path: str = "train_patient_flattened.csv"
    val_csv_path: str = "val_patient_flattened.csv"
    codebook_json_path: str = "clinical_codebooks.json"
    checkpoint_dir: str = "./checkpoints"
    xai_export_dir: str = "./xai_exports"
    
    # Artifact File Names
    best_ssl_backbone_filename: str = "best_ssl_backbone.pt"
    unified_intermediary_filename: str = "unified_jepa_and_probe_checkpoint.pt"
    unified_final_filename: str = "unified_jepa_and_probe.pt"
    calibrated_thresholds_filename: str = "calibrated_thresholds.json"
    telemetry_asl_filename: str = "telemetry_asl_probe-fitting.csv"

    # ─── 2. DATA PROCESSING & FEATURE BOUNDS ───
    train_split_ratio: float = 0.65
    random_seed: int = 42
    clinical_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'sbp': (70.0, 200.0),       
        'dbp': (40.0, 130.0),       
        'mach': (30.0, 160.0),      
        'nhietdo': (35.0, 42.0),    
        'cannang': (30.0, 150.0),   
        'chieucao': (100.0, 200.0)  
    })
    clinical_stop_words: Set[str] = field(default_factory=lambda: {
        'triển', 'khai', 'thí', 'điểm', 'không', 'in', 'phim', 'theo', 'đề', 'án', 'byt',
        'và', 'các', 'của', 'tại', 'khoa', 'đề_án', 'thí_điểm'
    })

    # ─── 3. ARCHITECTURE & LATENT DIMENSIONS ───
    latent_dim: int = 512                  # Dimensionality of the shared latent space
    max_sequence_len: int = 256            # Max chronological sequence timeline tokens
    max_targets: int = 10                  # Max multi-label ICD targets recorded per patient
    num_slots: int = 24                    # Number of Perceiver latent pooling query slots
    augmented_slots: int = 26              # 24 dynamic slots + 1 age + 1 gender
    encoder_layers: int = 6                # Transformer Encoder layers in Context/Target backbones
    nhead: int = 8                         # Transformer attention heads
    fourier_frequencies: int = 32          # Fourier frequency channels for periodic numerical embeddings
    
    probe_type: str = "attentive"          # Options: "attentive" (Hopfield+LAAT) or "linear"
    num_prototypes: int = 64               # Hopfield memory prototype centroids
    num_hopfield_heads: int = 8            # Subspace heads in Modern Hopfield layer
    probe_dropout: float = 0.20
    covariate_scale: float = 0.50          # L2 norm scaling factor for static age/gender tokens

    # ─── 4. LORA ADAPTER CONFIGURATION ───
    encoder_lora: bool = True
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_target_names: List[str] = field(default_factory=lambda: [
        "linear1", "linear2", "channel_mlp", "slot_combiner"
    ])

    # ─── 5. HARDWARE & COMPUTE ROUTING ───
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size: int = 256                  # Restored from 4 to 256 for stable VICReg statistics
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16
    patience: int = 4                      # Early-stopping patience budget (epochs)
    log_interval: int = 50                 # Logging step cadence

    # ─── 6. PHASE 1: SELF-SUPERVISED PRETRAINING (JEPA) ───
    k_min: int = 2                         # Minimum dynamic masking target horizon
    k_max: int = 6                         # Maximum dynamic masking target horizon
    alpha_align: float = 25.0              # L1 Smooth Alignment Loss weight
    alpha_var: float = 25.0                # Projected VICReg Variance Loss weight
    alpha_backbone_v: float = 150.0        # Direct Backbone Variance Loss weight
    alpha_cov: float = 150.0               # Projected VICReg Covariance Loss weight
    alpha_diverse: float = 25.0            # Cross-slot orthogonal diversity loss weight
    alignment_smooth_l1_beta: float = 0.5
    target_std: float = 0.20               # Target feature standard deviation floor
    tau: float = 0.996                     # Momentum EMA coefficient for Target Encoder
    pretrain_lr: float = 4.2e-4
    pretrain_epochs: int = 10
    pretrain_wgt_decay: float = 1e-2
    pretrain_warmup_ratio: float = 0.15

    # ─── 7. PHASE 2: DOWNSTREAM PROBE FITTING ───
    probe_lr: float = 2e-4
    probe_lr_backbone_scale: float = 0.25  # Scaler for fine-tuning backbone LoRA weights
    probe_lr_assembler_scale: float = 0.50 # Scaler for Manifold Assembler weights
    probe_epochs: int = 10
    probe_wgt_decay: float = 1e-4
    
    # Class-Aware Asymmetric Loss (ASL) Hyperparameters
    asl_gamma_pos: float = 0.0
    asl_gamma_neg_base: float = 4.5
    asl_beta_neg_base: float = 2.5
    asl_delta_beta: float = 1.0            # Dynamic boost for rare long-tail classes
    
    # Phase 2 Multi-Task Loss Weighting
    loss_weight_cls: float = 1.00
    loss_weight_cardinality_mse: float = 0.05
    loss_weight_prototype_diversity: float = 1.00
    loss_weight_label_cooccurrence: float = 15.0
    proto_loss_scale: float = 100.0

    # ─── 8. PHASE 2 MODEL SELECTION SCORECARD (STRATEGY 2: HARMONIC) ───
    harmonic_core_weight: float = 0.90     # Weight for Harmonic Mean of (PR-AUC, F1, Adaptive Precision)
    top5_rate_weight: float = 0.10         # Weight for Top-5 Differential Coverage Safety Net

    # ─── 9. EVALUATION & SAFETY CALIBRATION ───
    eval_temp_alpha: float = 0.15          # Temperature scaling alpha for rare-class un-suppression
    eval_flat_threshold: float = 0.15       # Fixed anchor decision threshold for baseline audit
    min_positive_prevalence: int = 2       # Minimum positive validation samples required per class

    # ─── 10. XAI ANALYTICS & VISUALIZATION ───
    xai_max_samples: int = 3000
    xai_umap_n_neighbors: int = 50
    xai_umap_min_dist: float = 0.40
    xai_umap_metric: str = "cosine"
    xai_hopfield_beta_vis: float = 16.0

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.xai_export_dir, exist_ok=True)