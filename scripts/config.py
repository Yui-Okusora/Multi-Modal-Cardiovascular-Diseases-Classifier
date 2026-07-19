from dataclasses import dataclass
import torch
import os

@dataclass
class CardioConfig:
    # Storage and Artifact Paths
    train_csv_path: str = "train_patient_flattened.csv"
    val_csv_path: str = "val_patient_flattened.csv"
    codebook_json_path: str = "clinical_codebooks.json"
    checkpoint_dir: str = "./checkpoints"
    
    # Structural Sequence Dimension Budgets
    latent_dim: int = 512                  # Structural capacity of the shared 512-D latent coordinates
    max_sequence_len: int = 256            # Max chronological sequence timeline blocks allowed per session
    max_targets: int = 10                  # Max simultaneous multi-label ICD discharge categories recorded
    num_slots: int = 24                    # Number of fixed Perceiver latent query pooling slots (K)
    encoder_layers: int = 6                # Number of Transformer layers in Context/Target Encoders
    
    probe_type: str = "attentive"       # Options: "spatial_conv" or "linear"

    # Compute Hardware Allocation Routing
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 256                  # Batched data dimensions balanced against sequence capacity
    grad_clip_norm: float = 1.0            # Restricts gradient explosions over deep self-attention steps
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16
    
    # =================== Phase 1 ===================
    # JEPA Alignment Coefficients
    alpha_align: float = 25.0              # Weighting multiplier L1 Loss
    alpha_var: float = 25.0                # Weighted slightly higher to aggressively break collapse
    alpha_backbone_v: float = 150.0
    alpha_cov: float = 150.0               # Weighting multiplier factor for VICReg cross-channel decorrelation
    alpha_diverse: float = 25.0            # Weighting multiplier embedding slots diversity
    tau: float = 0.996                     # EMA tracking coefficient (m)
    
    # Optimization Learning Velocity Constraints
    pretrain_lr: float = 4.2e-4            # Conservative pretraining learning rate for coordinate scaling
    pretrain_epochs: int = 10              # Epoch loops allowing associative valleys to organize
    pretrain_wgt_decay: float = 1e-2       # L2 regularization factor over trainable matrices
    
    # =================== Phase 2 ===================
    # Optimization Learning Velocity Constraints
    probe_lr: float = 1e-3                 # Slightly increased to accelerate initial coordinate transitions
    probe_epochs: int = 10                 # Epoch loops allowing associative valleys to organize
    probe_wgt_decay: float = 1e-2          # L2 regularization factor to cleanly curve the loss bowl
    patience: int = 3                      # Real early-stopping constraint tailored to a 5-epoch budget

    # Runtime Logging Cadence
    log_interval: int = 50

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)