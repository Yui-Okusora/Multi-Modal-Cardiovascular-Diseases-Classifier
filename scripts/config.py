from dataclasses import dataclass
import torch

@dataclass
class CardioConfig:
    # Path Configuration
    cdha_path: str = "master_cdha_cleaned.csv"
    xn_path: str = "master_xn_cleaned.csv"
    checkpoint_dir: str = "./checkpoints"
    
    # Model Configurations
    text_model_name: str = "vinai/phobert-base-v2"
    vitals_dim: int = 12
    latent_dim: int = 128
    text_dim: int = 768
    max_sequence_len: int = 64
    
    # Training Control Knobs
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 256            # Maximizes VRAM parallelism, smoothing out gradient steps
    pretrain_lr: float = 5e-5       # Slightly higher to account for the larger batch size
    downstream_lr: float = 5e-4     # More conservative to prevent the classification head from over-correcting
    pretrain_epochs: int = 35       # Extended to allow the deeper coordinate alignment to fully mature
    downstream_epochs: int = 30     # Paired with weight decay for a smooth convergence curve
    grad_clip_norm: float = 0.5     # Stricter clipping to prevent gradient explosions in deep layers
    weight_decay: float = 1e-4      # L2 Regularization barrier against memorization
    
    # Telemetry
    log_interval: int = 10