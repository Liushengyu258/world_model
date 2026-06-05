"""全局配置"""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """训练超参数"""

    # ── 通用 ──────────────────────────────────
    batch_size: int = 64
    latent_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    num_gru_layers: int = 2
    dropout: float = 0.1

    # ── Phase 1: VAE ─────────────────────────
    vae_epochs: int = 80
    vae_lr: float = 1e-3
    kl_anneal_ratio: float = 0.2        # 前 20% epochs 线性退火 KL

    # ── Phase 2: Dynamics ────────────────────
    dyn_epochs: int = 300
    dyn_lr: float = 1e-3
    grad_clip: float = 1.0
    tf_decay_ratio: float = 0.5         # teacher forcing 衰减到 0 的进度

    # ── 输出 ─────────────────────────────────
    vae_weights: str = "vae_weights.pth"
    dyn_weights: str = "dyn_weights.pth"
    gif_path: str = "world_model_dream.gif"
