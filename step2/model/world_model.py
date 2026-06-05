"""组合世界模型 (VAE + Dynamics)"""

import torch.nn as nn

from .vae import VAE
from .dynamics import DynamicsModel


class WorldModel(nn.Module):
    """感知 (VAE) + 想象 (Dynamics) 的完整世界模型"""

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.vae = VAE(latent_dim)
        self.dynamics = DynamicsModel(latent_dim, hidden_dim)
