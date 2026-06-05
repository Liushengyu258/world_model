"""变分自编码器 (VAE)"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock


class Encoder(nn.Module):
    """
    卷积编码器: 64x64 -> 4x4 -> (mu, logvar)
    每个下采样阶段后接残差块，增强特征提取
    """

    def __init__(self, latent_dim: int, in_channels: int = 1):
        super().__init__()
        self.down = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            # 32 -> 16
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            # 16 -> 8
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            # 8 -> 4
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.down(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """
    卷积解码器: z -> 4x4 -> 64x64
    每个上采样阶段后接残差块
    """

    def __init__(self, latent_dim: int, out_channels: int = 1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.up = nn.Sequential(
            # 4 -> 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            # 8 -> 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            # 16 -> 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            # 32 -> 64
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.up(self.fc(z).unflatten(1, (256, 4, 4)))


class VAE(nn.Module):
    """变分自编码器: 重参数化采样 + ELBO 损失"""

    def __init__(self, latent_dim: int = 128, in_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, in_channels)
        self.decoder = Decoder(latent_dim, in_channels)

    # ── 采样 ──────────────────────────────────────

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + std * eps  (重参数化技巧)"""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """仅编码，返回采样后的 z (推理用)"""
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    # ── 前向 ──────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    # ── 损失 ──────────────────────────────────────

    def loss_function(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ELBO = BCE + kl_weight * KL(q(z|x) || N(0,1))"""
        bce = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return bce + kl_weight * kl, bce, kl
