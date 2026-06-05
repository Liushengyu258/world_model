"""潜空间动力学模型: GRU + Self-Attention + FFN"""

import torch
import torch.nn as nn

from .blocks import SelfAttention


class DynamicsModel(nn.Module):
    """
    在潜空间预测下一帧的序列模型

    结构:  GRU (时序建模)
         -> Self-Attention (捕获长距离依赖)
         -> FFN + 残差 (非线性变换)
         -> Linear (映射回潜空间)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_gru_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(latent_dim, hidden_dim, num_gru_layers, batch_first=True)
        self.attn = SelfAttention(hidden_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, z_seq: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_seq:  (B, T, latent_dim) 输入潜向量序列
            hidden: (num_layers, B, hidden_dim) GRU 隐状态
        Returns:
            z_pred: (B, T, latent_dim) 预测的下一帧潜向量
            h_n:    (num_layers, B, hidden_dim) 更新后的隐状态
        """
        gru_out, h_n = self.gru(z_seq, hidden)
        attn_out = self.attn(gru_out)
        out = self.norm(attn_out + self.ffn(attn_out))
        z_pred = self.fc(out)
        return z_pred, h_n
