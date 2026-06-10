"""
潜空间动力学预测模型模块 (Latent Dynamics Model):
使用 GRU 时序模型 + 自注意力机制 (Self-Attention) + 前馈网络 (FFN) 在 VAE 的低维隐空间 z 中建模时序演化。
该模块扮演了世界模型中“思维与推演大脑”的角色，能够仅凭前几帧的物理记忆，预测未来任何步的运动轨迹。
"""

import torch
import torch.nn as nn

from .blocks import SelfAttention


class DynamicsModel(nn.Module):
    """
    隐空间时序预测模型
    
    该模型学习隐特征序列的时空转换规律（即“物理法则”：比如物体的速度、方向、重力反弹等）。
    
    网络结构流程：
    输入隐特征 (latent_dim) -> GRU (时间步关联) -> Self-Attention (长依赖增强)
    -> FFN + 残差连接 (非线性特征抽象) -> 全连接线性映射 -> 预测的下一帧隐特征 (latent_dim)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_gru_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        参数:
            latent_dim (int): 隐空间状态向量维度 (z)
            hidden_dim (int): 动力学网络 GRU/Attention 内部隐层的表示维度
            num_heads (int): 时序自注意力机制的多头数
            num_gru_layers (int): GRU (门控循环单元) 的层数
            dropout (float): Dropout 比例，防止网络在拟合运动物理轨迹时过拟合
        """
        super().__init__()
        # 1. GRU 时序建模层
        # GRU (Gated Recurrent Unit) 是循环神经网络的一种。相较于 LSTM，其参数更少、计算更高效，且能很好地缓解时序梯度消失。
        # batch_first=True 规定输入形状为 (B, T, latent_dim)
        self.gru = nn.GRU(latent_dim, hidden_dim, num_gru_layers, batch_first=True)
        
        # 2. 自注意力层
        # 允许网络在预测未来时，跨越 GRU 的步骤限制，直接关注历史序列中某些最关键、信息量最大的帧。
        self.attn = SelfAttention(hidden_dim, num_heads)
        
        # 3. 前馈网络层 (Feed Forward Network / FFN)
        # 提升模型的非线性表达能力。
        # 结构为: Linear -> GELU -> Linear -> Dropout
        # GELU (高斯误差线性单元) 激活函数比传统的 ReLU 更平滑，在 Transformer 和生成时序网络中表现更佳。
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # 4. 残差之后的归一化层
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 5. 输出投影层
        # 将 GRU+Attention 计算出的隐层高维语义表示 (hidden_dim) 映射回 VAE 的低维隐状态空间 (latent_dim)，
        # 这样模型输出的预测特征可以直接送入 VAE 的 Decoder 解码成图像，或者在下一轮自回归中作为输入。
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, z_seq: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向时序预测
        
        参数:
            z_seq (Tensor): 输入的隐空间特征时序，Shape 为 (B, T, latent_dim)
            hidden (Tensor, 可选): GRU 的初始状态，Shape 为 (num_layers, B, hidden_dim)
                                 如果是序列的第一步，则传入 None，网络会自动初始化为全 0。
                                 
        返回:
            z_pred (Tensor): 预测未来的隐向量序列，Shape 为 (B, T, latent_dim)。
                             例如输入为 z_{0:T-1}，则 z_pred 对应预测的 z_{1:T}。
            h_n (Tensor): 经过本轮计算更新后的 GRU 隐层状态，Shape 为 (num_layers, B, hidden_dim)。
                          在自回归预测时，我们将 h_n 传递给下一步，用以维持时间连贯性。
        """
        # 1. 运行 GRU 循环层。gru_out 形状: (B, T, hidden_dim)，代表每个时间步的隐层输出。
        # h_n 代表整个序列计算结束后的最末时刻隐藏状态。
        gru_out, h_n = self.gru(z_seq, hidden)
        
        # 2. 运行多头自注意力层。attn_out 形状: (B, T, hidden_dim)
        attn_out = self.attn(gru_out)
        
        # 3. FFN 非线性映射与残差连接 + 层归一化
        # out 形状: (B, T, hidden_dim)
        out = self.norm(attn_out + self.ffn(attn_out))
        
        # 4. 线性变换，映射回 VAE 隐空间维度。
        # z_pred 形状: (B, T, latent_dim)
        z_pred = self.fc(out)
        
        return z_pred, h_n

