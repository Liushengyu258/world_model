"""
基础构建块模块:
包含图像特征提取常用的“残差卷积块 (Residual Convolutional Block)”和时序关系捕捉常用的“自注意力机制 (Self-Attention Block)”。
这些基础块是搭建更深层神经网络（如 VAE 和 DynamicsModel）的砖瓦。
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    残差卷积块 (Residual Conv Block)
    结构: Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d + Shortcut (Skip Connection) -> ReLU
    
    残差连接 (Skip Connection / Shortcut) 的核心作用：
    1. 允许输入信号毫无阻碍地向后传递，极大地缓解了在深层卷积网络中容易出现的“梯度消失 (Gradient Vanishing)”问题。
    2. 改变了网络学习的目标：网络不再需要去逼近一个复杂的恒等映射 (Identity Mapping)，
       而是只需要学习输入与输出之间的“残差 (Residual)”，使优化问题变得简单许多。
    """

    def __init__(self, channels: int):
        """
        参数:
            channels (int): 输入与输出的通道数。
                            本模块不改变特征图的尺寸(H, W)与通道数(C)，专用于深层特征提取和非线性抽象。
        """
        super().__init__()
        # nn.Conv2d 参数说明：输入通道，输出通道，卷积核大小=3，步长=1，填充=1。
        # 步长为 1 且 填充为 1 的 3x3 卷积，可以完美保持特征图的高度 (H) 和宽度 (W) 不变。
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels), # 批归一化 (Batch Normalization)：在通道维度上规范化特征分布，加速收敛并起到正则化作用
            nn.ReLU(inplace=True),    # inplace=True 可以直接覆盖内存，节省显存开销
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x (Tensor): 输入特征图，形状为 (B, C, H, W)
        返回:
            Tensor: 残差激活后的输出，形状仍为 (B, C, H, W)
        """
        # 恒等映射分支 (x) 与残差学习分支 (self.block(x)) 元素级相加，最后通过 ReLU 激活函数输出
        return self.relu(self.block(x) + x)


class SelfAttention(nn.Module):
    """
    多头自注意力机制 (Multi-head Self-Attention) + 残差连接 + 层归一化 (LayerNorm)
    
    多头自注意力在动力学模型中的作用：
    1. GRU 等循环网络随着时间步拉长，容易遗忘很久之前的时序信息（长期依赖问题）。
    2. 自注意力机制允许每一个时间步直接与其他所有时间步进行交互，打破了距离限制，能够精确捕获长距离的时间依赖。
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        """
        参数:
            hidden_dim (int): 时序特征向量维度
            num_heads (int): 多头注意力的头数，必须能被 hidden_dim 整除
        """
        super().__init__()
        # batch_first=True 说明输入的张量形状为 (Batch_Size, Seq_Len, Feature_Dim)。
        # PyTorch 默认期望时序在第一维 (Seq_Len, Batch_Size, Feature_Dim)，
        # 设置该参数可使其符合我们最习惯的时序数据流动格式。
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # 使用层归一化 (LayerNorm) 而不是批归一化 (BatchNorm)。
        # 时序/Transformer 架构中，LayerNorm（在单个样本的特征维度上做归一化）效果更稳定，
        # 不受 Sequence 长度变化或 Batch Size 过小的干扰。
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x (Tensor): 时序特征张量，形状为 (B, T, hidden_dim)
        返回:
            Tensor: 自注意力增强并归一化后的输出，形状仍为 (B, T, hidden_dim)
        """
        # 为什么输入三个相同的值 x？
        # 在“自注意力 (Self-Attention)”中，查询向量 (Query)、键向量 (Key) 和值向量 (Value) 均源自相同的输入 x。
        # self.attn 返回一个元组：(注意力层输出, 注意力权重)，我们只关注特征输出，所以用占位符 _ 丢弃权重。
        # attn_out 形状: (B, T, hidden_dim)
        attn_out, _ = self.attn(x, x, x)
        
        # 加上残差连接 (x + attn_out)，然后进行 LayerNorm
        return self.norm(x + attn_out)

