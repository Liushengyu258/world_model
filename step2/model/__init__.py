"""
模型包 (Models Package):
将变分自编码器 (VAE)、循环门控动力学网络 (DynamicsModel) 以及各基础组件 (ResidualBlock, SelfAttention) 整合暴露。

包内子模块结构划分：
- blocks.py       : 基础神经网砖块，包含卷积残差块 (ResidualBlock) 和多头自注意力机制 (SelfAttention)。
- vae.py          : 视觉感知系统，负责 64x64 图像观测与低维连续隐向量的编解码映射与重参数化。
- dynamics.py     : 认知演化系统，负责在隐空间中结合 GRU 与 Attention 建立数字运动物理学的动力学网络。
- world_model.py  : 顶层容器，将上述两者（感知与想象）融合为完整世界模型 (WorldModel)。
"""

from .blocks import ResidualBlock, SelfAttention
from .vae import VAE
from .dynamics import DynamicsModel
from .world_model import WorldModel

# 定义外界使用 from model import * 时暴露出的公开类与函数
__all__ = [
    "ResidualBlock",
    "SelfAttention",
    "VAE",
    "DynamicsModel",
    "WorldModel",
]

