"""
组合世界模型 (World Model):
结合 感知 (VAE) 与 想象 (Dynamics) 的完整世界模型。

在强化学习和具身智能 (Embodied AI) 中，世界模型的核心思想是：
1. 感知 (Perception) -> 由 VAE 承担：将外部环境复杂、高维、具有冗余信息的像素观测 (Images) 压缩为低维的、语义紧凑的特征隐向量 z。
2. 想象/预测 (Prediction/Imagination) -> 由 DynamicsModel 承担：在纯粹的隐空间内，根据过去的状态模拟/推演未来世界状态 z_next 的演化，从而实现无需与真实世界交互即可在“梦境”中进行规划和策略学习。
"""

import torch.nn as nn

from .vae import VAE
from .dynamics import DynamicsModel


class WorldModel(nn.Module):
    """
    完整世界模型类
    
    整合感知网络 (VAE) 与动力学预测网络 (Dynamics)，
    为智能体构建一个“大脑环境模拟器”。
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        """
        参数:
            latent_dim (int): 隐空间状态向量维度 (z)，控制智能体感知的特征瓶颈大小
            hidden_dim (int): 动力学网络 GRU/Attention 内部隐层的表示维度，控制智能体的“思维/记忆”容量
        """
        super().__init__()
        # 1. 感知子网络 (视觉)：负责与像素世界进行桥接
        self.vae = VAE(latent_dim)
        
        # 2. 动力学子网络 (时序)：负责在隐层中推演时空演化规律
        self.dynamics = DynamicsModel(latent_dim, hidden_dim)

