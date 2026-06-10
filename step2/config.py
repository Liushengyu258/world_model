"""
全局配置模块:
统一管理世界模型训练所需的全部超参数和文件保存路径。
"""

import os
from dataclasses import dataclass, field

# 获取当前配置文件所在的绝对路径目录，用于定位模型权重的保存位置
_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class TrainConfig:
    """
    训练超参数配置类
    使用 Python 的 dataclass 装饰器，自动提供默认值且便于属性访问。
    """

    # ── 通用超参数 (General) ──────────────────────────────────
    
    # 批大小：每次梯度更新所使用的样本数量
    batch_size: int = 64
    
    # 隐空间维度 (Latent Dimension)：VAE 将图像压缩后得到的紧凑状态表示的维数 (z 的维度)
    latent_dim: int = 128
    
    # 循环神经网络 (GRU) 的隐层状态维度 (Hidden Dimension)
    hidden_dim: int = 256
    
    # 时序自注意力机制 (Self-Attention) 中的多头注意力数 (Multi-head Attention)
    num_heads: int = 4
    
    # 动力学模型中 GRU 的层数 (Layers of GRU)
    num_gru_layers: int = 2
    
    # 神经元失活率 (Dropout Rate)：用于防止过拟合，提高模型泛化能力
    dropout: float = 0.1

    # ── 阶段 1: VAE 训练参数 (Phase 1: VAE) ─────────────────────────
    
    # VAE 的总训练轮数 (Epochs)
    vae_epochs: int = 80
    
    # VAE 优化器 (Adam) 的初始学习率
    vae_lr: float = 1e-3
    
    # KL 散度退火比率：在前百分之多少的 Epochs 中将 KL 散度权重从 0 线性增加到 1.0
    # 0.2 表示在前 20% (即前 16 个 epoch) 进行 KL 退火，防止隐空间早期崩溃 (Latent Collapse)
    kl_anneal_ratio: float = 0.2

    # ── 阶段 2: 动力学模型训练参数 (Phase 2: Dynamics) ────────────────────
    
    # 动力学模型 (预测未来) 的总训练轮数 (Epochs)
    dyn_epochs: int = 300
    
    # 动力学模型优化器的初始学习率 (后续会通过 CosineAnnealing 调度器进行余弦退火)
    dyn_lr: float = 1e-3
    
    # 梯度剪切阈值：在反向传播时限制梯度的最大 L2 范数，防止循环神经网络中产生梯度爆炸 (Gradient Explosion)
    grad_clip: float = 1.0
    
    # Teacher Forcing (教师强迫) 衰减比率进度：
    # 0.5 表示在动力学模型训练的前 50% epochs 中，其混合采样率 (tf_ratio) 会从 1.0 线性递减至 0.0。
    # 之后 (后 50% 周期) 完全以自由运行模式 (Free-running) 预测，即用前一步的输出作为下一步的输入。
    tf_decay_ratio: float = 0.5

    # ── 输出保存路径 (Output Paths) ─────────────────────────────────
    
    # 阶段 1 训练完的 VAE 权重参数保存路径
    vae_weights: str = os.path.join(_DIR, "vae_weights.pth")
    
    # 阶段 2 训练完的动力学模型权重参数保存路径
    dyn_weights: str = os.path.join(_DIR, "dyn_weights.pth")
    
    # 评估时生成的模型「梦境」预测与真实时序对比图 (GIF 动图) 保存路径
    gif_path: str = os.path.join(_DIR, "world_model_dream.gif")

