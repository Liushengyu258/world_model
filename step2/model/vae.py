"""
变分自编码器模块 (Variational Autoencoder - VAE):
实现高维像素图像与低维连续分布隐空间之间的双向映射。
相比普通自编码器 (AE)，VAE 的隐空间具有连续性和可采样性，是世界模型中处理“视觉感知”的标准组件。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock


class Encoder(nn.Module):
    """
    卷积编码器 (Convolutional Encoder)
    功能: 将 64x64 的单通道输入图像压缩提取，最后输出隐层分布的均值 (mu) 与对数方差 (logvar)。
    
    下采样控制：
    使用 Conv2d(kernel_size=4, stride=2, padding=1) 逐层将特征图的高宽减半：
    64x64 (输入) -> 32x32 -> 16x16 -> 8x8 -> 4x4 (最末卷积层)
    并在每个下采样阶段后加入残差块 (ResidualBlock)，增强深层特征表达。
    """

    def __init__(self, latent_dim: int, in_channels: int = 1):
        """
        参数:
            latent_dim (int): 压缩后的隐状态维度 (z 的维数)
            in_channels (int): 输入图像通道数，灰度图为 1，彩色图为 3
        """
        super().__init__()
        self.down = nn.Sequential(
            # 第一阶段: 64x64 -> 32x32，通道数 1 -> 32
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            
            # 第二阶段: 32x32 -> 16x16，通道数 32 -> 64
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            
            # 第三阶段: 16x16 -> 8x8，通道数 64 -> 128
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            
            # 第四阶段: 8x8 -> 4x4，通道数 128 -> 256
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        
        # 展平后的大小为 256 * 4 * 4 = 4096 维特征
        # 分别使用两个全连接层来输出：
        # fc_mu: 隐状态概率分布的均值 (mu)
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        # fc_logvar: 隐状态概率分布的对数方差 (log_variance)，使用对数是为了保证计算出的标准差为正数
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：编码图像
        参数:
            x (Tensor): 输入图像，形状为 (B, C, H, W)
        返回:
            mu (Tensor): 均值张量，形状为 (B, latent_dim)
            logvar (Tensor): 对数方差张量，形状为 (B, latent_dim)
        """
        # self.down(x) 输出形状为 (B, 256, 4, 4)，通过 .flatten(1) 将其展平为 (B, 4096)
        h = self.down(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """
    卷积解码器 (Convolutional Decoder)
    功能: 输入隐向量 z，通过转置卷积 (Deconvolution) 逐步放大恢复，重构出原始的 64x64 图像。
    
    上采样控制：
    与 Encoder 的下采样正好相反，利用 ConvTranspose2d(kernel_size=4, stride=2, padding=1) 将高宽加倍：
    4x4 (输入) -> 8x8 -> 16x16 -> 32x32 -> 64x64 (重构输出)
    """

    def __init__(self, latent_dim: int, out_channels: int = 1):
        """
        参数:
            latent_dim (int): 输入隐向量的维度
            out_channels (int): 输出重构图像的通道数，这里为 1 (Moving MNIST 灰度图)
        """
        super().__init__()
        # 先用一个全连接层将低维隐向量 (latent_dim) 映射回最末端卷积层的展平尺寸 256 * 4 * 4 = 4096 维
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.up = nn.Sequential(
            # 第一阶段: 4x4 -> 8x8，通道数 256 -> 128
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            
            # 第二阶段: 8x8 -> 16x16，通道数 128 -> 64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            
            # 第三阶段: 16x16 -> 32x32，通道数 64 -> 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            
            # 第四阶段: 32x32 -> 64x64，通道数 32 -> out_channels
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
            # 最后一层使用 Sigmoid 激活函数，强迫输出像素值限制在 [0.0, 1.0] 区间内，以便匹配归一化后的真实图像
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播：解码重构
        参数:
            z (Tensor): 隐空间向量，形状为 (B, latent_dim)
        返回:
            Tensor: 重构的图像序列，形状为 (B, out_channels, 64, 64)
        """
        # 1. 线性映射并解压还原为四维卷积格式：
        # self.fc(z) 的形状是 (B, 4096)，通过 .unflatten(1, (256, 4, 4)) 变形为特征图格式 (B, 256, 4, 4)
        h_flat = self.fc(z)
        h_conv = h_flat.unflatten(1, (256, 4, 4))
        # 2. 通过转置卷积上采样网络
        return self.up(h_conv)


class VAE(nn.Module):
    """
    变分自编码器主类 (VAE)
    
    不同于普通自编码器 (AE) 直接将输入映射成隐空间的一个确定点，
    VAE 将输入映射成隐空间的一个“概率分布” (用均值 mu 和方差 std 描述)，然后再从该分布中采样出一个隐向量 z 进行解码。
    这确保了整个隐空间是连续且没有空洞的，便于我们进行生成和自回归预测。
    """

    def __init__(self, latent_dim: int = 128, in_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, in_channels)
        self.decoder = Decoder(latent_dim, in_channels)

    # ── 重参数化采样 ──────────────────────────────────────

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧 (Reparameterization Trick)
        
        【为什么需要重参数化？】
        如果我们直接从 N(mu, std^2) 这个分布中进行随机采样，采样操作 (Sampling) 是无法求导的。
        无法求导意味着误差无法通过反向传播 (Backpropagation) 传回给 Encoder，导致网络无法使用梯度下降训练。
        
        【解决方案】：
        将随机性剥离出来！
        1. 从标准正态分布中随机采样一个随机噪声 eps ~ N(0, I)。(无需训练，不需要梯度)
        2. 通过确定的代数运算计算: z = mu + std * eps
        这样，z 依然具有随机性，且对于均值 mu 和标准差 std 而言是完全可导的。
        
        公式: z = mu + exp(0.5 * logvar) * eps
        """
        # 根据对数方差计算标准差：std = e^(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        # 采样一个与标准差形状完全相同的标准正态分布噪声 epsilon
        eps = torch.randn_like(std)
        # 缩放平移得到 z
        return mu + std * eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅执行编码与采样阶段 (主要用于动力学训练或测试推理)。
        
        输入图像，返回采样后得到的隐空间向量 z。
        """
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    # ── 前向传播 ──────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE 前向流程：
        输入图像 x -> 编码得到分布参数 (mu, logvar) -> 重参数化采样 z -> 解码得到重构图像 x_recon
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    # ── 损失函数 (ELBO) ──────────────────────────────────────

    def loss_function(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 VAE 的损失函数：变分下界损失 (ELBO Loss)
        
        由两部分组成：
        1. 重构损失 (Reconstruction Loss - BCE)：
           衡量重构出来的像素图 x_recon 与原图 x 的接近程度。这里使用二值交叉熵 (Binary Cross Entropy)。
        2. KL 散度损失 (Kullback-Leibler Divergence)：
           衡量编码器预测的隐层分布 N(mu, std^2) 与标准正态分布 N(0, I) 的差异程度。
           其作用是作为正则项，强迫隐空间各区域分布紧凑、连续，不产生过拟合孤岛。
        
        参数:
            x_recon (Tensor): VAE 重构的像素图，Shape 为 (B, C, H, W)
            x (Tensor): 真实的输入像素图，Shape 为 (B, C, H, W)
            mu (Tensor): 均值向量，Shape 为 (B, latent_dim)
            logvar (Tensor): 对数方差向量，Shape 为 (B, latent_dim)
            kl_weight (float): KL 散度的退火权重，在训练早期较小，后期增大至 1.0
            
        返回:
            total_loss (Tensor): 结合后的总损失 (bce + kl_weight * kl)
            bce (Tensor): 平均单样本重构损失
            kl (Tensor): 平均单样本 KL 散度损失
        """
        # 1. 重构损失：
        # F.binary_cross_entropy 算出的总 BCE 除以批大小 x.size(0)，得到每个样本的平均重构损失
        bce = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
        
        # 2. KL 散度损失的数学闭式解 (Closed-form solution for Gaussian KL)：
        # 对于多维独立高斯分布，KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # 同样除以 x.size(0) 取平均值
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # 返回加权后的总损失，以及两个子项（用于单独打印监控）
        return bce + kl_weight * kl, bce, kl

