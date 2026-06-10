"""
模型定义模块 (Baseline 阶段):
包含基本的 卷积自编码器 (Autoencoder) 和 隐空间时序预测模型 (LSTM)。

【重要对比：Baseline 阶段 VS 进阶阶段】
1. 视觉感知：
   - Step 1: 采用普通自编码器 (Autoencoder)。Encoder 提取的是一个固定的隐向量 z（无随机分布），容易导致隐空间分布极不规则。
   - Step 2: 采用变分自编码器 (VAE)。Encoder 预测的是高斯分布的 mu 和 logvar，结合重参数化采样得到 z。隐空间连续可积、更平滑，且加入了残差卷积块 (ResidualBlock)。
2. 动力学模型：
   - Step 1: 采用标准的 双层 LSTM (长短期记忆网络) 做时序建模，利用全连接层映射。
   - Step 2: 采用更轻量高效的 双层 GRU，并创新地融合了多头自注意力机制 (Self-Attention) 和前馈网络 (FFN) 以捕获超长程依赖。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    基础卷积编码器 (Simple Convolutional Encoder)
    功能: 将 64x64 单通道图像降维压缩至 latent_dim 维隐空间向量。
    
    结构选择：
    无残差块，使用 4 层普通的 nn.Conv2d(kernel_size=4, stride=2, padding=1) 进行 2 倍下采样，中间配合普通的 ReLU 激活。
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        # 逐层高宽折半，通道数递增：
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)   # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1) # 16x16 -> 8x8
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1) # 8x8 -> 4x4
        # 最末端 256*4*4 = 4096 维，使用单层全连接层直接映射到 latent_dim 隐特征空间
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)
        
    def forward(self, x):
        """
        前向编码：
        参数:
            x (Tensor): 形状为 (Batch_Size, 1, 64, 64) 的单帧图像
        返回:
            z (Tensor): 形状为 (Batch_Size, latent_dim) 的确定性隐层表示
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # 将其展平为 2 维张量 (Batch_Size, 4096)
        x = x.reshape(x.size(0), -1)
        # 映射到隐状态空间
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    """
    基础卷积解码器 (Simple Convolutional Decoder)
    功能: 输入确定的隐层向量 z，通过 4 层转置卷积 (ConvTranspose2d) 恢复重建为 64x64 单通道像素图。
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        # 全连接层将隐向量放大为 4096 维 (便于恢复为特征图尺寸 256x4x4)
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # 逐层高宽翻倍，通道数缩减：
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1) # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   # 16x16 -> 32x32
        self.deconv4 = nn.ConvTranspose2d(32, 1, 4, 2, 1)    # 32x32 -> 64x64
        
    def forward(self, z):
        """
        前向解码重构：
        参数:
            z (Tensor): 形状为 (Batch_Size, latent_dim) 的隐层特征
        返回:
            x (Tensor): 形状为 (Batch_Size, 1, 64, 64) 的重构像素图像
        """
        x = self.fc(z)
        # 将一维向量形状重组回四维特征图格式: (Batch_Size, Channel=256, Height=4, Width=4)
        x = x.reshape(x.size(0), 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        # 最后一层转置卷积后通过 Sigmoid 激活，使得每个像素值都限定在 0.0 到 1.0 的合理区间内
        x = torch.sigmoid(self.deconv4(x)) 
        return x


class Autoencoder(nn.Module):
    """
    卷积自编码器顶层容器 (Autoencoder)
    
    经典 AE 结构：输入高维图像 -> Encoder 压缩至瓶颈层隐向量 z -> Decoder 解码重构。
    其优化目标是最小化重构误差（即图像前后对比的 BCE / MSE 损失）。
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def forward(self, x):
        """
        前向流程：
        返回:
            x_recon: 重建图像，Shape 为 (B, 1, 64, 64)
            z: 输入对应的瓶颈层隐状态，Shape 为 (B, latent_dim)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class LatentDynamicsModel(nn.Module):
    """
    隐空间时序预测动力学模型 (Simple LSTM Latent Dynamics)
    
    使用 PyTorch 经典的 双层 LSTM (Long Short-Term Memory) 网络，
    基于历史连续隐特征时序预测未来的隐特征值。
    """

    def __init__(self, latent_dim=128, hidden_dim=256):
        """
        参数:
            latent_dim (int): VAE / AE 隐特征维度
            hidden_dim (int): LSTM 隐层特征维度
        """
        super().__init__()
        # batch_first=True 规定输入的数据组织格式为 (Batch_Size, Seq_Len, latent_dim)
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        # 最后的线性全连接层，负责将 LSTM 隐层输出 (hidden_dim) 重新投射到隐特征空间维度 (latent_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, z_seq):
        """
        前向时序关联计算：
        参数:
            z_seq (Tensor): 隐时序输入，Shape 为 (B, T, latent_dim)
        返回:
            z_next_pred (Tensor): 全序列在各个对应下一时刻的预测输出，Shape 为 (B, T, latent_dim)
            (h_n, c_n): LSTM 最终的时间步特征记忆隐藏状态和单元状态
        """
        # lstm_out 形状: (B, T, hidden_dim)；元组 (h_n, c_n) 代表 LSTM 在最末时间步的记忆
        lstm_out, (h_n, c_n) = self.lstm(z_seq)
        
        # 对每一个时间步提取出的序列表示进行线性投影，直接自回归式预测其对应的下一时刻状态。
        # 也就是说，输入 z_{0:T-1}，模型预测并输出 z_pred_{1:T}。
        z_next_pred = self.fc(lstm_out)
        return z_next_pred, (h_n, c_n)


class WorldModel(nn.Module):
    """
    组合世界模型类 (Combined World Model)
    包含：
    1. 负责感知重构的自编码器部分 (Autoencoder)
    2. 负责时序演化联想的动力学预测部分 (LatentDynamicsModel)
    
    虽然我们通常分别进行这两个子模型的训练，但在推理或完整规划时，它们会被融为一体。
    """
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.autoencoder = Autoencoder(latent_dim)
        self.dynamics = LatentDynamicsModel(latent_dim, hidden_dim)


if __name__ == "__main__":
    # 模型架构与控制流张量维度的简易单元测试
    device = torch.device('cpu')
    x = torch.rand(4, 1, 64, 64)
    ae = Autoencoder().to(device)
    x_recon, z = ae(x)
    print(f"Autoencoder Input: {x.shape}, Recon: {x_recon.shape}, Latent: {z.shape}")
    
    z_seq = torch.rand(4, 10, 128) # 10 帧长度的隐特征视频时序
    dyn = LatentDynamicsModel().to(device)
    z_next_pred, _ = dyn(z_seq)
    print(f"Dynamics Input: {z_seq.shape}, Next Pred: {z_next_pred.shape}")

