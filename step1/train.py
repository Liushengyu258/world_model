"""
训练管理模块 (Baseline 阶段):
实现自编码器 (AE) 和 LSTM 时序模型的多阶段训练。

【核心流程与原理】
1. 阶段一：训练自编码器 (Autoencoder)。
   - 自编码器对每一帧图像进行独立压缩，不带有任何时间序列的因果推演。
   - 目标是建立一个稳定降维、升维重构的通道，获得静态隐表征能力。
2. 阶段二：训练 LSTM 动力学时序模型。
   - 将自编码器的参数完全冻结。使用自编码器将连续视频帧映射为隐向量序列 z_seq。
   - 利用 LSTM 学习如何从当前的隐状态 z_t 预测下一时刻的隐状态 z_{t+1}。
   - 【注意】在 Step 1 的 Baseline 阶段，动力学模型是在 100% 的 **Teacher Forcing (教师强迫)** 模式下训练的，没有加入计划采样 (Scheduled Sampling)，这可以作为学习该技术演进的最佳对比案例。
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MovingMNISTDataset
from model import Autoencoder, LatentDynamicsModel

# 获取当前脚本所在目录
_DIR = os.path.dirname(os.path.abspath(__file__))


def get_device():
    """检测可用计算设备：Apple Silicon GPU (MPS) -> NVIDIA GPU (CUDA) -> CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_autoencoder(ae, dataloader, device, epochs=20, lr=1e-3, save_path='ae_weights.pth'):
    """
    第一阶段：训练自编码器
    
    参数:
        ae (Autoencoder): 自编码器模型
        dataloader (DataLoader): 数据加载器
        device (torch.device): 计算设备
        epochs (int): 训练周期数
        lr (float): 优化器的学习率
        save_path (str): 模型权重保存路径
    """
    print("--- Starting Phase 1: Training Autoencoder ---")
    ae.to(device)
    # 使用 Adam 优化器
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    # 二值交叉熵损失函数 (BCE Loss)，适合归一化到 [0, 1] 的图像重构
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        # 设为训练模式
        ae.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # batch 原始形状为: (B, seq_len, 1, 64, 64)
            # 自编码器关注的是对每一帧进行单独重构，无时序关联，
            # 因此这里使用 .view 将时序维度 (seq_len) 压入 Batch 维度：形状变形为 (B * seq_len, 1, 64, 64)
            B, seq_len, C, H, W = batch.shape
            x = batch.view(B * seq_len, C, H, W).to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播，x_recon 为重建图像
            x_recon, _ = ae(x)
            # 计算重构误差
            loss = criterion(x_recon, x)
            # 反向传播计算梯度
            loss.backward()
            # 优化权重
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 20 == 0:
                print(f"AE Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"AE Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
        
    # 保存训练好的 AE 权重
    torch.save(ae.state_dict(), save_path)
    print(f"Autoencoder weights saved to {save_path}\n")


def train_dynamics(ae, dyn, dataloader, device, epochs=50, lr=1e-3, save_path='dyn_weights.pth'):
    """
    第二阶段：训练 LSTM 时序动力学模型
    
    参数:
        ae (Autoencoder): 已完成训练的自编码器模型
        dyn (LatentDynamicsModel): 动力学时序模型
        dataloader (DataLoader): 数据加载器
        device (torch.device): 计算设备
        epochs (int): 训练周期数
        lr (float): 优化器学习率
        save_path (str): 模型权重保存路径
    """
    print("--- Starting Phase 2: Training Latent Dynamics Model ---")
    
    # 1. 冻结自编码器：将其推入评估模式 (.eval())，这样可以关闭 Dropout/BatchNorm。
    # 动力学训练不需要更新自编码器的权重。
    ae.to(device)
    ae.eval() 
    
    dyn.to(device)
    optimizer = optim.Adam(dyn.parameters(), lr=lr)
    # 使用均方误差 (MSE Loss) 衡量预测隐向量与真实隐向量之间的空间欧氏距离
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        dyn.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # batch 形状: (B, seq_len, 1, 64, 64)
            batch = batch.to(device)
            B, seq_len, C, H, W = batch.shape
            
            # 步骤一：通过自编码器的 Encoder，将一整段视频时序全部转换/编码为隐向量时序。
            # 包裹在 torch.no_grad() 内以节省计算显存。
            with torch.no_grad():
                # 将时序展平
                x_flat = batch.view(B * seq_len, C, H, W)
                # 获取静态隐向量表示
                z_flat = ae.encoder(x_flat)
                # 重新恢复回时序排列，形状变为: (B, seq_len, latent_dim)
                z_seq = z_flat.view(B, seq_len, -1) # (B, seq_len, latent_dim)
            
            # 步骤二：切分输入与预测目标 (以错开一个时间步)
            # 输入 (z_input) 是 0 到 T-2 时刻的特征: [z_0, z_1, ..., z_{T-2}]，形状为 (B, seq_len-1, latent_dim)
            z_input = z_seq[:, :-1, :]
            # 目标 (z_target) 是 1 到 T-1 时刻的特征: [z_1, z_2, ..., z_{T-1}]，形状为 (B, seq_len-1, latent_dim)
            # 模型需要预测下一帧状态
            z_target = z_seq[:, 1:, :]
            
            # 梯度清零
            optimizer.zero_grad()
            # 前向计算得到全时序的预测输出
            z_pred, _ = dyn(z_input)
            # 计算预测隐状态与真实隐状态的均方误差
            loss = criterion(z_pred, z_target)
            # 反向传播并更新 LSTM 的参数
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 20 == 0:
                print(f"Dynamics Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"Dynamics Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
        
    # 保存训练好的时序动力学模型权重
    torch.save(dyn.state_dict(), save_path)
    print(f"Dynamics weights saved to {save_path}\n")


if __name__ == "__main__":
    # 解析终端可选控制参数，允许修改批大小和 epochs 轮数
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ae_epochs', type=int, default=50)
    parser.add_argument('--dyn_epochs', type=int, default=200)
    args = parser.parse_args()

    # 获取计算设备
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. 载入数据集
    print("Preparing dataset...")
    dataset = MovingMNISTDataset(download=True, train=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. 模型实例化
    ae = Autoencoder(latent_dim=128)
    dyn = LatentDynamicsModel(latent_dim=128, hidden_dim=256)
    
    # 3. 确定模型权重保存位置
    ae_path = os.path.join(_DIR, 'ae_weights.pth')
    dyn_path = os.path.join(_DIR, 'dyn_weights.pth')
    
    # 4. 依次执行两个训练阶段
    train_autoencoder(ae, dataloader, device, epochs=args.ae_epochs, save_path=ae_path)
    train_dynamics(ae, dyn, dataloader, device, epochs=args.dyn_epochs, save_path=dyn_path)
        
    print("Training finished! You can now run inference.py to see the model 'dream'.")

