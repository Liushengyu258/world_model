"""
推理与可视化模块 (Baseline 阶段):
加载训练完成的普通自编码器 (AE) 和 LSTM 动力学模型，自回归地“想象/预测”未来的视频帧，并保存为左右对比的 GIF 动图。
"""

import os
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from dataset import MovingMNISTDataset
from torch.utils.data import DataLoader
from model import Autoencoder, LatentDynamicsModel
from train import get_device

# 获取当前脚本所在目录
_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_gif(ae, dyn, dataloader, device, num_prompt_frames=5, total_frames=20, save_path=os.path.join(_DIR, "world_model_dream.gif")):
    """
    加载模型并生成自回归推演 GIF 动图
    
    参数:
        ae (Autoencoder): 已训练的自编码器
        dyn (LatentDynamicsModel): 已训练的 LSTM 动力学时序模型
        dataloader (DataLoader): 测试集数据加载器 (Batch Size = 1)
        device (torch.device): 计算设备
        num_prompt_frames (int): 提示帧数，用于初始化/热身时序记忆模型
        total_frames (int): 时序总帧数 (Moving MNIST 默认为 20 帧)
        save_path (str): GIF 保存路径
    """
    print("Generating predictions...")
    ae.to(device)
    dyn.to(device)
    ae.eval()
    dyn.eval()
    
    # 1. 从测试数据加载器中抓取一个 Batch。
    # 形状为 (Batch=1, Time_Steps=20, Channel=1, Height=64, Width=64)
    batch = next(iter(dataloader)) 
    # 提取第 0 个样本，保留 Batch 维度方便送入网络，形状变为 (1, 20, 1, 64, 64)
    seq = batch[0].unsqueeze(0).to(device) 
    
    # 2. 提取出真实帧，转换成 (T=20, 64, 64) 的 NumPy 数组，作为可视化左侧的参考对比 (Ground Truth)
    gt_frames = seq[0].cpu().numpy() # (20, 1, 64, 64)
    gt_frames = gt_frames.squeeze(1) # (20, 64, 64)
    
    predicted_frames = []
    
    # 3. 开始预测，关闭梯度计算
    with torch.no_grad():
        # A. 提取全部 20 帧真实的隐向量 z
        z_seq_gt = []
        for t in range(total_frames):
            frame = seq[:, t, :, :, :]
            z = ae.encoder(frame)
            z_seq_gt.append(z)
            
        # 拼接成一个形状为 (1, 20, latent_dim) 的张量
        z_seq_gt = torch.stack(z_seq_gt, dim=1) 
        
        # B. 提示期 (Prompt)：直接使用前 num_prompt_frames 帧的真实 z。
        # 这一阶段不进行动力学预测，直接用自编码器解码重建。
        current_z_seq = z_seq_gt[:, :num_prompt_frames, :]
        
        # 用自编码器的解码器恢复这 5 张图，并存入预测结果列表
        for t in range(num_prompt_frames):
            frame_recon = ae.decoder(current_z_seq[:, t, :])
            predicted_frames.append(frame_recon[0, 0].cpu().numpy())
            
        # C. 动力学时序热身 (Warm-up)：
        # 将前 5 帧真实隐向量 [z_0, z_1, z_2, z_3, z_4] 送入 LSTM，
        # 在内部让两个隐藏状态 (h, c) 初始化并累计关于物体轨迹移动、碰撞等必要的物理记忆。
        # h, c 形状为 (num_layers=2, Batch=1, hidden_dim=256)
        _, (h, c) = dyn.lstm(current_z_seq)
        
        # D. 预测首帧未来（第 6 帧，即索引 5）：
        # 在热身完毕后，获取 LSTM 在最后一个时间步 (t=4) 时全连接层映射出的下一时刻预测隐向量 z_next
        lstm_out, _ = dyn.lstm(current_z_seq)
        z_next = dyn.fc(lstm_out[:, -1:, :]) # 提取最后一个时间步对应的输出，形状: (1, 1, latent_dim)
        
        # 解码并记录首帧预测图
        frame_recon = ae.decoder(z_next[:, 0, :])
        predicted_frames.append(frame_recon[0, 0].cpu().numpy())
        
        # E. 自回归循环想象阶段 (Autoregressive Dreaming)：
        # 从第 7 帧开始，我们不再给模型看真实的 z。
        # 此时，模型的输入 `current_z_input` 永远是其在上一阶段预测出来的隐向量。
        current_z_input = z_next
        
        for t in range(num_prompt_frames + 1, total_frames):
            # 将上一步的预测 current_z_input 送入 LSTM，并传入更新后的时序记忆 (h, c)
            lstm_out, (h, c) = dyn.lstm(current_z_input, (h, c))
            # 投影输出当前预测未来的隐状态 z_next
            z_next = dyn.fc(lstm_out)
            
            # 使用自编码器 Decoder 解码为 64x64 重构图像
            frame_recon = ae.decoder(z_next[:, 0, :])
            predicted_frames.append(frame_recon[0, 0].cpu().numpy())
            
            # 滚雪球式更新：将当前步预测出的 z_next 作为下一帧预测的输入
            current_z_input = z_next
            
    # 4. GIF 时序拼接保存
    frames_to_save = []
    for t in range(total_frames):
        # 左右两侧拼接对比：左侧真实 (gt_img)，右侧自回归想象 (pred_img)
        gt_img = (gt_frames[t] * 255).astype(np.uint8)
        pred_img = (predicted_frames[t] * 255).astype(np.uint8)
        
        # 横向级联 (concatenate)，拼接为 64x128 宽屏图像
        combined = np.concatenate([gt_img, pred_img], axis=1) # (64, 128)
        frames_to_save.append(combined)
        
    # 保存时序图片流为 5 fps 的 GIF 动图
    imageio.mimsave(save_path, frames_to_save, fps=5)
    print(f"Dream saved to {save_path}!")
    print("Left: Ground Truth | Right: Model Dream")
    print(f"(First {num_prompt_frames} frames are given as prompt)")


if __name__ == "__main__":
    # 检测计算设备
    device = get_device()
    
    # 载入测试数据 (train=False)
    print("Loading test dataset...")
    dataset = MovingMNISTDataset(download=True, train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 模型实例化
    ae = Autoencoder(latent_dim=128)
    dyn = LatentDynamicsModel(latent_dim=128, hidden_dim=256)
    
    # 获取本地权重存储路径
    ae_path = os.path.join(_DIR, 'ae_weights.pth')
    dyn_path = os.path.join(_DIR, 'dyn_weights.pth')
    
    # 检查模型是否已加载训练权重
    if not os.path.exists(ae_path) or not os.path.exists(dyn_path):
        print("Model weights not found! Please run train.py first.")
    else:
        ae.load_state_dict(torch.load(ae_path, map_location=device, weights_only=True))
        dyn.load_state_dict(torch.load(dyn_path, map_location=device, weights_only=True))
        # 执行自回归预测并生成对比动图
        generate_gif(ae, dyn, dataloader, device)

