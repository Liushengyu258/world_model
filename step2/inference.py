"""
推理与可视化模块:
加载训练好的 VAE 和 Dynamics 模型权重，对测试集视频进行“梦境自回归预测 (Model Dreaming)”，并输出左右对比的 GIF 动图。
左侧是环境真实的演变轨迹 (Ground Truth)，右侧是世界模型纯粹依靠大脑在隐空间中模拟/想象的未来演变。
"""

import os

import torch
import numpy as np
import imageio
from torch.utils.data import DataLoader

from dataset import MovingMNISTDataset
from model import VAE, DynamicsModel
from config import TrainConfig
from trainer import get_device


def generate_gif(
    vae: VAE,
    dyn: DynamicsModel,
    loader: DataLoader,
    device: torch.device,
    num_prompt: int = 5,
    total_frames: int = 20,
    save_path: str = "world_model_dream.gif",
):
    """
    自回归预测并生成左右对比的 GIF 动图。
    
    参数:
        vae (VAE): 已训练完的 VAE 模块
        dyn (DynamicsModel): 已训练完的动力学预测模块
        loader (DataLoader): 测试集数据加载器 (Batch Size = 1)
        device (torch.device): 计算设备
        num_prompt (int): 提示帧数。前 num_prompt 帧会给模型看真实视频，用以“建立隐层时序状态”；
                          从第 num_prompt 帧之后，模型将处于自残运行（自由想象）模式，不给其看任何真实状态。
        total_frames (int): 视频的总帧数 (默认为 20)
        save_path (str): 导出的 GIF 文件路径
    """
    # 将模型都设为评估模式 (.eval())：这会关闭 Dropout，并稳定 BatchNorm 的均值和方差
    vae.to(device).eval()
    dyn.to(device).eval()

    # 1. 随机获取测试集中的一个视频序列。
    # 原始 batch 为 (1, T, 1, 64, 64) -> 维度分别为: (Batch=1, Time_Steps=20, Channel=1, Height=64, Width=64)
    seq = next(iter(loader))[0].unsqueeze(0).to(device)   # (1, T, 1, 64, 64)
    
    # 2. 提取出真实视频 (Ground Truth) 作为对比。
    # seq[0] 形状为 (20, 1, 64, 64)，.squeeze(1) 移除通道维度，变为 (20, 64, 64) 的 Numpy 数组
    gt = seq[0].cpu().numpy().squeeze(1)                    # (T, 64, 64)
    predicted = []

    # 3. 开启无梯度预测
    with torch.no_grad():
        # A. 编码所有帧：将真实的 20 帧图像送入 VAE Encoder，得到每一帧对应的隐空间向量
        # z_all 形状为: (1, T, Latent_Dim)
        z_all = torch.stack([vae.encode(seq[:, t]) for t in range(total_frames)], dim=1)

        # B. Prompt 提示阶段 (第 0 到第 num_prompt - 1 帧)：
        # 在提示期，我们将真实的隐向量送入 VAE Decoder 重构。
        # 目的是向学习者展示模型在有观测时的重建质量。
        for t in range(num_prompt):
            # 将隐向量 z_all[:, t] 输入解码器解码回图像，提取出 [0, 0] 表示 Batch 内第0个样本，且单通道灰度。形状: (64, 64)
            predicted.append(vae.decoder(z_all[:, t])[0, 0].cpu().numpy())

        # C. 自回归热身 (Warm-up) 与预测起点：
        # 第一步：我们将前 num_prompt 帧的真实隐向量 z_all[:, :num_prompt] 一次性喂给 Dynamics 模型。
        # 这时 GRU 经历了连续 5 步状态传递，其内部积累了数字当下移动的方向、速度等“物理状态”，储存在返回的 hidden 中。
        _, hidden = dyn(z_all[:, :num_prompt])
        
        # 第二步：将提示帧的最后一帧 z_all[:, num_prompt-1:num_prompt] (即第 4 帧) 作为输入，配合刚建好的时序记忆状态 hidden。
        # 动力学模型预测出第 5 帧（自回归的第一步预测）对应的隐向量 z_next，并更新 hidden
        z_next, hidden = dyn(z_all[:, num_prompt - 1:num_prompt], hidden)
        
        # 第三步：将预测出的 z_next 解码成第一张“想象”的图像帧并追加到列表中
        predicted.append(vae.decoder(z_next[:, 0])[0, 0].cpu().numpy())

        # D. 完全自回归自由运行阶段 (从第 num_prompt + 1 帧到 total_frames-1 帧)：
        # 此时，世界模型已经不能再接触到外界真实的 z_all 观测了。
        # 它的输入必须永远是它自己前一步预测出来的 z_next（自食其力）。
        for _ in range(num_prompt + 1, total_frames):
            # 用上一步自己预测的隐状态 z_next 和累积的记忆 hidden，迭代预测未来的隐向量，并持续更新 hidden 状态
            z_next, hidden = dyn(z_next, hidden)
            # 将这一步模型靠大脑想象出的隐向量，使用 VAE 解码器重构成可视化图像
            predicted.append(vae.decoder(z_next[:, 0])[0, 0].cpu().numpy())

    # 4. 图像合并与保存：
    # 遍历时序上的每一帧，将真实的图像 (gt[t]) 与预测图像 (predicted[t]) 在宽度方向 (axis=1) 上横向拼接
    # gt 和 predicted 均乘以 255 转换为 uint8 的 0~255 像素值，以便保存为标准 GIF。
    frames = [
        np.concatenate([(gt[t] * 255).astype(np.uint8), (predicted[t] * 255).astype(np.uint8)], axis=1)
        for t in range(total_frames)
    ]
    
    # 将 20 帧拼接图保存为 GIF 动图，帧率 fps=5 (每秒播放 5 帧)
    imageio.mimsave(save_path, frames, fps=5)
    print(f"Saved → {save_path}")
    print(f"Left: Ground Truth | Right: Model Dream  (first {num_prompt} frames as prompt)")


def main():
    # 载入配置和计算设备
    cfg = TrainConfig()
    device = get_device()

    # 载入测试集 (train=False)
    dataset = MovingMNISTDataset(download=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 模型实例化
    vae = VAE(latent_dim=cfg.latent_dim)
    dyn = DynamicsModel(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim)

    # 检查本地是否已存在训练完成的权重，若不存在则提示需先训练
    if not (os.path.exists(cfg.vae_weights) and os.path.exists(cfg.dyn_weights)):
        raise FileNotFoundError("Weights not found — run train.py first.")

    # 载入训练权重 (使用 map_location 确保在没有 GPU 的机器上也能顺利在 CPU 加载)
    vae.load_state_dict(torch.load(cfg.vae_weights, map_location=device, weights_only=True))
    dyn.load_state_dict(torch.load(cfg.dyn_weights, map_location=device, weights_only=True))
    
    # 自回归生成 GIF 动图
    generate_gif(vae, dyn, loader, device, save_path=cfg.gif_path)


if __name__ == "__main__":
    main()

