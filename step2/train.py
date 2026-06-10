"""
主程序入口:
该脚本负责编排并执行“感知 (VAE) + 动力学 (Dynamics)”世界模型的双阶段训练管线。
"""

from torch.utils.data import DataLoader

# 导入自定义的数据集、模型架构、训练配置参数和训练器逻辑
from dataset import MovingMNISTDataset
from model import VAE, DynamicsModel
from config import TrainConfig
from trainer import get_device, train_vae, train_dynamics


def main():
    # 1. 实例化全局训练参数配置对象
    cfg = TrainConfig()
    
    # 2. 自动获取当前系统的最优计算设备 (GPU / MPS / CPU)
    device = get_device()
    print(f"device: {device}")

    # 3. 准备 Moving MNIST 训练数据集
    print("Preparing dataset...")
    # 实例化数据集对象。首次运行若本地无数据，会触发自动下载。
    dataset = MovingMNISTDataset(download=True, train=True)
    
    # 使用 PyTorch 的 DataLoader 将数据集打包成小批量 (Mini-batch) 数据。
    # batch_size 通过配置类传入，shuffle=True 确保每次 epoch 打乱序列，增强模型泛化性。
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # 4. 模型初始化
    # 初始化 Phase 1: 变分自编码器 VAE。负责将 64x64 高维图像空间编解码至 latent_dim 维隐空间。
    vae = VAE(latent_dim=cfg.latent_dim)
    
    # 初始化 Phase 2: 潜空间动力学预测模型。
    # 内部结合了双层 GRU 时序模型、多头自注意力机制和前馈神经网络。
    dyn = DynamicsModel(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_gru_layers=cfg.num_gru_layers,
        dropout=cfg.dropout,
    )

    # 5. 执行 Phase 1: 训练 VAE 模型
    # 训练结束后，VAE 的参数权重会被自动保存，从而得到一个稳定的、可泛化的潜空间表示。
    train_vae(vae, loader, cfg, device)
    
    # 6. 执行 Phase 2: 训练 Dynamics 动力学模型
    # 在这个阶段，已经训练好的 VAE 编码器作为特征提取器处于冻结状态 (eval 模式且不计算梯度)。
    # 动力学模型在 VAE 的潜空间 z 内，通过计划采样机制学习历史视频帧对未来帧演变的规律。
    train_dynamics(vae, dyn, loader, cfg, device)

    # 提示运行推理，通过「梦境自回归」生成时序 GIF 动图
    print("\nDone! Run  python inference.py  to see the model dream.")


if __name__ == "__main__":
    main()

