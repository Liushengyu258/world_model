"""入口: 训练 VAE + Dynamics 世界模型"""

from torch.utils.data import DataLoader

from dataset import MovingMNISTDataset
from model import VAE, DynamicsModel
from config import TrainConfig
from trainer import get_device, train_vae, train_dynamics


def main():
    cfg = TrainConfig()
    device = get_device()
    print(f"device: {device}")

    # 数据
    print("Preparing dataset...")
    dataset = MovingMNISTDataset(download=True, train=True)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # 模型
    vae = VAE(latent_dim=cfg.latent_dim)
    dyn = DynamicsModel(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_gru_layers=cfg.num_gru_layers,
        dropout=cfg.dropout,
    )

    # 训练
    train_vae(vae, loader, cfg, device)
    train_dynamics(vae, dyn, loader, cfg, device)

    print("\nDone! Run  python inference.py  to see the model dream.")


if __name__ == "__main__":
    main()
