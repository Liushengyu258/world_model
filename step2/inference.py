"""入口: 加载权重，生成世界模型「梦境」GIF"""

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
    vae.to(device).eval()
    dyn.to(device).eval()

    seq = next(iter(loader))[0].unsqueeze(0).to(device)   # (1, T, 1, 64, 64)
    gt = seq[0].cpu().numpy().squeeze(1)                    # (T, 64, 64)
    predicted = []

    with torch.no_grad():
        # 编码所有帧
        z_all = torch.stack([vae.encode(seq[:, t]) for t in range(total_frames)], dim=1)

        # Prompt: 解码前几帧
        for t in range(num_prompt):
            predicted.append(vae.decoder(z_all[:, t])[0, 0].cpu().numpy())

        # 自回归: 建立隐状态后逐步预测
        _, hidden = dyn(z_all[:, :num_prompt])
        z_next, hidden = dyn(z_all[:, num_prompt - 1:num_prompt], hidden)
        predicted.append(vae.decoder(z_next[:, 0])[0, 0].cpu().numpy())

        for _ in range(num_prompt + 1, total_frames):
            z_next, hidden = dyn(z_next, hidden)
            predicted.append(vae.decoder(z_next[:, 0])[0, 0].cpu().numpy())

    # 拼接 GT | Pred 并保存
    frames = [
        np.concatenate([(gt[t] * 255).astype(np.uint8), (predicted[t] * 255).astype(np.uint8)], axis=1)
        for t in range(total_frames)
    ]
    imageio.mimsave(save_path, frames, fps=5)
    print(f"Saved → {save_path}")
    print(f"Left: Ground Truth | Right: Model Dream  (first {num_prompt} frames as prompt)")


def main():
    cfg = TrainConfig()
    device = get_device()

    dataset = MovingMNISTDataset(download=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    vae = VAE(latent_dim=cfg.latent_dim)
    dyn = DynamicsModel(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim)

    if not (os.path.exists(cfg.vae_weights) and os.path.exists(cfg.dyn_weights)):
        raise FileNotFoundError("Weights not found — run train.py first.")

    vae.load_state_dict(torch.load(cfg.vae_weights, map_location=device, weights_only=True))
    dyn.load_state_dict(torch.load(cfg.dyn_weights, map_location=device, weights_only=True))
    generate_gif(vae, dyn, loader, device, save_path=cfg.gif_path)


if __name__ == "__main__":
    main()
