"""训练器: 封装 VAE 和 Dynamics 的训练逻辑"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import VAE, DynamicsModel
from config import TrainConfig


# ─── 工具 ─────────────────────────────────────────────


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── Phase 1: VAE ─────────────────────────────────────


def train_vae(vae: VAE, loader: DataLoader, cfg: TrainConfig, device: torch.device):
    """训练 VAE，使用 KL 退火策略"""
    print("\n" + "=" * 50)
    print(" Phase 1: Training VAE")
    print("=" * 50)

    vae.to(device)
    opt = optim.Adam(vae.parameters(), lr=cfg.vae_lr)

    for epoch in range(1, cfg.vae_epochs + 1):
        vae.train()
        accum = {"loss": 0.0, "bce": 0.0, "kl": 0.0}
        kl_w = min(1.0, (epoch - 1) / max(1, cfg.vae_epochs * cfg.kl_anneal_ratio))

        for batch_idx, batch in enumerate(loader):
            x = batch.flatten(0, 1).to(device)           # (B*T, C, H, W)

            opt.zero_grad()
            x_recon, mu, logvar = vae(x)
            loss, bce, kl = vae.loss_function(x_recon, x, mu, logvar, kl_weight=kl_w)
            loss.backward()
            opt.step()

            for k, v in zip(("loss", "bce", "kl"), (loss, bce, kl)):
                accum[k] += v.item()

            if batch_idx % 20 == 0:
                print(f"  [{epoch}/{cfg.vae_epochs}] batch {batch_idx:>3d}/{len(loader)}  "
                      f"loss={loss.item():.1f}  bce={bce.item():.1f}  kl={kl.item():.1f}  kl_w={kl_w:.3f}")

        n = len(loader)
        print(f"  [{epoch}/{cfg.vae_epochs}] avg  "
              f"loss={accum['loss']/n:.1f}  bce={accum['bce']/n:.1f}  kl={accum['kl']/n:.1f}")

    torch.save(vae.state_dict(), cfg.vae_weights)
    print(f"  ✓ saved → {cfg.vae_weights}")


# ─── Phase 2: Dynamics ─────────────────────────────────


def train_dynamics(vae: VAE, dyn: DynamicsModel, loader: DataLoader, cfg: TrainConfig, device: torch.device):
    """训练动力学模型，Scheduled Sampling 混合策略"""
    print("\n" + "=" * 50)
    print(" Phase 2: Training Dynamics")
    print("=" * 50)

    vae.to(device).eval()
    dyn.to(device)
    opt = optim.Adam(dyn.parameters(), lr=cfg.dyn_lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.dyn_epochs)
    criterion = nn.MSELoss()

    for epoch in range(1, cfg.dyn_epochs + 1):
        dyn.train()
        total_loss = 0.0
        tf_ratio = max(0.0, 1.0 - (epoch - 1) / max(1, cfg.dyn_epochs * cfg.tf_decay_ratio))

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            B, T, C, H, W = batch.shape

            # 编码整段序列
            with torch.no_grad():
                z_seq = vae.encode(batch.flatten(0, 1)).view(B, T, -1)

            z_input = z_seq[:, :-1, :]
            z_target = z_seq[:, 1:, :]

            opt.zero_grad()
            loss = _dynamics_step(dyn, z_input, z_target, tf_ratio, criterion)
            loss.backward()
            nn.utils.clip_grad_norm_(dyn.parameters(), cfg.grad_clip)
            opt.step()

            total_loss += loss.item()
            if batch_idx % 20 == 0:
                print(f"  [{epoch}/{cfg.dyn_epochs}] batch {batch_idx:>3d}/{len(loader)}  "
                      f"loss={loss.item():.4f}  tf={tf_ratio:.2f}")

        sched.step()
        print(f"  [{epoch}/{cfg.dyn_epochs}] avg loss={total_loss/len(loader):.4f}  "
              f"lr={sched.get_last_lr()[0]:.6f}")

    torch.save(dyn.state_dict(), cfg.dyn_weights)
    print(f"  ✓ saved → {cfg.dyn_weights}")


def _dynamics_step(dyn, z_input, z_target, tf_ratio, criterion):
    """单步训练: teacher forcing 或自由运行"""
    if tf_ratio > 0.5 or torch.rand(1).item() < tf_ratio:
        # teacher forcing
        z_pred, _ = dyn(z_input)
        return criterion(z_pred, z_target)

    # 自由运行: 逐步预测，用自身输出作为下一步输入
    preds, hidden = [], None
    z_step = z_input[:, 0:1, :]

    for t in range(z_input.size(1)):
        z_pred, hidden = dyn(z_step, hidden)
        preds.append(z_pred)
        z_step = z_input[:, t + 1:t + 2, :] if (t + 1 < z_input.size(1) and torch.rand(1).item() < tf_ratio) else z_pred

    return criterion(torch.cat(preds, dim=1), z_target)
