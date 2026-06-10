"""
训练器: 封装 VAE (变分自编码器) 和 Dynamics (世界模型的动力学/预测模型) 的训练逻辑。
该模块是构建 Model-Based RL (基于模型的强化学习) 或世界模型 (World Model) 的核心训练流程。

包含两个阶段的训练：
Phase 1: 训练 VAE。将高维的图像观测（如 64x64）压缩到低维的隐空间 (Latent Space)，学习一个紧凑的特征表示。
Phase 2: 训练动力学模型。在 VAE 提取的隐特征之上，利用循环神经网络 (RNN/GRU) 预测未来状态的演变。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import VAE, DynamicsModel
from config import TrainConfig


# ─── 工具函数 ─────────────────────────────────────────────


def get_device() -> torch.device:
    """
    自动检测并获取当前系统最快的可用计算设备。
    优先级：Apple Silicon GPU (MPS) -> Nvidia GPU (CUDA) -> 处理器 (CPU)
    
    返回:
        torch.device: 选定的计算设备对象
    """
    # 检查是否为苹果的 Apple Silicon M1/M2/M3 等芯片，若是则使用 MPS 加速
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # 检查是否有 NVIDIA 显卡支持，若是则使用 CUDA 加速
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 默认回退到 CPU
    return torch.device("cpu")


# ─── Phase 1: VAE 训练阶段 ──────────────────────────────


def train_vae(vae: VAE, loader: DataLoader, cfg: TrainConfig, device: torch.device):
    """
    训练 VAE 模型，使用 KL 散度退火 (KL Annealing) 策略。
    
    VAE 训练的核心在于平衡两个损失：
    1. 重构误差 (Reconstruction Loss / BCE): 迫使 VAE 能够精细还原输入图像。
    2. KL 散度误差 (KL Divergence Loss): 迫使隐空间服从标准正态分布 N(0, I)。
    
    在训练初期，直接引入高权重的 KL 散度会导致“隐空间坍塌 (Latent Collapse)”，
    即模型为了将 KL 降低到 0 而忽略了重构，导致隐变量不包含任何有效信息。
    KL 退火策略在训练前期将 KL 权重设为 0，随着 epoch 增加线性递增到 1.0，
    这使得模型先学会“如何高质量重构图像”，再逐步整理隐空间，使其平滑并服从标准分布。
    
    参数:
        vae (VAE): 待训练的变分自编码器模型
        loader (DataLoader): PyTorch 数据加载器，输出 Shape 为 (B, T, C, H, W)
        cfg (TrainConfig): 训练超参数配置对象
        device (torch.device): 运行计算的设备 (CPU/CUDA/MPS)
    """
    print("\n" + "=" * 50)
    print(" Phase 1: Training VAE")
    print("=" * 50)

    # 将 VAE 模型移动到指定的计算设备 (例如 GPU)
    vae.to(device)
    # 使用 Adam 优化器更新 VAE 参数，设置学习率为配置中的 vae_lr
    opt = optim.Adam(vae.parameters(), lr=cfg.vae_lr)

    # 循环遍历每一个训练周期 (Epoch)
    for epoch in range(1, cfg.vae_epochs + 1):
        # 将模型设置为训练模式 (激活 Dropout, BatchNorm 等)
        vae.train()
        # 初始化累加器，用于统计当前 Epoch 的各项平均损失
        accum = {"loss": 0.0, "bce": 0.0, "kl": 0.0}
        
        # 计算当前 Epoch 的 KL 散度权重系数 (kl_w)
        # kl_w 从 0 开始，随着 epoch 增长线性增加，直到在特定的 epoch (由 kl_anneal_ratio 决定) 达到最大值 1.0
        # 公式: kl_w = min(1.0, 当前步数 / 退火结束步数)
        kl_w = min(1.0, (epoch - 1) / max(1, cfg.vae_epochs * cfg.kl_anneal_ratio))

        # 遍历数据集的每个 Batch
        for batch_idx, batch in enumerate(loader):
            # batch 原始形状为 (B, T, C, H, W)，其中 B 是 Batch 大小，T 是时序长度。
            # VAE 在处理单帧图像时，在时间维度上是相互独立的。
            # 为了提高并行计算效率，我们使用 .flatten(0, 1) 将第 0 维 (B) 和第 1 维 (T) 合并为一维 (B*T)。
            # 变形后的 x 的 Shape 为: (B*T, C, H, W)
            x = batch.flatten(0, 1).to(device)           # (B*T, C, H, W)

            # 梯度清零，防止上一次迭代的梯度累加影响本次更新
            opt.zero_grad()
            
            # 前向传播：将图像输入 VAE，得到重构图像 x_recon，隐空间均值 mu，以及对数方差 logvar
            # x_recon 形状: (B*T, C, H, W); mu, logvar 形状: (B*T, Latent_Dim)
            x_recon, mu, logvar = vae(x)
            
            # 计算 VAE 的损失函数。包含：
            # loss: 最终总损失 (bce + kl_w * kl)
            # bce: 二值交叉熵重构误差（或 MSE），衡量重构图像与原图的相似度
            # kl: KL 散度，衡量隐分布与标准正态分布的差异
            loss, bce, kl = vae.loss_function(x_recon, x, mu, logvar, kl_weight=kl_w)
            
            # 反向传播：计算总损失相对于模型参数的梯度
            loss.backward()
            
            # 优化器步骤：根据计算出的梯度，更新 VAE 所有的权重和偏置
            opt.step()

            # 记录并累加当前 Batch 的各项损失值 (转换为 Python 标量数值)
            for k, v in zip(("loss", "bce", "kl"), (loss, bce, kl)):
                accum[k] += v.item()

            # 每隔 20 个 Batch 打印一次当前的训练状态，展示损失下降趋势和当前 KL 权重
            if batch_idx % 20 == 0:
                print(f"  [{epoch}/{cfg.vae_epochs}] batch {batch_idx:>3d}/{len(loader)}  "
                      f"loss={loss.item():.1f}  bce={bce.item():.1f}  kl={kl.item():.1f}  kl_w={kl_w:.3f}")

        # 计算并打印当前 Epoch 的平均损失
        n = len(loader)
        print(f"  [{epoch}/{cfg.vae_epochs}] avg  "
              f"loss={accum['loss']/n:.1f}  bce={accum['bce']/n:.1f}  kl={accum['kl']/n:.1f}")

    # 训练完成后，将训练好的 VAE 权重参数保存到磁盘，以便后续动力学模型训练或评估时直接加载
    torch.save(vae.state_dict(), cfg.vae_weights)
    print(f"  ✓ saved → {cfg.vae_weights}")


# ─── Phase 2: Dynamics 动力学模型训练阶段 ─────────────────


def train_dynamics(vae: VAE, dyn: DynamicsModel, loader: DataLoader, cfg: TrainConfig, device: torch.device):
    """
    训练动力学模型，采用计划采样 (Scheduled Sampling) 的混合训练策略。
    
    动力学模型 (Dynamics Model) 的任务是输入当前的隐状态 z_t，预测下一个时刻的隐状态 z_{t+1}。
    在真实应用/测试时，动力学模型需要进行多步自我循环预测 (Autoregressive Free-Running)：
    即用自己预测的 \\hat{z}_{t+1} 作为下一步的输入去预测 \\hat{z}_{t+2}。
    
    然而，如果仅用“自由运行 (Free-Running)”模式训练，在训练初期模型预测极不准确，误差会迅速累积，导致模型根本无法收敛。
    如果仅用“教师强迫 (Teacher Forcing)”模式训练（即每一步的输入永远是真实观测的 z_t），
    会导致“曝光偏差 (Exposure Bias)”问题：模型在训练时从未见过自己预测的错误输入，一旦在测试时预测错了一步，
    就会因为进入未知的输入空间而引发雪崩式的误差。
    
    计划采样 (Scheduled Sampling) 完美地结合了两者：
    在训练初期，tf_ratio (Teacher Forcing 比例) 接近 1.0，几乎完全使用真实值作为输入，让模型快速学会基本规律。
    随着 epoch 增加，tf_ratio 逐渐减小到 0.0，强迫模型越来越多地使用自己前一步生成的预测值作为下一步的输入。
    这培养了模型自我纠偏的能力，极大地提升了长期预测的稳定性。
    
    参数:
        vae (VAE): 已训练完成并冻结的变分自编码器，用于将图像编码为隐向量
        dyn (DynamicsModel): 待训练的动力学 (世界) 模型，一般基于循环网络 (RNN/GRU/LSTM)
        loader (DataLoader): PyTorch 数据加载器
        cfg (TrainConfig): 训练超参数配置对象
        device (torch.device): 运行计算的设备 (CPU/CUDA/MPS)
    """
    print("\n" + "=" * 50)
    print(" Phase 2: Training Dynamics")
    print("=" * 50)

    # 1. 冻结 VAE 的权重：由于 VAE 已经在第一阶段训练完成，我们将其移至设备并设为评估模式 (.eval())。
    # 这样可以关闭 VAE 的 Dropout 并且不更新其 BatchNorm 统计量，节约显存和计算资源。
    vae.to(device).eval()
    
    # 2. 将动力学模型移动到计算设备并设为训练模式。
    dyn.to(device)
    
    # 3. 定义优化器与余弦退火学习率调度器。
    # CosineAnnealingLR 会让学习率按照余弦函数曲线从初始 lr 逐渐衰减到 0（或设定的最小值），这有助于模型在后期平稳收敛到更优的局部极小值。
    opt = optim.Adam(dyn.parameters(), lr=cfg.dyn_lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.dyn_epochs)
    
    # 4. 定义损失函数：使用均方误差 (MSE) 来衡量动力学模型预测的隐向量与 VAE 编码得到的真实隐向量之间的距离。
    criterion = nn.MSELoss()

    # 循环遍历每一个训练周期 (Epoch)
    for epoch in range(1, cfg.dyn_epochs + 1):
        # 激活动力学模型的训练模式
        dyn.train()
        total_loss = 0.0
        
        # 计算当前 Epoch 的 Teacher Forcing 比例 (tf_ratio)。
        # tf_ratio 随训练进程线性递减。在指定的 tf_decay_ratio (例如前 50% 周期) 之后衰减到 0.0，完全转为自由运行/自主预测。
        tf_ratio = max(0.0, 1.0 - (epoch - 1) / max(1, cfg.dyn_epochs * cfg.tf_decay_ratio))

        # 遍历数据集的每个 Batch
        for batch_idx, batch in enumerate(loader):
            # batch 形状: (B, T, C, H, W)，表示 B 个视频序列，每个序列长度为 T，每帧图像通道为 C，高 H 宽 W
            batch = batch.to(device)
            B, T, C, H, W = batch.shape

            # 编码整段序列：
            # 动力学模型是在隐空间进行时序预测的。所以我们需要先将所有的图像帧通过 VAE 的编码器 (Encoder) 转换为隐向量。
            # 使用 torch.no_grad() 确保这一步不计算梯度，不更新 VAE。
            # .flatten(0, 1) 的形状为 (B*T, C, H, W)，送入 vae.encode() 后得到 (B*T, Latent_Dim) 的隐向量表示。
            # 接着通过 .view(B, T, -1) 将其重新整理回时序格式，形状变为: (B, T, Latent_Dim)
            with torch.no_grad():
                z_seq = vae.encode(batch.flatten(0, 1)).view(B, T, -1)

            # 提取动力学模型的输入与预测目标：
            # 对于一个长度为 T 的序列 [z_0, z_1, ..., z_{T-1}]:
            # z_input 是从第 0 帧到倒数第二帧 [z_0, z_1, ..., z_{T-2}]，形状: (B, T-1, Latent_Dim)
            z_input = z_seq[:, :-1, :]
            # z_target 是从第 1 帧到最后一帧 [z_1, z_2, ..., z_{T-1}]，形状: (B, T-1, Latent_Dim)
            # 模型的目标就是根据前一帧预测后一帧。
            z_target = z_seq[:, 1:, :]

            # 梯度清零
            opt.zero_grad()
            
            # 调用内部辅助函数执行单步动力学计算，获取当前 Batch 在混合策略下的 MSE 损失值
            loss = _dynamics_step(dyn, z_input, z_target, tf_ratio, criterion)
            
            # 反向传播计算动力学模型参数的梯度
            loss.backward()
            
            # 梯度裁剪 (Gradient Clipping)：
            # 循环神经网络 (RNN/GRU/LSTM) 随时间步反向传播 (BPTT) 时，极易产生梯度爆炸 (Gradient Explosion)。
            # nn.utils.clip_grad_norm_ 将梯度的 L2 范数裁剪到 cfg.grad_clip 设定值以内，有效保证网络更新的数值稳定性。
            nn.utils.clip_grad_norm_(dyn.parameters(), cfg.grad_clip)
            
            # 更新动力学模型参数
            opt.step()

            # 累加损失值
            total_loss += loss.item()
            
            # 每隔 20 个 Batch 打印训练指标
            if batch_idx % 20 == 0:
                print(f"  [{epoch}/{cfg.dyn_epochs}] batch {batch_idx:>3d}/{len(loader)}  "
                      f"loss={loss.item():.4f}  tf={tf_ratio:.2f}")

        # 更新学习率调度器，根据余弦曲线衰减学习率
        sched.step()
        
        # 打印当前 Epoch 的平均损失和当前的学习率
        print(f"  [{epoch}/{cfg.dyn_epochs}] avg loss={total_loss/len(loader):.4f}  "
              f"lr={sched.get_last_lr()[0]:.6f}")

    # 保存训练好的动力学模型参数
    torch.save(dyn.state_dict(), cfg.dyn_weights)
    print(f"  ✓ saved → {cfg.dyn_weights}")


# ─── 动力学内部辅助函数 ────────────────────────────────────


def _dynamics_step(dyn, z_input, z_target, tf_ratio, criterion):
    """
    单步动力学训练辅助函数，根据 tf_ratio (计划采样比例) 决定是采用 Teacher Forcing 还是 Free Running 模式。
    
    参数:
        dyn (DynamicsModel): 动力学模型，其 forward 接收输入和隐状态，输出预测值和更新后的隐状态
        z_input (Tensor): 真实的动力学模型隐状态输入，形状为 (B, T-1, Latent_Dim)
        z_target (Tensor): 对应的真实下一步隐状态目标，形状为 (B, T-1, Latent_Dim)
        tf_ratio (float): 当前 Epoch 的 Teacher Forcing 概率 (0.0 到 1.0)
        criterion (Loss Function): 用于计算误差的损失函数 (通常为 MSE)
        
    返回:
        Tensor: 计算得到的标量损失 (Loss)
    """
    # 决策：是否使用完全的 Teacher Forcing 模式？
    # 1) 如果 tf_ratio > 0.5，为了训练初期的超强数值稳定性，强制执行 Teacher Forcing。
    # 2) 或者随机生成的浮点数小于当前的 tf_ratio 概率值，也进入 Teacher Forcing 流程。
    if tf_ratio > 0.5 or torch.rand(1).item() < tf_ratio:
        # 【模式一：Teacher Forcing (教师强迫)】
        # 在该模式下，我们将完整的输入时序 z_input (B, T-1, Latent_Dim) 直接一次性输入到动力学模型中。
        # 动力学模型在内部会高效地、并行地通过 GRU/RNN 依次处理每个时间步。
        # 在处理到第 t 步时，网络接收的绝对是真实图像编码出来的真实状态 z_t，不受之前步骤预测好坏的影响。
        # z_pred 形状: (B, T-1, Latent_Dim)；后面的占位符 _ 代表丢弃的最终循环状态 (Hidden State)
        z_pred, _ = dyn(z_input)
        # 计算整条预测时序与真实目标 z_target 的平均 MSE 误差
        return criterion(z_pred, z_target)

    # 【模式二：Free Running (自由运行 / 计划采样自回归)】
    # 在该模式下，我们不能一次性喂入整条时序，而是必须像推理一样，沿着时间步进行循环预测。
    # 因为在每个步骤，我们都需要根据一定概率使用模型自己生成的预测结果作为下一步的输入。
    
    preds, hidden = [], None
    # 提取第 0 时刻的真实隐状态作为循环的起点。
    # 它的 Shape 是 (B, 1, Latent_Dim)，保留了时间维度 1，便于送入动力学模型
    z_step = z_input[:, 0:1, :]

    # 遍历时序中的每一个时间步 (长度为 T-1)
    for t in range(z_input.size(1)):
        # 将当前的输入状态 z_step 和上一步的循环隐状态 hidden 送入动力学网络。
        # 得到对下一个时刻的预测 z_pred: (B, 1, Latent_Dim) 和最新的隐藏状态 hidden
        z_pred, hidden = dyn(z_step, hidden)
        # 将当前的预测值保存到列表中
        preds.append(z_pred)
        
        # 计划采样 (Scheduled Sampling) 的核心逻辑：更新下一步的输入 z_step。
        # 如果还没到最后一个时间步 (t + 1 < z_input.size(1))，且随机出的概率小于当前的 tf_ratio：
        #    - 我们仍会“喂一口蜜糖”，给模型提供真实的下一时刻状态作为输入：z_input[:, t+1:t+2, :]
        # 否则 (当概率大于等于 tf_ratio，或者到达最后一步时)：
        #    - 模型必须“自食其力”，使用自己刚刚预测出来的 z_pred 作为下一步的输入
        z_step = z_input[:, t + 1:t + 2, :] if (t + 1 < z_input.size(1) and torch.rand(1).item() < tf_ratio) else z_pred

    # 将列表中各个时间步的预测结果 preds (共 T-1 个 (B, 1, Latent_Dim)) 在时间维度 (dim=1) 上拼接起来
    # 拼接后的最终预测张量形状为 (B, T-1, Latent_Dim)
    # 计算并返回拼接后的预测时序与真实目标 z_target 之间的 MSE 损失
    return criterion(torch.cat(preds, dim=1), z_target)
