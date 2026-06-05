# 简易世界模型 (Simple World Model)

本项目用于帮助初学者理解世界模型的基本运作原理：**给定当前世界的状态，预测世界未来的走向**。

项目分为两个递进阶段，每个阶段是一个独立可运行的文件夹：

| 文件夹 | 模型 | 说明 |
|--------|------|------|
| `step1/` | AE + LSTM | 最简版世界模型，用于快速理解核心概念 |
| `step2/` | VAE + GRU + Self-Attention | 加入残差块、KL 退火、Scheduled Sampling 等进阶技巧 |

## 任务背景：Moving MNIST

为了在个人电脑上能快速跑通流程，我们选择了一个经典且轻量的任务：**Moving MNIST**。
该任务包含两个在 64x64 空间中做匀速直线运动、碰到边缘会反弹的手写数字。模型需要通过观察视频，学会其中的"物理运动规律"。

---

## Step 1：极简版 (AE + LSTM)

### 模型架构

1. **感知模块 (Autoencoder, `model.py`)**
   - 将图像压缩成 128 维特征向量，再解码还原。使用 `BCELoss` 避免模型输出全黑画面。
2. **动力学模块 (LSTM, `model.py`)**
   - 接收特征序列，预测下一帧的特征向量。

### 运行方法

```bash
cd step1

# 训练（数据集约 160MB，首次运行自动下载）
python -u train.py

# 推理（生成 world_model_dream.gif）
python inference.py
```

> **训练策略**：分阶段训练。
> - Phase 1：训练 Autoencoder，学习图像压缩与还原。
> - Phase 2：冻结 AE，训练 LSTM 预测下一帧特征。
>
> 如需重新训练，先删除 `ae_weights.pth` 和 `dyn_weights.pth`。

### 可选参数

```bash
python -u train.py --ae_epochs 50 --dyn_epochs 200 --batch_size 64
```

---

## Step 2：进阶版 (VAE + GRU + Attention)

### 相比 Step 1 的改进

- **VAE** 替代 AE：引入重参数化采样 + KL 散度正则化，潜空间更平滑
- **残差块 (ResidualBlock)**：每层编解码器后加残差连接，提升特征提取
- **GRU + Self-Attention** 替代 LSTM：GRU 更轻量，Attention 捕获长距离依赖
- **FFN + LayerNorm**：动力学模块增加非线性变换层
- **KL 退火**：前期关闭 KL 让模型先学重建，逐步开启防止后验坍缩
- **Scheduled Sampling**：训练后期用模型自身预测作为输入，减少推理时的误差累积
- **Cosine LR + Gradient Clipping**：更稳定的训练

### 运行方法

```bash
cd step2

# 训练
python -u train.py

# 推理
python inference.py
```

> 超参数集中在 `config.py` 中管理，可直接修改。

---

## 📊 如何解读生成的 GIF？

打开生成的 GIF，你会看到左右对比的画面：
- **左边 (Ground Truth)**：真实的未来画面。
- **右边 (Model Dream)**：模型推演的未来。前 5 帧为"引子"（与左边一致），第 6 帧起为模型纯自回归预测。

**画面模糊？** 训练轮数不足会导致重建质量差，且自回归误差会滚雪球式累积。增加训练轮数即可改善。
