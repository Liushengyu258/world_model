"""
数据集管理模块 (Baseline 阶段):
主要负责下载、载入、预处理 Moving MNIST 数据集。
Moving MNIST 是一个包含时序演化的视频数据集。在这个基础版本中，我们同样使用 20 帧的单通道 64x64 灰度视频序列。
"""

import os
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 获取当前脚本所在目录
_DIR = os.path.dirname(os.path.abspath(__file__))


class MovingMNISTDataset(Dataset):
    """
    Moving MNIST 视频时序数据集。
    继承自 PyTorch 的 Dataset 基类，实现时序视频的多帧读取。
    """

    def __init__(self, root=os.path.join(_DIR, 'data'), download=True, train=True):
        """
        参数:
            root (str): 数据集文件下载和存放的根目录。
            download (bool): 若本地文件不存在，是否自动从网络下载。
            train (bool): 是否作为训练集。True 返回前 9000 个序列，False 返回后 1000 个序列用于测试。
        """
        self.root = root
        self.file_path = os.path.join(root, 'mnist_test_seq.npy')
        
        # 1. 自动下载
        if download and not os.path.exists(self.file_path):
            self.download()
            
        if not os.path.exists(self.file_path):
            raise RuntimeError('Dataset not found or corrupted.')
            
        # 2. 载入原始的 Numpy 数组
        # 原始数据集 Shape 为: (20, 10000, 64, 64)
        # 维度含义: (时间步长T=20, 样本总数N=10000, 高度H=64, 宽度W=64)
        data = np.load(self.file_path)
        
        # 3. 维度重排 (Transpose)：
        # PyTorch 和常规定向中，通常期望将“样本维度 (Batch/Sample)”置于第一维。
        # 这里使用 np.transpose 将维度从 (20, 10000, 64, 64) 转换为 (10000, 20, 64, 64)
        # 转换后维度含义: (样本数N=10000, 时间步长T=20, 高度H=64, 宽度W=64)
        data = np.transpose(data, (1, 0, 2, 3)) # Shape: (10000, 20, 64, 64)
        
        # 4. 训练/测试集切分 (Train/Test Split)
        if train:
            # 前 9000 个视频序列作为训练集
            self.data = data[:9000]
        else:
            # 后 1000 个视频序列作为测试/评估集
            self.data = data[9000:]
            
        # 5. 数据类型转换与归一化
        # 原始像素值为 uint8 (0~255)，转换为 float32 并除以 255.0，缩放到 [0.0, 1.0] 区间
        # 这样做利于神经网络的权重初始化和梯度平稳流动，且配合最后一层 Decoder 的 Sigmoid 激活函数
        self.data = self.data.astype(np.float32) / 255.0
        
    def download(self):
        """从网络中下载 Moving MNIST 数据集二进制文件 (NPY格式)"""
        os.makedirs(self.root, exist_ok=True)
        url = 'http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy'
        print(f"Downloading Moving MNIST from {url}...")
        urllib.request.urlretrieve(url, self.file_path)
        print("Download complete.")

    def __len__(self):
        """返回数据集的样本总数"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引读取一个样本序列。
        
        返回:
            torch.Tensor: 时序视频张量，Shape 为 (T=20, C=1, H=64, W=64)
        """
        # 1. 提取第 idx 个视频序列，形状为 (20, 64, 64)
        seq = self.data[idx]
        
        # 2. 增加通道维度 (Channel Dimension)
        # 原始 Moving MNIST 是单通道灰度图，所以通道数为 1。
        # 使用 np.expand_dims 在第 1 维插入通道，将其变形为 (20, 1, 64, 64)
        # 变形后维度含义: (时间步长T=20, 通道数C=1, 高度H=64, 宽度W=64)
        seq = np.expand_dims(seq, axis=1) # (20, 1, 64, 64)
        
        # 3. 将 NumPy 数组转为 PyTorch 张量返回
        return torch.from_numpy(seq)


if __name__ == "__main__":
    # 本地数据加载器单元测试
    dataset = MovingMNISTDataset(download=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in loader:
        # 期望的 batch 输出形状为: [Batch_Size=4, Time_Steps=20, Channels=1, Height=64, Width=64]
        print(f"Batch shape: {batch.shape}") # 应该显示 [4, 20, 1, 64, 64]
        print(f"Min: {batch.min():.2f}, Max: {batch.max():.2f}")
        break

