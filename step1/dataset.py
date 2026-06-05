import os
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MovingMNISTDataset(Dataset):
    def __init__(self, root='./data', download=True, train=True):
        self.root = root
        self.file_path = os.path.join(root, 'mnist_test_seq.npy')
        
        if download and not os.path.exists(self.file_path):
            self.download()
            
        if not os.path.exists(self.file_path):
            raise RuntimeError('Dataset not found or corrupted.')
            
        # Load the dataset
        # Original shape: (20, 10000, 64, 64)
        data = np.load(self.file_path)
        
        # Split into train (9000) and test (1000)
        # We also swap axes to get (num_samples, seq_len, H, W)
        data = np.transpose(data, (1, 0, 2, 3)) # Shape: (10000, 20, 64, 64)
        
        if train:
            self.data = data[:9000]
        else:
            self.data = data[9000:]
            
        # Convert to float32 and scale to [0, 1]
        self.data = self.data.astype(np.float32) / 255.0
        
    def download(self):
        os.makedirs(self.root, exist_ok=True)
        url = 'http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy'
        print(f"Downloading Moving MNIST from {url}...")
        urllib.request.urlretrieve(url, self.file_path)
        print("Download complete.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # returns sequence of shape (seq_len, channels, H, W)
        # Moving MNIST is grayscale, so we add a channel dimension
        # Resulting shape: (20, 1, 64, 64)
        seq = self.data[idx]
        seq = np.expand_dims(seq, axis=1) # (20, 1, 64, 64)
        return torch.from_numpy(seq)

if __name__ == "__main__":
    # Test dataloader
    dataset = MovingMNISTDataset(download=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in loader:
        print(f"Batch shape: {batch.shape}") # Should be [4, 20, 1, 64, 64]
        print(f"Min: {batch.min():.2f}, Max: {batch.max():.2f}")
        break
