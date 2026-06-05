import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MovingMNISTDataset
from model import Autoencoder, LatentDynamicsModel

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train_autoencoder(ae, dataloader, device, epochs=20, lr=1e-3, save_path='ae_weights.pth'):
    print("--- Starting Phase 1: Training Autoencoder ---")
    ae.to(device)
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        ae.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # batch shape: (B, seq_len, 1, 64, 64)
            # We fold the sequence dimension into the batch dimension for AE training
            B, seq_len, C, H, W = batch.shape
            x = batch.view(B * seq_len, C, H, W).to(device)
            
            optimizer.zero_grad()
            x_recon, _ = ae(x)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 20 == 0:
                print(f"AE Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"AE Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
        
    torch.save(ae.state_dict(), save_path)
    print(f"Autoencoder weights saved to {save_path}\n")

def train_dynamics(ae, dyn, dataloader, device, epochs=50, lr=1e-3, save_path='dyn_weights.pth'):
    print("--- Starting Phase 2: Training Latent Dynamics Model ---")
    ae.to(device)
    ae.eval() # Freeze Autoencoder
    dyn.to(device)
    optimizer = optim.Adam(dyn.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        dyn.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # batch shape: (B, seq_len, 1, 64, 64)
            batch = batch.to(device)
            B, seq_len, C, H, W = batch.shape
            
            # 1. Encode the whole sequence to get latents
            with torch.no_grad():
                x_flat = batch.view(B * seq_len, C, H, W)
                z_flat = ae.encoder(x_flat)
                z_seq = z_flat.view(B, seq_len, -1) # (B, seq_len, latent_dim)
            
            # 2. Train LSTM to predict z_{t+1} from z_t
            # Input to LSTM: z_0 to z_{T-2}
            # Target for LSTM: z_1 to z_{T-1}
            z_input = z_seq[:, :-1, :]
            z_target = z_seq[:, 1:, :]
            
            optimizer.zero_grad()
            z_pred, _ = dyn(z_input)
            loss = criterion(z_pred, z_target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 20 == 0:
                print(f"Dynamics Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"Dynamics Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
        
    torch.save(dyn.state_dict(), save_path)
    print(f"Dynamics weights saved to {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ae_epochs', type=int, default=50)
    parser.add_argument('--dyn_epochs', type=int, default=200)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Preparing dataset...")
    dataset = MovingMNISTDataset(download=True, train=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Initialize Models
    ae = Autoencoder(latent_dim=128)
    dyn = LatentDynamicsModel(latent_dim=128, hidden_dim=256)
    
    # 3. Train
    ae_path = 'ae_weights.pth'
    dyn_path = 'dyn_weights.pth'
    
    # Always train (overwrites existing weights)
    train_autoencoder(ae, dataloader, device, epochs=args.ae_epochs, save_path=ae_path)
    train_dynamics(ae, dyn, dataloader, device, epochs=args.dyn_epochs, save_path=dyn_path)
        
    print("Training finished! You can now run inference.py to see the model 'dream'.")
