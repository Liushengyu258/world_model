import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)   # -> 32x32
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # -> 16x16
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1) # -> 8x8
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1) # -> 4x4
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)
        
    def forward(self, x):
        # x shape: (batch, 1, 64, 64)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.size(0), -1)
        z = self.fc(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1) # -> 8x8
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # -> 16x16
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   # -> 32x32
        self.deconv4 = nn.ConvTranspose2d(32, 1, 4, 2, 1)    # -> 64x64
        
    def forward(self, z):
        # z shape: (batch, latent_dim)
        x = self.fc(z)
        x = x.reshape(x.size(0), 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x)) # Output values between 0 and 1
        return x

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

class LatentDynamicsModel(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, z_seq):
        # z_seq shape: (batch, seq_len, latent_dim)
        lstm_out, (h_n, c_n) = self.lstm(z_seq)
        # We predict the next latent for every step in the sequence
        z_next_pred = self.fc(lstm_out)
        return z_next_pred, (h_n, c_n)

class WorldModel(nn.Module):
    """
    Combined model for convenience (though we train components separately in this project)
    """
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.autoencoder = Autoencoder(latent_dim)
        self.dynamics = LatentDynamicsModel(latent_dim, hidden_dim)

if __name__ == "__main__":
    # Test shapes
    device = torch.device('cpu')
    x = torch.rand(4, 1, 64, 64)
    ae = Autoencoder().to(device)
    x_recon, z = ae(x)
    print(f"Autoencoder Input: {x.shape}, Recon: {x_recon.shape}, Latent: {z.shape}")
    
    z_seq = torch.rand(4, 10, 128) # 10 frames sequence
    dyn = LatentDynamicsModel().to(device)
    z_next_pred, _ = dyn(z_seq)
    print(f"Dynamics Input: {z_seq.shape}, Next Pred: {z_next_pred.shape}")
