import os
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from dataset import MovingMNISTDataset
from torch.utils.data import DataLoader
from model import Autoencoder, LatentDynamicsModel
from train import get_device

def generate_gif(ae, dyn, dataloader, device, num_prompt_frames=5, total_frames=20, save_path="world_model_dream.gif"):
    print("Generating predictions...")
    ae.to(device)
    dyn.to(device)
    ae.eval()
    dyn.eval()
    
    # Get one batch from test set
    batch = next(iter(dataloader)) # (B, seq_len, 1, 64, 64)
    # We just take the first sequence in the batch
    seq = batch[0].unsqueeze(0).to(device) # (1, 20, 1, 64, 64)
    
    # Ground truth for comparison
    gt_frames = seq[0].cpu().numpy() # (20, 1, 64, 64)
    gt_frames = gt_frames.squeeze(1) # (20, 64, 64)
    
    predicted_frames = []
    
    with torch.no_grad():
        # Encode all ground truth frames to latents just to get the prompt latents
        # and to compare if we want
        z_seq_gt = []
        for t in range(total_frames):
            frame = seq[:, t, :, :, :]
            z = ae.encoder(frame)
            z_seq_gt.append(z)
            
        z_seq_gt = torch.stack(z_seq_gt, dim=1) # (1, 20, latent_dim)
        
        # 1. Provide the initial prompt
        # We start with the first `num_prompt_frames`
        current_z_seq = z_seq_gt[:, :num_prompt_frames, :]
        
        # Add prompt frames to predicted_frames (we just use GT for prompt)
        for t in range(num_prompt_frames):
            frame_recon = ae.decoder(current_z_seq[:, t, :])
            predicted_frames.append(frame_recon[0, 0].cpu().numpy())
            
        # 2. Autoregressive prediction
        # We need hidden states for LSTM
        # We pass the prompt sequence through LSTM to build up hidden state
        _, (h, c) = dyn.lstm(current_z_seq)
        
        # The last predicted z is the input for the next step
        # Wait, the LSTM outputs predictions for the whole sequence.
        # The prediction for the *next* frame (t+1) comes from the last output.
        lstm_out, _ = dyn.lstm(current_z_seq)
        z_next = dyn.fc(lstm_out[:, -1:, :]) # (1, 1, latent_dim)
        
        frame_recon = ae.decoder(z_next[:, 0, :])
        predicted_frames.append(frame_recon[0, 0].cpu().numpy())
        
        # Now we autoregressively predict the rest
        current_z_input = z_next
        
        for t in range(num_prompt_frames + 1, total_frames):
            # Pass the single step to LSTM, passing along the hidden state
            lstm_out, (h, c) = dyn.lstm(current_z_input, (h, c))
            z_next = dyn.fc(lstm_out)
            
            # Decode to image
            frame_recon = ae.decoder(z_next[:, 0, :])
            predicted_frames.append(frame_recon[0, 0].cpu().numpy())
            
            # Update input for next step
            current_z_input = z_next
            
    # Combine GT and Predicted side by side
    frames_to_save = []
    for t in range(total_frames):
        gt_img = (gt_frames[t] * 255).astype(np.uint8)
        pred_img = (predicted_frames[t] * 255).astype(np.uint8)
        
        # Mark prompt frames with a border or text?
        # A simpler way: combine side-by-side
        combined = np.concatenate([gt_img, pred_img], axis=1) # (64, 128)
        frames_to_save.append(combined)
        
    imageio.mimsave(save_path, frames_to_save, fps=5)
    print(f"Dream saved to {save_path}!")
    print("Left: Ground Truth | Right: Model Dream")
    print(f"(First {num_prompt_frames} frames are given as prompt)")

if __name__ == "__main__":
    device = get_device()
    
    print("Loading test dataset...")
    dataset = MovingMNISTDataset(download=True, train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    ae = Autoencoder(latent_dim=128)
    dyn = LatentDynamicsModel(latent_dim=128, hidden_dim=256)
    
    ae_path = 'ae_weights.pth'
    dyn_path = 'dyn_weights.pth'
    
    if not os.path.exists(ae_path) or not os.path.exists(dyn_path):
        print("Model weights not found! Please run train.py first.")
    else:
        ae.load_state_dict(torch.load(ae_path, map_location=device, weights_only=True))
        dyn.load_state_dict(torch.load(dyn_path, map_location=device, weights_only=True))
        generate_gif(ae, dyn, dataloader, device)
