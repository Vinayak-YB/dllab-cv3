import torch
import torch.nn as nn
import os
import sys
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm

sys.path.append('..')

from dataset import FramePredictionDataset
from autoencoder import ConvAutoencoder

class LatentDynamics(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.rnn = nn.LSTM(latent_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latent_seq):
        out, _ = self.rnn(latent_seq)
        return self.out(out[:, -1])

def main():
    device = 'cuda'
    
    # Load models
    ae = ConvAutoencoder(latent_dim=128).to(device)
    ae.load_state_dict(torch.load("../models/autoencoder/best.pth", map_location=device, weights_only=True))
    ae.eval()

    dynamics = LatentDynamics().to(device)
    dynamics.load_state_dict(torch.load("../models/latent_dynamics/best.pth", map_location=device, weights_only=True))
    dynamics.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])

    # Take a few trajectories for visualization
    data_dirs = [os.path.join("..", "data_id", f'traj-{i}') for i in range(10, 15)]
    dataset = FramePredictionDataset(data_dirs, context=5, transform=transform)
    
    os.makedirs("../models/latent_rollouts", exist_ok=True)

    print("Generating latent rollouts...")
    for idx in range(len(dataset)):
        input_seq, target = dataset[idx]
        input_seq = input_seq.unsqueeze(0).to(device)   # [1, context, 1, H, W]

        # Encode context (last frame)
        with torch.no_grad():
            last_frame = input_seq[:, -1]
            latent = ae.encode(last_frame)

        # Simple rollout (predict next latent)
        pred_latents = []
        current_latent = latent
        for _ in range(20):   # rollout 20 steps
            pred_latent = dynamics(current_latent.unsqueeze(1))
            pred_latents.append(pred_latent)
            current_latent = pred_latent

        # Decode to images
        with torch.no_grad():
            recon_frames = ae.decode(torch.cat(pred_latents, dim=0))

        # Save some frames
        for t in range(0, min(10, len(recon_frames)), 2):
            save_image(recon_frames[t], f"../models/latent_rollouts/rollout_{idx}_step_{t}.png")

        print(f"Saved rollout for trajectory {idx}")

    print("✅ Rollouts saved in ../models/latent_rollouts/")

if __name__ == "__main__":
    main()