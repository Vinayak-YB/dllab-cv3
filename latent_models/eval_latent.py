import torch
import torch.nn as nn
import os
import sys
from torchvision import transforms
from tqdm import tqdm

sys.path.append('..')

from dataset import FramePredictionDataset
from autoencoder import ConvAutoencoder

# Import the dynamics class (we'll define it inline for simplicity)
class LatentDynamics(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.rnn = nn.LSTM(latent_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latent_seq):
        out, _ = self.rnn(latent_seq)
        return self.out(out[:, -1])

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Autoencoder
    ae = ConvAutoencoder(latent_dim=128).to(device)
    ae.load_state_dict(torch.load("../models/autoencoder/best.pth", map_location=device, weights_only=True))
    ae.eval()

    # Load Dynamics
    dynamics = LatentDynamics().to(device)
    dynamics.load_state_dict(torch.load("../models/latent_dynamics/best.pth", map_location=device, weights_only=True))
    dynamics.eval()

    print("✅ Both models loaded successfully!")

    # Quick test on a few trajectories
    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
    data_dirs = [os.path.join("..", "data_id", f'traj-{i}') for i in range(50, 55)]
    dataset = FramePredictionDataset(data_dirs, context=5, transform=transform)
    
    print("Models are ready for rollout evaluation!")
    # We can expand this later to generate videos

if __name__ == "__main__":
    main()