import torch
import torch.nn as nn
import os
import sys
import numpy as np
from tqdm import tqdm
from torchvision import transforms

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load models
    ae = ConvAutoencoder(latent_dim=128).to(device)
    ae.load_state_dict(torch.load("../models/autoencoder/best.pth", map_location=device, weights_only=True))
    ae.eval()

    dynamics = LatentDynamics().to(device)
    dynamics.load_state_dict(torch.load("../models/latent_dynamics/best.pth", map_location=device, weights_only=True))
    dynamics.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])

    n_samples = 50
    data_dirs = [os.path.join("..", "data_id", f'traj-{i}') for i in range(100, 100 + n_samples)]

    print(f"Evaluating Latent Dynamics on {n_samples} trajectories...")

    pos_errors = []

    for idx in tqdm(range(len(data_dirs))):
        try:
            dataset = FramePredictionDataset([data_dirs[idx]], context=5, transform=transform)
            if len(dataset) == 0:
                continue

            input_seq, target = dataset[0]
            input_seq = input_seq.unsqueeze(0).to(device)   # [1, 5, 1, H, W]
            target = target.squeeze(0).to(device)           # [1, H, W]

            traj_dir = data_dirs[idx]
            gt_positions = np.load(os.path.join(traj_dir, "positions.npy"))

            with torch.no_grad():
                # Use full context (5 frames)
                B, T, C, H, W = input_seq.shape
                latent_seq_list = [ae.encode(input_seq[:, t]) for t in range(T)]
                latent_seq = torch.stack(latent_seq_list, dim=1)   # [1, 5, 128]

                pred_latent = dynamics(latent_seq)
                pred_frame = ae.decode(pred_latent)

            # Center of mass
            pred_center = pred_frame.mean(dim=[2, 3]).cpu().numpy().flatten()
            gt_center = gt_positions[5 + 1]   # after context + 1 step

            error = np.abs(pred_center - gt_center).mean()
            pos_errors.append(error)

        except Exception as e:
            continue

    if pos_errors:
        print(f"\n=== Latent Dynamics Results (ID Split) ===")
        print(f"Average Position Error: {np.mean(pos_errors):.4f}")
        print(f"Median Position Error: {np.median(pos_errors):.4f}")
        print(f"Evaluated trajectories: {len(pos_errors)}")
    else:
        print("No valid trajectories evaluated.")

if __name__ == "__main__":
    main()