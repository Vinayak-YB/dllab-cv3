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
    def __init__(self, latent_dim=256):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, 512, num_layers=3, batch_first=True)
        self.out = nn.Linear(512, latent_dim)

    def forward(self, latent_seq):
        out, _ = self.rnn(latent_seq)
        return self.out(out[:, -1])

class EndToEndPredictor(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.ae = ConvAutoencoder(latent_dim=latent_dim)
        self.dynamics = LatentDynamics(latent_dim=latent_dim)

    def forward(self, input_seq):
        B, T, C, H, W = input_seq.shape
        latent_seq = torch.stack([self.ae.encode(input_seq[:, t]) for t in range(T)], dim=1)
        pred_latent = self.dynamics(latent_seq)
        pred_frame = self.ae.decode(pred_latent)
        return pred_frame

def main():
    device = 'cuda'

    model = EndToEndPredictor(latent_dim=256).to(device)
    model.load_state_dict(torch.load("../models/end_to_end/best.pth", map_location=device, weights_only=True))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])

    # Use a range that definitely has data
    data_dirs = [os.path.join("..", "data_id", f'traj-{i}') for i in range(50, 100)]

    print("Evaluating End-to-End Latent Predictor...")

    pos_errors = []

    for idx in tqdm(range(len(data_dirs))):
        try:
            dataset = FramePredictionDataset([data_dirs[idx]], context=5, transform=transform)
            if len(dataset) == 0:
                continue

            input_seq, target = dataset[0]
            input_seq = input_seq.unsqueeze(0).to(device)
            target = target.squeeze(0).to(device)

            traj_dir = data_dirs[idx]
            gt_positions = np.load(os.path.join(traj_dir, "positions.npy"))

            with torch.no_grad():
                pred_frame = model(input_seq)

            pred_center = pred_frame.mean(dim=[2, 3]).cpu().numpy().flatten()
            gt_center = gt_positions[6]   # after context

            error = np.abs(pred_center - gt_center).mean()
            pos_errors.append(error)

        except Exception as e:
            continue

    if pos_errors:
        print(f"\n=== End-to-End Latent Predictor Results ===")
        print(f"Average Position Error: {np.mean(pos_errors):.4f}")
        print(f"Median Position Error: {np.median(pos_errors):.4f}")
        print(f"Evaluated trajectories: {len(pos_errors)}")
    else:
        print("No valid trajectories evaluated.")

if __name__ == "__main__":
    main()