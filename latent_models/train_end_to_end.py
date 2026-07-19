import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
from tqdm import tqdm

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])

    data_dirs = [os.path.join("..", "data_id", f'traj-{i}') for i in range(600)]
    dataset = FramePredictionDataset(data_dirs, context=5, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    model = EndToEndPredictor(latent_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)  # lower LR
    criterion = nn.MSELoss()

    print("Training Improved End-to-End Latent Predictor...")

    for epoch in range(80):
        total_loss = 0.0
        for input_seq, target in tqdm(loader, desc=f"Epoch {epoch+1}"):
            input_seq = input_seq.to(device)
            target = target.squeeze(1).to(device)

            optimizer.zero_grad()
            pred_frame = model(input_seq)
            loss = criterion(pred_frame, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} - Frame MSE: {avg_loss:.6f}")

    os.makedirs("../models/end_to_end", exist_ok=True)
    torch.save(model.state_dict(), "../models/end_to_end/best_v2.pth")
    print("✅ Improved End-to-End training finished!")

if __name__ == "__main__":
    main()