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

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
    ])

    data_dirs = [os.path.join("..", "data_id", f'traj-{i}') for i in range(300)]
    dataset = FramePredictionDataset(data_dirs, context=5, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    model = ConvAutoencoder(latent_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("Starting Autoencoder Training...")

    for epoch in range(25):
        model.train()
        total_loss = 0.0
        for input_seq, target in tqdm(loader, desc=f"Epoch {epoch+1}/25"):
            # Remove extra dimension: [B, 1, 1, H, W] -> [B, 1, H, W]
            target = target.squeeze(1).to(device)

            optimizer.zero_grad()
            recon, latent = model(target)
            loss = criterion(recon, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} - Avg Reconstruction Loss: {avg_loss:.6f}")

    # Save
    os.makedirs("../models/autoencoder", exist_ok=True)
    torch.save(model.state_dict(), "../models/autoencoder/best.pth")
    print("✅ Autoencoder training finished and saved!")

if __name__ == "__main__":
    main()