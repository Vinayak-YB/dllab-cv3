import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
from tqdm import tqdm
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

sys.path.append('..')

from dataset import FramePredictionDataset
from autoencoder import ConvAutoencoder

class SimpleDiffusionPredictor(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.ae = ConvAutoencoder(latent_dim=latent_dim)
        self.noise_predictor = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, noisy_latent, timestep):
        # Simple noise predictor
        return self.noise_predictor(noisy_latent)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Autoencoder
    ae = ConvAutoencoder(latent_dim=128).to(device)
    ae.load_state_dict(torch.load("../models/autoencoder/best.pth", map_location=device, weights_only=True))
    ae.eval()

    model = SimpleDiffusionPredictor(latent_dim=128).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=5000)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])

    data_dirs = [os.path.join("..", "data_id", f'traj-{i}') for i in range(400)]
    dataset = FramePredictionDataset(data_dirs, context=5, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    print("Training Latent Diffusion...")

    for epoch in range(30):
        total_loss = 0.0
        for input_seq, target in tqdm(loader, desc=f"Epoch {epoch+1}"):
            input_seq = input_seq.to(device)
            target = target.squeeze(1).to(device)

            with torch.no_grad():
                target_latent = ae.encode(target)

            # Add noise
            noise = torch.randn_like(target_latent)
            timesteps = torch.randint(0, 1000, (target_latent.shape[0],)).long().to(device)
            noisy_latent = noise_scheduler.add_noise(target_latent, noise, timesteps)

            # Predict noise
            noise_pred = model(noisy_latent, timesteps)

            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} - Diffusion Loss: {avg_loss:.6f}")

    os.makedirs("../models/diffusion", exist_ok=True)
    torch.save(model.state_dict(), "../models/diffusion/best.pth")
    print("✅ Latent Diffusion training finished!")

if __name__ == "__main__":
    main()