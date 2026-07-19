import torch
import torch.nn as nn
import os
import sys
from torchvision.utils import save_image
from diffusers import DDPMScheduler
from tqdm import tqdm

sys.path.append('..')

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
        return self.noise_predictor(noisy_latent)

def main():
    device = 'cuda'

    model = SimpleDiffusionPredictor(latent_dim=128).to(device)
    model.load_state_dict(torch.load("../models/diffusion/best.pth", map_location=device, weights_only=True))
    model.eval()

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise_scheduler.set_timesteps(50)  # fewer steps for faster sampling

    print("Generating samples from Diffusion model...")

    os.makedirs("../models/diffusion/samples", exist_ok=True)

    # Generate a few samples
    for i in range(8):
        # Start from noise
        latent = torch.randn(1, 128).to(device)

        for t in tqdm(noise_scheduler.timesteps, desc=f"Sample {i}"):
            with torch.no_grad():
                noise_pred = model(latent, t)
                latent = noise_scheduler.step(noise_pred, t, latent).prev_sample

        # Decode
        with torch.no_grad():
            image = model.ae.decode(latent)

        save_image(image, f"../models/diffusion/samples/sample_{i}.png")

    print("✅ Samples saved in ../models/diffusion/samples/")

if __name__ == "__main__":
    main()