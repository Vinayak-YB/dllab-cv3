import torch
from torchvision.utils import save_image
from torchvision import transforms
import os
import sys

sys.path.append('..')

from autoencoder import ConvAutoencoder
from dataset import FramePredictionDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load autoencoder
ae = ConvAutoencoder(latent_dim=128).to(device)
ae.load_state_dict(torch.load("../models/autoencoder/best.pth", map_location=device, weights_only=True))
ae.eval()

print("Autoencoder loaded successfully!")

# Take one sample
transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
data_dirs = [os.path.join("..", "data_id", "traj-100")]
dataset = FramePredictionDataset(data_dirs, context=5, transform=transform)

input_seq, target = dataset[0]
target = target.squeeze(0).unsqueeze(0).to(device)   # [1, 1, 128, 128]

# Reconstruct
with torch.no_grad():
    recon, latent = ae(target)

print(f"Original shape: {target.shape}")
print(f"Reconstructed shape: {recon.shape}")
print(f"Latent shape: {latent.shape}")

# Save images for visual check
os.makedirs("../models/autoencoder/test", exist_ok=True)
save_image(target, "../models/autoencoder/test/original.png")
save_image(recon, "../models/autoencoder/test/reconstructed.png")

print("✅ Saved original.png and reconstructed.png in ../models/autoencoder/test/")
print("Open them to see how well the autoencoder reconstructs the ball!")