# Replace latent_dynamics.py with this
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
        self.rnn = nn.GRU(latent_dim*2, 512, num_layers=3, batch_first=True)  # velocity + position
        self.out = nn.Linear(512, latent_dim)

    def forward(self, latent_seq):
        out, _ = self.rnn(latent_seq)
        return self.out(out[:, -1])

# ... (rest of training code with velocity concatenation)