from model import SpatioTemporalTransformer
from dataset import FramePredictionDataset
from torchvision import transforms
import torch
import argparse
import os
import json
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load config
    with open(os.path.join(args.model_dir, 'args.json')) as f:
        config = json.load(f)

    # Create model
    model = SpatioTemporalTransformer(
        input_channels=1,
        n_layers=config.get('transformer_layers', 2),
        d_model=config.get('transformer_dim', 128),
        n_heads=config.get('transformer_heads', 4),
        patch_size=config.get('patch_size', 16),
        img_size=config.get('image_size', 128),
        context=config.get('context', 5),
    ).to(device)

    # Load checkpoint
    ckpt_path = os.path.join(args.model_dir, 'checkpoints', 'last.ckpt')
    if not os.path.exists(ckpt_path):
        print("No last.ckpt found. Looking for any checkpoint...")
        for root, dirs, files in os.walk(args.model_dir):
            for f in files:
                if f.endswith('.ckpt'):
                    ckpt_path = os.path.join(root, f)
                    print("Using:", ckpt_path)
                    break

    state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
    model.load_state_dict({k.replace('net.', ''): v for k, v in state_dict.items() if 'net.' in k or k.startswith('patch_embed')})
    model.eval()

    print("STT model loaded successfully!")

    # Simple evaluation (add more as needed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
    val_dataset = FramePredictionDataset([os.path.join(args.data_dir, f'traj-{i}') for i in range(900, 910)], context=5, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("Running inference...")
    # Add your prediction logic here similar to frame_prediction in create_video.py

    print("Evaluation complete for STT.")