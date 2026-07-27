import torch
import numpy as np
from torch.utils.data import Dataset
import os
from glob import glob
from PIL import Image


class FramePredictionDataset(Dataset):
    def __init__(self, sequence_dirs, context=5, rollout=1, transform=None, return_state=False, stride=None):
        '''
        sequence_dirs: list of directories, each containing ordered image frames.
        context: number of input frames before target.
        rollout: number of future frames to return after context.
        stride: number of frames to skip between samples (default: rollout // 2).
        '''
        self.samples = []
        self.context = context
        self.rollout = rollout
        self.transform = transform
        self.return_state = return_state
        self.stride = stride or max(1, rollout // 2)

        for seq_dir in sequence_dirs:
            frame_paths = sorted(glob(os.path.join(seq_dir, '*.png')))
            if len(frame_paths) < context + rollout:
                continue

            positions = np.load(os.path.join(seq_dir, 'positions.npy'))
            velocities = np.load(os.path.join(seq_dir, 'velocities.npy'))

            for i in range(0, len(frame_paths) - context - rollout + 1, self.stride):
                input_paths = frame_paths[i:i+context]
                target_paths = frame_paths[i+context : i+context+rollout]
                target_positions = positions[i+context : i+context+rollout]
                target_velocities = velocities[i+context : i+context+rollout]

                self.samples.append((input_paths, target_paths, target_positions, target_velocities))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_paths, target_paths, positions, velocities = self.samples[idx]

        input_frames = [self._load_image(p) for p in input_paths]
        target_frames = [self._load_image(p) for p in target_paths]

        input_tensor = torch.stack(input_frames, dim=0)       # (context, C, H, W)
        target_tensor = torch.stack(target_frames, dim=0)     # (rollout, C, H, W)

        if self.return_state:
            positions_tensor = torch.tensor(positions)        # (rollout, 2)
            velocities_tensor = torch.tensor(velocities)      # (rollout, 2)
            return input_tensor, target_tensor, positions_tensor, velocities_tensor

        return input_tensor, target_tensor

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


class TrajectoryPredictionDataset(Dataset):
    def __init__(self, sequence_dirs, context=5, transform=None, return_state=False):
        '''
        sequence_dirs: list of directories, each containing ordered image frames.
        context: number of input frames before target.
        '''
        self.samples = []
        self.context = context
        self.transform = transform
        self.return_state = return_state

        for seq_dir in sequence_dirs:
            frame_paths = sorted(glob(os.path.join(seq_dir, '*.png')))
            if len(frame_paths) < context + 1:
                continue

            positions = np.load(os.path.join(seq_dir, 'positions.npy'))
            velocities = np.load(os.path.join(seq_dir, 'velocities.npy'))

            # first context frames are the initial state
            input_paths = frame_paths[:context]
            target_paths = frame_paths[context:]
            self.samples.append((input_paths, target_paths, positions[context:], velocities[context:]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_paths, target_paths, positions, velocities = self.samples[idx]

        input_frames = [self._load_image(p) for p in input_paths]
        target_frames = [self._load_image(p) for p in target_paths]

        input_seq = torch.stack(input_frames, dim=0)  # (T, C, H, W)
        target_seq = torch.stack(target_frames, dim=0)

        if self.return_state:
            return input_seq, target_seq, torch.tensor(positions), torch.tensor(velocities)

        return input_seq, target_seq  # (T, C, H, W), (C, H, W)

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


if __name__ == '__main__':
    # Example usage
    sequence_dirs = [os.path.join('physics-data-v2', f'traj-{i}') for i in range(500)]
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
    ])
    dataset = FramePredictionDataset(sequence_dirs, transform=transform)
    print(f'Number of samples: {len(dataset)}')
