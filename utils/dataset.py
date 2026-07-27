import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class FramePredictionDataset(Dataset):
    def __init__(
        self,
        sequence_dirs,
        context=5,
        rollout=1,
        stride=1,
        transform=None,
        grayscale=True,
        invert=False,
        return_state=False
    ):
        self.sequence_dirs = sequence_dirs
        self.context = context
        self.rollout = rollout
        self.stride = stride
        self.transform = transform
        self.grayscale = grayscale
        self.invert = invert
        self.return_state = return_state
        self.samples = []

        for seq_dir in sequence_dirs:
            frame_paths = sorted(glob(os.path.join(seq_dir, "frame_*.png")))
            pos_path = os.path.join(seq_dir, "positions.npy")
            vel_path = os.path.join(seq_dir, "velocities.npy")

            if len(frame_paths) < context + rollout:
                continue
            if not os.path.exists(pos_path) or not os.path.exists(vel_path):
                continue

            positions = np.load(pos_path)
            velocities = np.load(vel_path)

            n_frames = min(len(frame_paths), len(positions), len(velocities))
            frame_paths = frame_paths[:n_frames]
            positions = positions[:n_frames]
            velocities = velocities[:n_frames]

            max_start = n_frames - context - rollout + 1
            for start_idx in range(0, max_start, stride):
                input_paths = frame_paths[start_idx : start_idx + context]
                target_paths = frame_paths[start_idx + context : start_idx + context + rollout]

                target_positions = positions[start_idx + context : start_idx + context + rollout]
                target_velocities = velocities[start_idx + context : start_idx + context + rollout]

                self.samples.append(
                    {
                        "seq_dir": seq_dir,
                        "input_paths": input_paths,
                        "target_paths": target_paths,
                        "target_positions": target_positions,
                        "target_velocities": target_velocities,
                        "start_idx": start_idx
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_frames = [self._load_image(p) for p in sample["input_paths"]]
        target_frames = [self._load_image(p) for p in sample["target_paths"]]

        input_tensor = torch.stack(input_frames, dim=0)
        target_tensor = torch.stack(target_frames, dim=0)

        if self.return_state:
            positions_tensor = torch.tensor(sample["target_positions"], dtype=torch.float32)
            velocities_tensor = torch.tensor(sample["target_velocities"], dtype=torch.float32)
            return input_tensor, target_tensor, positions_tensor, velocities_tensor

        return input_tensor, target_tensor

    def _load_image(self, path):
        img = Image.open(path)
        img = img.convert("L" if self.grayscale else "RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self._default_to_tensor(img)

        if self.invert:
            img = 1.0 - img

        return img

    @staticmethod
    def _default_to_tensor(img):
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[None, :, :]
        else:
            arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)


class TrajectoryPredictionDataset(Dataset):
    def __init__(
        self,
        sequence_dirs,
        context=5,
        transform=None,
        grayscale=True,
        invert=False,
        return_state=False
    ):
        self.sequence_dirs = sequence_dirs
        self.context = context
        self.transform = transform
        self.grayscale = grayscale
        self.invert = invert
        self.return_state = return_state
        self.samples = []

        for seq_dir in sequence_dirs:
            frame_paths = sorted(glob(os.path.join(seq_dir, "frame_*.png")))
            pos_path = os.path.join(seq_dir, "positions.npy")
            vel_path = os.path.join(seq_dir, "velocities.npy")

            if len(frame_paths) < context + 1:
                continue
            if not os.path.exists(pos_path) or not os.path.exists(vel_path):
                continue

            positions = np.load(pos_path)
            velocities = np.load(vel_path)

            n_frames = min(len(frame_paths), len(positions), len(velocities))
            frame_paths = frame_paths[:n_frames]
            positions = positions[:n_frames]
            velocities = velocities[:n_frames]

            input_paths = frame_paths[:context]
            target_paths = frame_paths[context:]

            target_positions = positions[context:]
            target_velocities = velocities[context:]

            self.samples.append(
                {
                    "seq_dir": seq_dir,
                    "input_paths": input_paths,
                    "target_paths": target_paths,
                    "target_positions": target_positions,
                    "target_velocities": target_velocities
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_frames = [self._load_image(p) for p in sample["input_paths"]]
        target_frames = [self._load_image(p) for p in sample["target_paths"]]

        input_tensor = torch.stack(input_frames, dim=0)
        target_tensor = torch.stack(target_frames, dim=0)

        if self.return_state:
            positions_tensor = torch.tensor(sample["target_positions"], dtype=torch.float32)
            velocities_tensor = torch.tensor(sample["target_velocities"], dtype=torch.float32)
            return input_tensor, target_tensor, positions_tensor, velocities_tensor

        return input_tensor, target_tensor

    def _load_image(self, path):
        img = Image.open(path)
        img = img.convert("L" if self.grayscale else "RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self._default_to_tensor(img)

        if self.invert:
            img = 1.0 - img

        return img

    @staticmethod
    def _default_to_tensor(img):
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[None, :, :]
        else:
            arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)


def get_sequence_dirs(root_dir):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    seq_dirs = sorted(
        [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("traj-")
        ]
    )
    return seq_dirs