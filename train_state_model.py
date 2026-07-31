import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.state_model import StateMLP


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StateSequenceDataset(Dataset):
    def __init__(self, sequence_dirs, context=5):
        self.sequence_dirs = sequence_dirs
        self.context = context
        self.samples = []

        for seq_dir in sequence_dirs:
            pos_path = os.path.join(seq_dir, "positions.npy")
            vel_path = os.path.join(seq_dir, "velocities.npy")

            if not os.path.exists(pos_path) or not os.path.exists(vel_path):
                continue

            positions = np.load(pos_path)
            velocities = np.load(vel_path)

            if len(positions) != len(velocities):
                continue

            traj_length = len(positions)
            if traj_length < context + 1:
                continue

            states = np.concatenate([positions, velocities], axis=1)  # (T, 4)

            for start_idx in range(traj_length - context):
                input_seq = states[start_idx: start_idx + context]
                target_state = states[start_idx + context]
                self.samples.append((input_seq, target_state))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_state = self.samples[idx]
        return (
            torch.tensor(input_seq, dtype=torch.float32),
            torch.tensor(target_state, dtype=torch.float32),
        )


def get_sequence_dirs(root_dir):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    seq_dirs = sorted(
        [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("traj-")
        ]
    )
    return seq_dirs


def split_sequence_dirs(sequence_dirs, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    sequence_dirs = sequence_dirs.copy()
    rng.shuffle(sequence_dirs)

    val_size = max(1, int(len(sequence_dirs) * val_ratio))
    train_dirs = sequence_dirs[:-val_size]
    val_dirs = sequence_dirs[-val_size:]
    return train_dirs, val_dirs


def build_datasets(args):
    if args.train_dir is not None and args.val_dir is not None:
        train_dirs = get_sequence_dirs(args.train_dir)
        val_dirs = get_sequence_dirs(args.val_dir)

        if len(train_dirs) == 0 or len(val_dirs) == 0:
            raise ValueError(
                f"No valid sequences found. "
                f"train_dir: {args.train_dir} ({len(train_dirs)} dirs), "
                f"val_dir: {args.val_dir} ({len(val_dirs)} dirs)"
            )

        split_mode = "explicit_train_val_dirs"
        print(f"Loading train from: {args.train_dir}  ({len(train_dirs)} traj.)")
        print(f"Loading val   from: {args.val_dir}    ({len(val_dirs)} traj.)")

    else:
        if args.data_dir is None:
            raise ValueError("Either (train_dir, val_dir) or data_dir must be provided.")

        sequence_dirs = get_sequence_dirs(args.data_dir)
        if len(sequence_dirs) == 0:
            raise ValueError(f"No valid sequences found in data_dir: {args.data_dir}")

        train_dirs, val_dirs = split_sequence_dirs(
            sequence_dirs,
            val_ratio=args.val_ratio,
            seed=args.seed
        )

        split_mode = "legacy_internal_split"
        print(f"Loading data from: {args.data_dir} ({len(sequence_dirs)} traj.)")
        print(f"Split mode: trajectory-level internal split (val_ratio={args.val_ratio})")

    train_dataset = StateSequenceDataset(sequence_dirs=train_dirs, context=args.context)
    val_dataset = StateSequenceDataset(sequence_dirs=val_dirs, context=args.context)

    print(f"Split mode used: {split_mode}")
    print(f"Train trajectories: {len(train_dirs)}")
    print(f"Val trajectories:   {len(val_dirs)}")
    print(f"Train samples:      {len(train_dataset)}")
    print(f"Val samples:        {len(val_dataset)}")

    return train_dataset, val_dataset


def main(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    if args.train_dir is not None and args.val_dir is not None:
        print(f"Train dir: {args.train_dir}")
        print(f"Val dir:   {args.val_dir}")
    else:
        print(f"Data dir:  {args.data_dir}")

    train_dataset, val_dataset = build_datasets(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = StateMLP(
        context=args.context,
        state_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_train_loss = 0.0

        for input_seq, target_state in train_loader:
            input_seq = input_seq.to(device)
            target_state = target_state.to(device)

            optimizer.zero_grad()
            pred_state = model(input_seq)
            loss = F.mse_loss(pred_state, target_state)
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            running_train_loss += loss.item() * input_seq.size(0)

        avg_train_loss = running_train_loss / len(train_dataset)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for input_seq, target_state in val_loader:
                input_seq = input_seq.to(device)
                target_state = target_state.to(device)

                pred_state = model(input_seq)
                loss = F.mse_loss(pred_state, target_state)
                running_val_loss += loss.item() * input_seq.size(0)

        avg_val_loss = running_val_loss / len(val_dataset)

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss:   {avg_val_loss:.6f}")

        ckpt_path_last = os.path.join(save_dir, "last.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "args": vars(args),
            },
            ckpt_path_last,
            _use_new_zipfile_serialization=False,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path_best = os.path.join(save_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "args": vars(args),
                },
                ckpt_path_best,
                _use_new_zipfile_serialization=False,
            )
            print(f"Saved new best checkpoint to {ckpt_path_best}")

    print("\nTraining completed.")
    print(f"Checkpoints saved in: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train state-space model (MLP) for physics prediction")

    parser.add_argument("--data_dir", type=str, default=None,
                        help="Legacy: single data directory with internal train/val split.")
    parser.add_argument("--train_dir", type=str, default=None,
                        help="Explicit training directory (e.g., data/bouncing_ball/train).")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Explicit validation directory (e.g., data/bouncing_ball/val).")

    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default="state_mlp")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Used only in legacy mode (data_dir).")

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)

    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)