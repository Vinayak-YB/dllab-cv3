import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.rnn_baseline import BaselineVideoPredictor
from utils.dataset import FramePredictionDataset, get_sequence_dirs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weighted_mse_loss(pred, target, fg_weight=10.0, threshold=0.5):
    """
    Give more weight to foreground pixels.
    Assumes inverted images if invert=True during dataset loading:
    - background near 0
    - ball near 1
    """
    with torch.no_grad():
        weights = torch.ones_like(target)
        weights[target > threshold] = fg_weight

    loss = weights * (pred - target) ** 2
    return loss.mean()


def build_dataloaders(args):
    sequence_dirs = get_sequence_dirs(args.data_dir)

    dataset = FramePredictionDataset(
        sequence_dirs=sequence_dirs,
        context=args.context,
        rollout=1,
        stride=args.stride,
        grayscale=True,
        invert=args.invert,
        return_state=False
    )

    if len(dataset) == 0:
        raise ValueError(f"No valid samples found in {args.data_dir}")

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size

    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


def compute_loss(pred_next, target_next, args):
    if args.loss_type == "mse":
        return nn.functional.mse_loss(pred_next, target_next)
    elif args.loss_type == "l1":
        return nn.functional.l1_loss(pred_next, target_next)
    elif args.loss_type == "weighted_mse":
        return weighted_mse_loss(
            pred_next,
            target_next,
            fg_weight=args.fg_weight,
            threshold=args.fg_threshold
        )
    else:
        raise ValueError(f"Unsupported loss_type: {args.loss_type}")


def train_one_epoch(model, dataloader, optimizer, device, args):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for input_seq, target_seq in pbar:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        target_next = target_seq[:, 0]

        optimizer.zero_grad()

        pred_next = model(input_seq)
        loss = compute_loss(pred_next, target_next, args)

        loss.backward()

        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.5f}"})

    return running_loss / len(dataloader)


@torch.no_grad()
def validate_one_epoch(model, dataloader, device, args):
    model.eval()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Validation", leave=False)

    for input_seq, target_seq in pbar:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        target_next = target_seq[:, 0]

        pred_next = model(input_seq)
        loss = compute_loss(pred_next, target_next, args)

        running_loss += loss.item()
        pbar.set_postfix({"val_loss": f"{loss.item():.5f}"})

    return running_loss / len(dataloader)


def save_checkpoint(path, model, optimizer, epoch, train_loss, val_loss, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        },
        path,
    )


def main(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )   
    print(f"Using device: {device}")
    print(f"Data dir: {args.data_dir}")

    train_loader, val_loader = build_dataloaders(args)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    model = BaselineVideoPredictor(
        input_channels=1,
        encoder_channels=args.encoder_channels,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        norm_type=args.norm_type,
        norm_groups=args.norm_groups,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            args=args,
        )

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")

        save_checkpoint(
            path=os.path.join(save_dir, "last.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            args=args,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                path=os.path.join(save_dir, "best.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                args=args,
            )
            print(f"Saved new best checkpoint to {os.path.join(save_dir, 'best.pt')}")

    print("\nTraining completed.")
    print(f"Checkpoints saved in: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train recurrent baseline for bouncing ball video prediction")

    parser.add_argument("--data_dir", type=str, default="data/physics-data-id")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default="baseline_convgru")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--encoder_channels", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, nargs="+", default=[64])

    parser.add_argument("--norm_type", type=str, default=None, choices=[None, "group", "layer"])
    parser.add_argument("--norm_groups", type=int, default=None)

    parser.add_argument("--invert", action="store_true", help="Invert images so the ball becomes bright on dark background")

    parser.add_argument("--loss_type", type=str, default="weighted_mse", choices=["mse", "l1", "weighted_mse"])
    parser.add_argument("--fg_weight", type=float, default=15.0)
    parser.add_argument("--fg_threshold", type=float, default=0.5)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)