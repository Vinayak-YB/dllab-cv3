import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.latent_flow_video_predictor import LatentFlowVideoPredictor
from utils.dataset import FramePredictionDataset, get_sequence_dirs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(args):
    sequence_dirs = get_sequence_dirs(args.data_dir)

    dataset = FramePredictionDataset(
        sequence_dirs=sequence_dirs,
        context=args.context,
        rollout=1,
        stride=args.stride,
        grayscale=args.grayscale,
        invert=args.invert,
        return_state=True,
    )

    if len(dataset) == 0:
        raise ValueError(f"No valid samples found in {args.data_dir}")

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size

    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def build_target_state(target_positions, target_velocities):
    """
    target_positions: (B, 1, 2)
    target_velocities: (B, 1, 2)
    returns: (B, 4) = [x, y, vx, vy]
    """
    pos_next = target_positions[:, 0]
    vel_next = target_velocities[:, 0]
    target_state = torch.cat([pos_next, vel_next], dim=1)
    return target_state


def train_one_epoch(model, dataloader, optimizer, device, args):
    model.train()

    running_loss = 0.0
    running_flow_loss = 0.0
    running_recon_loss = 0.0
    running_motion_loss = 0.0
    running_state_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for input_seq, target_seq, target_positions, target_velocities in pbar:
        input_seq = input_seq.to(device)                    # (B, T, C, H, W)
        target_seq = target_seq.to(device)                  # (B, 1, C, H, W)
        target_positions = target_positions.to(device)      # (B, 1, 2)
        target_velocities = target_velocities.to(device)    # (B, 1, 2)

        target_next = target_seq[:, 0]                      # (B, C, H, W)
        target_state = build_target_state(target_positions, target_velocities)

        optimizer.zero_grad()

        losses = model.compute_loss(
            context_frames=input_seq,
            target_frame=target_next,
            target_state=target_state,
        )
        loss = losses["loss"]

        loss.backward()

        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        running_loss += loss.item()
        running_flow_loss += losses["flow_loss"].item()
        running_recon_loss += losses["recon_loss"].item()
        running_motion_loss += losses["motion_loss"].item()
        running_state_loss += losses["state_loss"].item()

        pbar.set_postfix({
            "loss": f"{loss.item():.5f}",
            "flow": f"{losses['flow_loss'].item():.5f}",
            "recon": f"{losses['recon_loss'].item():.5f}",
            "motion": f"{losses['motion_loss'].item():.5f}",
            "state": f"{losses['state_loss'].item():.5f}",
        })

    n = len(dataloader)
    return {
        "loss": running_loss / n,
        "flow_loss": running_flow_loss / n,
        "recon_loss": running_recon_loss / n,
        "motion_loss": running_motion_loss / n,
        "state_loss": running_state_loss / n,
    }


@torch.no_grad()
def validate_one_epoch(model, dataloader, device, args):
    model.eval()

    running_loss = 0.0
    running_flow_loss = 0.0
    running_recon_loss = 0.0
    running_motion_loss = 0.0
    running_state_loss = 0.0

    pbar = tqdm(dataloader, desc="Validation", leave=False)

    for input_seq, target_seq, target_positions, target_velocities in pbar:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        target_positions = target_positions.to(device)
        target_velocities = target_velocities.to(device)

        target_next = target_seq[:, 0]
        target_state = build_target_state(target_positions, target_velocities)

        losses = model.compute_loss(
            context_frames=input_seq,
            target_frame=target_next,
            target_state=target_state,
        )
        loss = losses["loss"]

        running_loss += loss.item()
        running_flow_loss += losses["flow_loss"].item()
        running_recon_loss += losses["recon_loss"].item()
        running_motion_loss += losses["motion_loss"].item()
        running_state_loss += losses["state_loss"].item()

        pbar.set_postfix({
            "val_loss": f"{loss.item():.5f}",
            "val_flow": f"{losses['flow_loss'].item():.5f}",
            "val_recon": f"{losses['recon_loss'].item():.5f}",
            "val_motion": f"{losses['motion_loss'].item():.5f}",
            "val_state": f"{losses['state_loss'].item():.5f}",
        })

    n = len(dataloader)
    return {
        "loss": running_loss / n,
        "flow_loss": running_flow_loss / n,
        "recon_loss": running_recon_loss / n,
        "motion_loss": running_motion_loss / n,
        "state_loss": running_state_loss / n,
    }


def save_checkpoint(path, model, optimizer, epoch, train_metrics, val_metrics, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "args": vars(args),
        },
        path,
    )


def main(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available()
        else "cuda" if torch.cuda.is_available()
        else "mps" if args.device == "mps" and torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Data dir: {args.data_dir}")

    train_loader, val_loader = build_dataloaders(args)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    input_channels = 1 if args.grayscale else 3

    model = LatentFlowVideoPredictor(
        input_channels=input_channels,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        context_frames=args.context,
        time_dim=args.time_dim,
        dynamics_hidden_channels=args.dynamics_hidden_channels,
        state_loss_weight=args.state_loss_weight,
        recon_loss_weight=args.recon_loss_weight,
        motion_loss_weight=args.motion_loss_weight,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            args=args,
        )

        print(
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Flow: {train_metrics['flow_loss']:.6f} | "
            f"Recon: {train_metrics['recon_loss']:.6f} | "
            f"Motion: {train_metrics['motion_loss']:.6f} | "
            f"State: {train_metrics['state_loss']:.6f}"
        )
        print(
            f"Val Loss:   {val_metrics['loss']:.6f} | "
            f"Flow: {val_metrics['flow_loss']:.6f} | "
            f"Recon: {val_metrics['recon_loss']:.6f} | "
            f"Motion: {val_metrics['motion_loss']:.6f} | "
            f"State: {val_metrics['state_loss']:.6f}"
        )

        save_checkpoint(
            path=os.path.join(save_dir, "last.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            args=args,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                path=os.path.join(save_dir, "best.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                args=args,
            )
            print(f"Saved new best checkpoint to {os.path.join(save_dir, 'best.pt')}")

    print("\nTraining completed.")
    print(f"Checkpoints saved in: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train improved latent flow video predictor for bouncing ball")

    parser.add_argument("--data_dir", type=str, default="data/bouncing_ball/train")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default="latent_flow_v2")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--latent_channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=64)
    parser.add_argument("--dynamics_hidden_channels", type=int, default=64)

    parser.add_argument("--recon_loss_weight", type=float, default=0.2)
    parser.add_argument("--motion_loss_weight", type=float, default=0.1)
    parser.add_argument("--state_loss_weight", type=float, default=0.1)

    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--invert", action="store_true")

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])

    args = parser.parse_args()
    main(args)