import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import torch
from torch.utils.data import DataLoader

from models.rnn_baseline import BaselineVideoPredictor
from utils.dataset import FramePredictionDataset, TrajectoryPredictionDataset, get_sequence_dirs


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_dir, ckpt_name, device):
    ckpt_path = os.path.join(model_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device)

    args_dict = ckpt.get("args", {})

    model = BaselineVideoPredictor(
        input_channels=1,
        encoder_channels=args_dict.get("encoder_channels", 32),
        hidden_channels=args_dict.get("hidden_channels", [64]),
        kernel_size=args_dict.get("kernel_size", 3),
        norm_type=args_dict.get("norm_type", None),
        norm_groups=args_dict.get("norm_groups", None),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, args_dict


def split_sequence_dirs(sequence_dirs, val_ratio=0.1):
    val_size = max(1, int(len(sequence_dirs) * val_ratio))
    train_dirs = sequence_dirs[:-val_size]
    val_dirs = sequence_dirs[-val_size:]
    return train_dirs, val_dirs


@torch.no_grad()
def get_frame_predictions(model, dataloader, device, max_samples=64):
    all_targets = []
    all_preds = []

    for input_seq, target_seq in dataloader:
        input_seq = input_seq.to(device)
        pred_next = model(input_seq).clamp(0, 1).cpu()
        target_next = target_seq[:, 0].cpu()

        all_preds.append(pred_next)
        all_targets.append(target_next)

        total = sum(x.shape[0] for x in all_preds)
        if total >= max_samples:
            break

    preds = torch.cat(all_preds, dim=0)[:max_samples]
    targets = torch.cat(all_targets, dim=0)[:max_samples]
    return preds, targets


@torch.no_grad()
def get_trajectory_predictions(model, dataloader, device, max_frames=100):
    all_targets = []
    all_preds = []

    for input_seq, target_seq in dataloader:
        input_seq = input_seq.to(device)
        pred_steps = target_seq.shape[1]

        rollout_preds = []
        current_context = input_seq.clone()

        for _ in range(pred_steps):
            pred_next = model(current_context).clamp(0, 1)
            rollout_preds.append(pred_next.cpu())
            current_context = torch.cat([current_context[:, 1:], pred_next.unsqueeze(1)], dim=1)

        preds = torch.cat(rollout_preds, dim=0)

        all_preds.append(preds)
        all_targets.append(target_seq.squeeze(0).cpu())

        total = sum(x.shape[0] for x in all_preds)
        if total >= max_frames:
            break

    preds = torch.cat(all_preds, dim=0)[:max_frames]
    targets = torch.cat(all_targets, dim=0)[:max_frames]
    return preds, targets


def save_comparison_video(pred_frames, target_frames, save_path, fps=10, title="Prediction vs Target"):
    pred_np = pred_frames.squeeze(1).numpy()
    target_np = target_frames.squeeze(1).numpy()

    n_frames = min(len(pred_np), len(target_np))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(title)

    ax_pred, ax_target = axes
    ax_pred.set_title("Prediction")
    ax_target.set_title("Target")

    im_pred = ax_pred.imshow(pred_np[0], cmap="gray", vmin=0, vmax=1, animated=True)
    im_target = ax_target.imshow(target_np[0], cmap="gray", vmin=0, vmax=1, animated=True)

    for ax in axes:
        ax.axis("off")

    frame_text = fig.text(0.5, 0.02, "Frame 0", ha="center")

    def update(i):
        im_pred.set_array(pred_np[i])
        im_target.set_array(target_np[i])
        frame_text.set_text(f"Frame {i}")
        return [im_pred, im_target, frame_text]

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"artist": "Federico"}, bitrate=1800)
    ani.save(save_path, writer=writer)
    plt.close(fig)


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    model, ckpt_args = load_model(args.model_dir, args.ckpt_name, device)

    context = ckpt_args.get("context", args.context)
    invert = ckpt_args.get("invert", args.invert)

    sequence_dirs = get_sequence_dirs(args.data_dir)
    _, val_dirs = split_sequence_dirs(sequence_dirs, val_ratio=args.val_ratio)

    if args.mode == "frame":
        dataset = FramePredictionDataset(
            sequence_dirs=val_dirs,
            context=context,
            rollout=1,
            stride=1,
            grayscale=True,
            invert=invert,
            return_state=False
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        pred_frames, target_frames = get_frame_predictions(
            model=model,
            dataloader=dataloader,
            device=device,
            max_samples=args.max_frames
        )
    elif args.mode == "trajectory":
        dataset = TrajectoryPredictionDataset(
            sequence_dirs=val_dirs,
            context=context,
            grayscale=True,
            invert=invert,
            return_state=False
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
        pred_frames, target_frames = get_trajectory_predictions(
            model=model,
            dataloader=dataloader,
            device=device,
            max_frames=args.max_frames
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    print("Pred frames shape:", pred_frames.shape)
    print("Target frames shape:", target_frames.shape)
    print("Pred min/max:", pred_frames.min().item(), pred_frames.max().item())
    print("Target min/max:", target_frames.min().item(), target_frames.max().item())
    print("Invert:", invert)

    save_dir = os.path.join(args.model_dir, "videos")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{args.mode}_{os.path.splitext(args.ckpt_name)[0]}.mp4")
    save_comparison_video(
        pred_frames=pred_frames,
        target_frames=target_frames,
        save_path=save_path,
        fps=args.fps,
        title=f"{args.mode.capitalize()} prediction"
    )

    print(f"Saved video to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create qualitative comparison videos for video prediction")

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="best.pt")
    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--mode", type=str, choices=["frame", "trajectory"], default="frame")
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--invert", action="store_true")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    main(args)