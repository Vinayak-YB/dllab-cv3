import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
from torch.utils.data import DataLoader

from models.latent_flow_video_predictor import LatentFlowVideoPredictor
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

    input_channels = 1 if args_dict.get("grayscale", True) else 3

    model = LatentFlowVideoPredictor(
        input_channels=input_channels,
        base_channels=args_dict.get("base_channels", 32),
        latent_channels=args_dict.get("latent_channels", 64),
        context_frames=args_dict.get("context", 5),
        time_dim=args_dict.get("time_dim", 64),
        dynamics_hidden_channels=args_dict.get("dynamics_hidden_channels", 64),
        state_loss_weight=args_dict.get("state_loss_weight", 0.1),
        recon_loss_weight=args_dict.get("recon_loss_weight", 0.2),
        motion_loss_weight=args_dict.get("motion_loss_weight", 0.1),
        generated_frame_loss_weight=args_dict.get("generated_frame_loss_weight", 0.0),
        generation_loss_steps=args_dict.get("generation_loss_steps", 5),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, args_dict


@torch.no_grad()
def get_frame_predictions(model, dataloader, device, fm_steps, max_samples=64):
    all_targets = []
    all_preds = []

    for batch in dataloader:
        if len(batch) == 4:
            input_seq, target_seq, _, _ = batch
        else:
            input_seq, target_seq = batch

        input_seq = input_seq.to(device)

        pred_next = model.predict_next_frame(
            input_seq,
            num_steps=fm_steps,
        ).clamp(0, 1).cpu()

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
def get_trajectory_predictions(model, dataloader, device, fm_steps, max_frames=100):
    all_targets = []
    all_preds = []

    for batch in dataloader:
        if len(batch) == 4:
            input_seq, target_seq, _, _ = batch
        else:
            input_seq, target_seq = batch

        input_seq = input_seq.to(device)
        pred_steps = target_seq.shape[1]

        rollout_preds = []
        current_context = input_seq.clone()

        for _ in range(pred_steps):
            pred_next = model.predict_next_frame(
                current_context,
                num_steps=fm_steps,
            ).clamp(0, 1)

            rollout_preds.append(pred_next.cpu())

            current_context = torch.cat(
                [current_context[:, 1:], pred_next.unsqueeze(1)],
                dim=1,
            )

        # batch_size should be 1 in trajectory mode
        preds = torch.cat(rollout_preds, dim=0)

        all_preds.append(preds)
        all_targets.append(target_seq.squeeze(0).cpu())

        total = sum(x.shape[0] for x in all_preds)
        if total >= max_frames:
            break

    preds = torch.cat(all_preds, dim=0)[:max_frames]
    targets = torch.cat(all_targets, dim=0)[:max_frames]
    return preds, targets


def frames_to_numpy(frames):
    frames = frames.detach().cpu()

    if frames.dim() != 4:
        raise ValueError(f"Expected frames with shape (N, C, H, W), got {frames.shape}")

    if frames.shape[1] == 1:
        return frames.squeeze(1).numpy(), "gray"

    if frames.shape[1] == 3:
        return frames.permute(0, 2, 3, 1).numpy(), None

    raise ValueError(f"Unsupported number of channels: {frames.shape[1]}")


def save_comparison_video(pred_frames, target_frames, save_path, fps=10, title="Latent flow prediction"):
    pred_np, pred_cmap = frames_to_numpy(pred_frames)
    target_np, target_cmap = frames_to_numpy(target_frames)

    n_frames = min(len(pred_np), len(target_np))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(title)

    ax_pred, ax_target = axes
    ax_pred.set_title("Prediction")
    ax_target.set_title("Target")

    if pred_cmap == "gray":
        im_pred = ax_pred.imshow(pred_np[0], cmap="gray", vmin=0, vmax=1, animated=True)
    else:
        im_pred = ax_pred.imshow(pred_np[0], vmin=0, vmax=1, animated=True)

    if target_cmap == "gray":
        im_target = ax_target.imshow(target_np[0], cmap="gray", vmin=0, vmax=1, animated=True)
    else:
        im_target = ax_target.imshow(target_np[0], vmin=0, vmax=1, animated=True)

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
    writer = PillowWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close(fig)


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    model, ckpt_args = load_model(args.model_dir, args.ckpt_name, device)

    context = ckpt_args.get("context", args.context)
    invert = ckpt_args.get("invert", args.invert)
    grayscale = ckpt_args.get("grayscale", args.grayscale)

    print(f"Context: {context}")
    print(f"Invert: {invert}")
    print(f"Grayscale: {grayscale}")
    print(f"FM steps: {args.fm_steps}")

    sequence_dirs = get_sequence_dirs(args.data_dir)

    if args.mode == "frame":
        dataset = FramePredictionDataset(
            sequence_dirs=sequence_dirs,
            context=context,
            rollout=1,
            stride=args.stride,
            grayscale=grayscale,
            invert=invert,
            return_state=False,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        pred_frames, target_frames = get_frame_predictions(
            model=model,
            dataloader=dataloader,
            device=device,
            fm_steps=args.fm_steps,
            max_samples=args.max_frames,
        )

    elif args.mode == "trajectory":
        dataset = TrajectoryPredictionDataset(
            sequence_dirs=sequence_dirs,
            context=context,
            grayscale=grayscale,
            invert=invert,
            return_state=False,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )

        pred_frames, target_frames = get_trajectory_predictions(
            model=model,
            dataloader=dataloader,
            device=device,
            fm_steps=args.fm_steps,
            max_frames=args.max_frames,
        )

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    print("Pred frames shape:", pred_frames.shape)
    print("Target frames shape:", target_frames.shape)
    print("Pred min/max:", pred_frames.min().item(), pred_frames.max().item())
    print("Target min/max:", target_frames.min().item(), target_frames.max().item())

    save_dir = os.path.join(args.model_dir, "videos")
    os.makedirs(save_dir, exist_ok=True)

    split_name = os.path.basename(os.path.normpath(args.data_dir))
    ckpt_base = os.path.splitext(args.ckpt_name)[0]

    save_path = os.path.join(
        save_dir,
        f"latent_flow_{args.mode}_{ckpt_base}_{split_name}.gif",
    )

    save_comparison_video(
        pred_frames=pred_frames,
        target_frames=target_frames,
        save_path=save_path,
        fps=args.fps,
        title=f"Latent flow {args.mode} prediction",
    )

    print(f"Saved video to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create qualitative comparison videos for latent flow video prediction"
    )

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="best.pt")
    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--mode", type=str, choices=["frame", "trajectory"], default="trajectory")
    parser.add_argument("--context", type=int, default=5)

    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--invert", action="store_true")

    parser.add_argument("--fm_steps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    main(args)
