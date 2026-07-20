import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.state_model import StateMLP


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

    model = StateMLP(
        context=args_dict.get("context", 5),
        state_dim=4,
        hidden_dim=args_dict.get("hidden_dim", 128),
        num_layers=args_dict.get("num_layers", 3),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, args_dict


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


def split_sequence_dirs(sequence_dirs, val_ratio=0.1):
    val_size = max(1, int(len(sequence_dirs) * val_ratio))
    train_dirs = sequence_dirs[:-val_size]
    val_dirs = sequence_dirs[-val_size:]
    return train_dirs, val_dirs


class StateRolloutDataset(Dataset):
    def __init__(self, sequence_dirs, context=5, rollout_steps=20):
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

            states = np.concatenate([positions, velocities], axis=1)

            min_len = context + rollout_steps
            if len(states) < min_len:
                continue

            for start_idx in range(len(states) - context - rollout_steps + 1):
                input_seq = states[start_idx : start_idx + context]
                future_seq = states[start_idx + context : start_idx + context + rollout_steps]
                self.samples.append((input_seq, future_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, future_seq = self.samples[idx]
        return (
            torch.tensor(input_seq, dtype=torch.float32),
            torch.tensor(future_seq, dtype=torch.float32),
        )


@torch.no_grad()
def get_rollout_predictions(model, dataloader, device, max_sequences=1):
    all_preds = []
    all_targets = []

    count = 0
    for input_seq, future_seq in dataloader:
        input_seq = input_seq.to(device)
        future_seq = future_seq.to(device)

        context = input_seq.clone()
        preds = []

        rollout_steps = future_seq.shape[1]

        for _ in range(rollout_steps):
            pred_next = model(context)
            preds.append(pred_next)
            context = torch.cat([context[:, 1:], pred_next.unsqueeze(1)], dim=1)

        preds = torch.stack(preds, dim=1)  # (B, H, 4)

        all_preds.append(preds.cpu())
        all_targets.append(future_seq.cpu())

        count += input_seq.shape[0]
        if count >= max_sequences:
            break

    preds = torch.cat(all_preds, dim=0)[:max_sequences]
    targets = torch.cat(all_targets, dim=0)[:max_sequences]
    return preds, targets


def render_state_frame(state, width=128, height=128, radius=5):
    x, y = state[:2]
    yy, xx = np.ogrid[:height, :width]

    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    mask = dist2 <= radius ** 2

    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    image[mask] = np.array([0, 0, 255], dtype=np.uint8)

    return image


def save_state_comparison_video(pred_states, target_states, save_path, fps=10, width=128, height=128, radius=5):
    pred_states = pred_states[0].numpy()    # (H, 4)
    target_states = target_states[0].numpy()  # (H, 4)

    pred_frames = [render_state_frame(s, width, height, radius) for s in pred_states]
    target_frames = [render_state_frame(s, width, height, radius) for s in target_states]

    n_frames = min(len(pred_frames), len(target_frames))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle("State MLP rollout: Prediction vs Target")

    ax_pred, ax_target = axes
    ax_pred.set_title("Prediction")
    ax_target.set_title("Target")

    im_pred = ax_pred.imshow(pred_frames[0], animated=True)
    im_target = ax_target.imshow(target_frames[0], animated=True)

    for ax in axes:
        ax.axis("off")

    frame_text = fig.text(0.5, 0.02, "Frame 0", ha="center")

    def update(i):
        im_pred.set_array(pred_frames[i])
        im_target.set_array(target_frames[i])
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

    sequence_dirs = get_sequence_dirs(args.data_dir)
    _, val_dirs = split_sequence_dirs(sequence_dirs, val_ratio=args.val_ratio)

    dataset = StateRolloutDataset(
        sequence_dirs=val_dirs,
        context=context,
        rollout_steps=args.rollout_steps,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    pred_states, target_states = get_rollout_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        max_sequences=1,
    )

    print("Pred states shape:", pred_states.shape)
    print("Target states shape:", target_states.shape)
    print("Pred first state:", pred_states[0, 0])
    print("Target first state:", target_states[0, 0])

    save_dir = os.path.join(args.model_dir, "videos")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"state_rollout_{os.path.splitext(args.ckpt_name)[0]}.mp4")
    save_state_comparison_video(
        pred_states=pred_states,
        target_states=target_states,
        save_path=save_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
        radius=args.radius,
    )

    print(f"Saved video to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create qualitative comparison video for explicit state model")

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="best.pt")
    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--rollout_steps", type=int, default=20)

    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--radius", type=int, default=5)

    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    main(args)