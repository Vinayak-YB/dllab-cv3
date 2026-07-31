import argparse
import os
import pandas as pd

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from models.latent_flow_video_predictor import LatentFlowVideoPredictor
from utils.dataset import FramePredictionDataset, get_sequence_dirs


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
        invert=args_dict.get("invert", False),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, args_dict


class FixedRolloutDataset(Dataset):
    def __init__(self, sequence_dirs, context=5, rollout_steps=10, grayscale=True, invert=False, stride=1):
        self.samples = []
        base_dataset = FramePredictionDataset(
            sequence_dirs=sequence_dirs,
            context=context,
            rollout=rollout_steps,
            stride=stride,
            grayscale=grayscale,
            invert=invert,
            return_state=True,
        )
        for i in range(len(base_dataset)):
            sample = base_dataset[i]
            if sample[1].shape[0] == rollout_steps and sample[2].shape[0] == rollout_steps:
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


@torch.no_grad()
def extract_trajectories(model, dataloader, device, rollout_steps, fm_steps):
    all_pred_states, all_gt_states = [], []

    for batch in tqdm.tqdm(dataloader, desc="Extracting Latent Physics"):
        input_seq, target_seq, pos, vel = batch
        input_seq = input_seq.to(device)

        current_context = input_seq.clone()
        rollout_states = []

        for _ in range(rollout_steps):
            rep = model.get_context_representation(current_context)
            state_pred = model.state_head(rep["context"])
            rollout_states.append(state_pred.cpu())

            pred_next = model.predict_next_frame(current_context, num_steps=fm_steps)
            current_context = torch.cat([current_context[:, 1:], pred_next.unsqueeze(1)], dim=1)

        pred_seq = torch.stack(rollout_states, dim=1)
        gt_seq = torch.cat([pos, vel], dim=-1).float()

        all_pred_states.append(pred_seq)
        all_gt_states.append(gt_seq)

    return torch.cat(all_pred_states, dim=0), torch.cat(all_gt_states, dim=0)


def evaluate_bouncing_ball(pred_states, gt_states):
    def get_accel_medians(states):
        v = states[..., 2:4]
        a = v[:, 1:] - v[:, :-1]
        return torch.median(a[..., 0]).item(), torch.median(a[..., 1]).item()

    gt_ax, gt_ay = get_accel_medians(gt_states)
    pred_ax, pred_ay = get_accel_medians(pred_states)

    results = {
        "Metric": ["Horizontal Acceleration (a_x)", "Vertical Acceleration / Gravity (a_y)"],
        "Ground Truth": [gt_ax, gt_ay],
        "Predicted (Latent)": [pred_ax, pred_ay],
        "Absolute Error": [abs(pred_ax - gt_ax), abs(pred_ay - gt_ay)],
    }
    return results


def evaluate_billiard(pred_states, gt_states):
    def get_momentum_std(states):
        v = states[..., 4:8]
        px = v[..., 0] + v[..., 2]
        py = v[..., 1] + v[..., 3]
        std_px = torch.mean(torch.std(px, dim=1)).item()
        std_py = torch.mean(torch.std(py, dim=1)).item()
        return std_px, std_py

    gt_std_px, gt_std_py = get_momentum_std(gt_states)
    pred_std_px, pred_std_py = get_momentum_std(pred_states)

    results = {
        "Metric": ["Momentum Variation P_x (Std Dev)", "Momentum Variation P_y (Std Dev)"],
        "Ground Truth": [gt_std_px, gt_std_py],
        "Predicted (Latent)": [pred_std_px, pred_std_py],
        "Difference": [abs(pred_std_px - gt_std_px), abs(pred_std_py - gt_std_py)],
    }
    return results


def evaluate_magnetic_wells(pred_states, gt_states):
    pred_v = pred_states[..., 2:4]
    gt_v = gt_states[..., 2:4]

    pred_a = pred_v[:, 1:] - pred_v[:, :-1]
    gt_a = gt_v[:, 1:] - gt_v[:, :-1]

    mse_a = torch.nn.functional.mse_loss(pred_a, gt_a).item()
    var_gt = torch.var(gt_a).item()
    r2_a = 1.0 - (mse_a / (var_gt + 1e-8))

    results = {
        "Metric": ["Force Field Accuracy (R^2 Score)", "Mean Squared Error (MSE)"],
        "Ground Truth": [1.0, 0.0],
        "Predicted (Latent)": [r2_a, mse_a],
        "Notes": ["(1.0 is optimal)", "(0.0 is optimal)"],
    }
    return results


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    model, ckpt_args = load_model(args.model_dir, args.ckpt_name, device)
    context = ckpt_args.get("context", args.context)
    invert = ckpt_args.get("invert", args.invert)
    grayscale = ckpt_args.get("grayscale", True)

    sequence_dirs = get_sequence_dirs(args.data_dir)
    print(f"Loaded {len(sequence_dirs)} sequences from {args.data_dir}")

    dataset = FixedRolloutDataset(
        sequence_dirs=sequence_dirs,
        context=context,
        rollout_steps=args.rollout_steps,
        grayscale=grayscale,
        invert=invert,
        stride=args.eval_stride,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    pred_states, gt_states = extract_trajectories(model, dataloader, device, args.rollout_steps, args.fm_steps)

    print("\n" + "=" * 60)
    print(f"REVERSE ENGINEERING PHYSICS: {args.env.upper()}")
    print("=" * 60)

    if args.env == "bouncing_ball":
        results = evaluate_bouncing_ball(pred_states, gt_states)
    elif args.env == "billiard":
        results = evaluate_billiard(pred_states, gt_states)
    elif args.env == "magnetic_wells":
        results = evaluate_magnetic_wells(pred_states, gt_states)
    else:
        raise ValueError(f"Unknown environment: {args.env}")

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("=" * 60 + "\n")

    out_csv = os.path.join(args.model_dir, f"physics_probe_results_{args.env}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Tabular report saved to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reverse Engineer Physical Laws from Model Latent Space")
    parser.add_argument("--env", type=str, required=True, choices=["bouncing_ball", "billiard", "magnetic_wells"])
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="best.pt")
    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--invert", action="store_true")

    parser.add_argument("--rollout_steps", type=int, default=15)
    parser.add_argument("--fm_steps", type=int, default=20)
    parser.add_argument("--eval_stride", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    main(args)
