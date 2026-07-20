import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset

from models.state_model import StateMLP


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def split_sequence_dirs(sequence_dirs, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    sequence_dirs = sequence_dirs.copy()
    rng.shuffle(sequence_dirs)

    val_size = max(1, int(len(sequence_dirs) * val_ratio))
    train_dirs = sequence_dirs[:-val_size]
    val_dirs = sequence_dirs[-val_size:]
    return train_dirs, val_dirs


class OneStepStateDataset(Dataset):
    def __init__(self, sequence_dirs, context=5):
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

            if len(states) < context + 1:
                continue

            for start_idx in range(len(states) - context):
                input_seq = states[start_idx:start_idx + context]
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


class RolloutStateDataset(Dataset):
    def __init__(self, sequence_dirs, context=5, rollout_steps=10):
        self.samples = []
        self.context = context
        self.rollout_steps = rollout_steps

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
                input_seq = states[start_idx:start_idx + context]
                future_seq = states[start_idx + context:start_idx + context + rollout_steps]
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
def evaluate_one_step(model, dataloader, device):
    pred_states = []
    target_states = []

    for input_seq, target_state in dataloader:
        input_seq = input_seq.to(device)
        pred_state = model(input_seq)

        pred_states.append(pred_state.cpu())
        target_states.append(target_state)

    pred_states = torch.cat(pred_states, dim=0)
    target_states = torch.cat(target_states, dim=0)

    return pred_states, target_states


@torch.no_grad()
def evaluate_rollout(model, dataloader, device, rollout_steps):
    all_pred = []
    all_target = []

    for input_seq, future_seq in dataloader:
        input_seq = input_seq.to(device)
        future_seq = future_seq.to(device)

        context = input_seq.clone()
        preds = []

        for _ in range(rollout_steps):
            pred_next = model(context)
            preds.append(pred_next)
            context = torch.cat([context[:, 1:], pred_next.unsqueeze(1)], dim=1)

        preds = torch.stack(preds, dim=1)
        all_pred.append(preds.cpu())
        all_target.append(future_seq.cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)

    return all_pred, all_target


def compute_flat_metrics(pred_states, target_states):
    pred_pos = pred_states[:, :2]
    target_pos = target_states[:, :2]

    pred_vel = pred_states[:, 2:]
    target_vel = target_states[:, 2:]

    pos_aee = torch.norm(pred_pos - target_pos, dim=1).mean().item()
    vel_aee = torch.norm(pred_vel - target_vel, dim=1).mean().item()

    pos_r2 = r2_score(target_pos.numpy(), pred_pos.numpy())
    vel_r2 = r2_score(target_vel.numpy(), pred_vel.numpy())

    return {
        "position_aee": pos_aee,
        "velocity_aee": vel_aee,
        "position_r2": pos_r2,
        "velocity_r2": vel_r2,
    }


def compute_rollout_metrics(pred_seq, target_seq):
    _, h, d = pred_seq.shape

    flat_metrics = compute_flat_metrics(
        pred_seq.reshape(-1, d),
        target_seq.reshape(-1, d),
    )

    per_step = {}
    for step in range(h):
        pred_step = pred_seq[:, step, :]
        target_step = target_seq[:, step, :]

        pred_pos = pred_step[:, :2]
        target_pos = target_step[:, :2]

        pred_vel = pred_step[:, 2:]
        target_vel = target_step[:, 2:]

        per_step[str(step + 1)] = {
            "position_aee": torch.norm(pred_pos - target_pos, dim=1).mean().item(),
            "velocity_aee": torch.norm(pred_vel - target_vel, dim=1).mean().item(),
            "position_r2": r2_score(target_pos.numpy(), pred_pos.numpy()),
            "velocity_r2": r2_score(target_vel.numpy(), pred_vel.numpy()),
        }

    return {
        "aggregate": flat_metrics,
        "per_step": per_step,
    }


def resolve_eval_dirs(args):
    eval_dirs = get_sequence_dirs(args.data_dir)
    if len(eval_dirs) == 0:
        raise ValueError(f"No valid sequences found in eval data dir: {args.data_dir}")

    if args.probe_train_dir is not None:
        probe_train_dirs = get_sequence_dirs(args.probe_train_dir)
        if len(probe_train_dirs) == 0:
            raise ValueError(f"No valid sequences found in probe train dir: {args.probe_train_dir}")
        split_mode = "explicit_eval_dir"
    else:
        _, eval_dirs = split_sequence_dirs(
            eval_dirs,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        probe_train_dirs = None
        split_mode = "legacy_internal_split"

    return probe_train_dirs, eval_dirs, split_mode


def main(args):
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    model, ckpt_args = load_model(args.model_dir, args.ckpt_name, device)
    context = ckpt_args.get("context", args.context)

    probe_train_dirs, eval_dirs, split_mode = resolve_eval_dirs(args)

    print(f"Evaluation split mode: {split_mode}")
    if probe_train_dirs is not None:
        print(f"Probe-train trajectories: {len(probe_train_dirs)}")
    print(f"Eval trajectories:        {len(eval_dirs)}")

    one_step_dataset = OneStepStateDataset(eval_dirs, context=context)
    one_step_loader = DataLoader(
        one_step_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    rollout_dataset = RolloutStateDataset(
        eval_dirs,
        context=context,
        rollout_steps=args.rollout_steps,
    )
    rollout_loader = DataLoader(
        rollout_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"One-step samples: {len(one_step_dataset)}")
    print(f"Rollout samples:  {len(rollout_dataset)}")

    results = {
        "metadata": {
            "model_dir": args.model_dir,
            "ckpt_name": args.ckpt_name,
            "data_dir": args.data_dir,
            "probe_train_dir": args.probe_train_dir,
            "split_mode": split_mode,
            "context": context,
            "batch_size": args.batch_size,
            "rollout_steps": args.rollout_steps,
            "seed": args.seed,
        }
    }

    pred_one, tgt_one = evaluate_one_step(model, one_step_loader, device)
    one_step_metrics = compute_flat_metrics(pred_one, tgt_one)
    results["one_step"] = one_step_metrics

    print("\n=== One-step ===")
    print(f"Position AEE: {one_step_metrics['position_aee']:.4f}")
    print(f"Velocity AEE: {one_step_metrics['velocity_aee']:.4f}")
    print(f"Position R2:  {one_step_metrics['position_r2']:.4f}")
    print(f"Velocity R2:  {one_step_metrics['velocity_r2']:.4f}")

    pred_roll, tgt_roll = evaluate_rollout(model, rollout_loader, device, args.rollout_steps)
    rollout_metrics = compute_rollout_metrics(pred_roll, tgt_roll)
    results["rollout"] = rollout_metrics

    agg = rollout_metrics["aggregate"]
    print("\n=== Rollout ===")
    print(f"Position AEE: {agg['position_aee']:.4f}")
    print(f"Velocity AEE: {agg['velocity_aee']:.4f}")
    print(f"Position R2:  {agg['position_r2']:.4f}")
    print(f"Velocity R2:  {agg['velocity_r2']:.4f}")

    print("\n=== Rollout per step ===")
    for step, metrics in rollout_metrics["per_step"].items():
        print(
            f"Step {step}: "
            f"PosAEE={metrics['position_aee']:.4f}, "
            f"VelAEE={metrics['velocity_aee']:.4f}, "
            f"PosR2={metrics['position_r2']:.4f}, "
            f"VelR2={metrics['velocity_r2']:.4f}"
        )

    split_name = os.path.basename(os.path.normpath(args.data_dir))
    save_dir = os.path.join(args.model_dir, "physics")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"evaluation_results_{split_name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved results to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate explicit state model on one-step and long rollouts")

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="best.pt")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--probe_train_dir",
        type=str,
        default=None,
        help="Optional training split path for protocol consistency. Not used for fitting probes here, but logged."
    )

    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--rollout_steps", type=int, default=10)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)