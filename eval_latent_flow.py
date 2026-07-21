import argparse
import json
import os
import random

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from models.latent_flow_video_predictor import LatentFlowVideoPredictor
from utils.dataset import FramePredictionDataset, get_sequence_dirs


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
        invert=args_dict.get("invert", False),
        generated_frame_loss_weight=args_dict.get("generated_frame_loss_weight", 0.0),
        generation_loss_steps=args_dict.get("generation_loss_steps", 5),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, args_dict


def split_sequence_dirs(sequence_dirs, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    sequence_dirs = sequence_dirs.copy()
    rng.shuffle(sequence_dirs)

    val_size = max(1, int(len(sequence_dirs) * val_ratio))
    train_dirs = sequence_dirs[:-val_size]
    val_dirs = sequence_dirs[-val_size:]
    return train_dirs, val_dirs


class FixedRolloutDataset(Dataset):
    def __init__(self, sequence_dirs, context=5, rollout_steps=10, grayscale=True, invert=False, return_state=True, stride=1):
        self.samples = []

        base_dataset = FramePredictionDataset(
            sequence_dirs=sequence_dirs,
            context=context,
            rollout=rollout_steps,
            stride=stride,
            grayscale=grayscale,
            invert=invert,
            return_state=return_state,
        )

        for i in range(len(base_dataset)):
            sample = base_dataset[i]
            if return_state:
                input_seq, target_seq, pos, vel = sample
                if target_seq.shape[0] == rollout_steps and pos.shape[0] == rollout_steps and vel.shape[0] == rollout_steps:
                    self.samples.append(sample)
            else:
                input_seq, target_seq = sample
                if target_seq.shape[0] == rollout_steps:
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _to_single_channel(frame_tensor):
    if frame_tensor.dim() == 2:
        return frame_tensor
    if frame_tensor.dim() != 3:
        raise ValueError(f"Unexpected frame shape: {frame_tensor.shape}")
    if frame_tensor.shape[0] == 1:
        return frame_tensor[0]
    return frame_tensor.mean(dim=0)


def estimate_ball_center_robust(
    frame_tensor, invert=False, threshold=0.5, min_mass=3,
    fallback_thresholds=(0.4, 0.3, 0.2, 0.15, 0.1, 0.05),
    use_topk_fallback=True, topk_ratio=0.01, debug=False,
):
    frame = _to_single_channel(frame_tensor).float().clamp(0, 1)
    thresholds = [threshold] + [t for t in fallback_thresholds if t != threshold]

    h, w = frame.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=frame.device),
        torch.arange(w, dtype=torch.float32, device=frame.device),
        indexing="ij"
    )

    for thr in thresholds:
        # If invert=True, the ball is bright on a dark background.
        mask = frame > thr if invert else frame < thr
        mass = int(mask.sum().item())
        if mass >= min_mass:
            weights = mask.float()
            cx = (xs * weights).sum() / weights.sum()
            cy = (ys * weights).sum() / weights.sum()
            return torch.tensor([cx.item(), cy.item()], dtype=torch.float32)

    if use_topk_fallback:
        flat = frame.reshape(-1)
        k = max(3, int(topk_ratio * flat.numel()))
        k = min(k, flat.numel())
        # If invert=True, take the brightest pixels; otherwise take the darkest pixels.
        vals, idx = torch.topk(flat, k=k, largest=invert)

        if vals.numel() >= min_mass:
            yy = (idx // w).float()
            xx = (idx % w).float()
            cx = xx.mean()
            cy = yy.mean()
            return torch.tensor([cx.item(), cy.item()], dtype=torch.float32)

    return torch.tensor([torch.nan, torch.nan], dtype=torch.float32)


def extract_positions_from_frames_robust(frames, **kwargs):
    positions = [estimate_ball_center_robust(frame, **kwargs) for frame in frames]
    return torch.stack(positions, dim=0)


@torch.no_grad()
def frame_prediction(model, dataloader, device, fm_steps, extract_only=False):
    """If extract_only=True, skips image generation to save RAM and time."""
    target_frames, predicted_frames = [], []
    latent_states, state_preds, last_context_frames = [], [], []
    positions, velocities = [], []

    for batch in tqdm.tqdm(dataloader, desc="One-step eval" if not extract_only else "Latent Extraction"):
        if len(batch) == 4:
            input_seq, target_seq, pos, vel = batch
            positions.append(pos[:, 0])
            velocities.append(vel[:, 0])
        else:
            input_seq, target_seq = batch

        input_seq = input_seq.to(device)
        rep = model.get_context_representation(input_seq)
        context_latent = rep["context"]
        state_pred = model.state_head(context_latent)

        if not extract_only:
            target_next = target_seq[:, 0]
            pred_next = model.predict_next_frame(input_seq, num_steps=fm_steps)
            target_frames.append(target_next.cpu())
            predicted_frames.append(pred_next.clamp(0, 1).cpu())
            last_context_frames.append(input_seq[:, -1].cpu())

        latent_states.append(torch.nn.functional.adaptive_avg_pool2d(context_latent, (2, 2)).cpu())
        state_preds.append(state_pred.cpu())

    latent_states = torch.cat(latent_states, dim=0)
    state_preds = torch.cat(state_preds, dim=0)

    if positions:
        positions = torch.cat(positions, dim=0)
        velocities = torch.cat(velocities, dim=0)
    else:
        positions, velocities = None, None

    if extract_only:
        return None, None, latent_states, state_preds, positions, velocities, None

    target_frames = torch.cat(target_frames, dim=0)
    predicted_frames = torch.cat(predicted_frames, dim=0)
    last_context_frames = torch.cat(last_context_frames, dim=0)

    return target_frames, predicted_frames, latent_states, state_preds, positions, velocities, last_context_frames


@torch.no_grad()
def rollout_prediction(model, dataloader, device, rollout_steps, fm_steps):
    target_frames, predicted_frames = [], []
    latent_states, state_preds, last_context_frames = [], [], []
    positions, velocities = [], []

    for batch in tqdm.tqdm(dataloader, desc="Rollout evaluation"):
        if len(batch) == 4:
            input_seq, target_seq, pos, vel = batch
        else:
            input_seq, target_seq = batch
            pos, vel = None, None

        input_seq = input_seq.to(device)
        current_context = input_seq.clone()
        rollout_preds, rollout_latents, rollout_states = [], [], []

        for _ in range(rollout_steps):
            rep = model.get_context_representation(current_context)
            context_latent = rep["context"]
            state_pred = model.state_head(context_latent)
            pred_next = model.predict_next_frame(current_context, num_steps=fm_steps)

            rollout_preds.append(pred_next.clamp(0, 1).cpu())
            rollout_latents.append(torch.nn.functional.adaptive_avg_pool2d(context_latent, (2, 2)).cpu())
            rollout_states.append(state_pred.cpu())

            current_context = torch.cat([current_context[:, 1:], pred_next.unsqueeze(1)], dim=1)

        pred_seq = torch.stack(rollout_preds, dim=1)
        latent_seq = torch.stack(rollout_latents, dim=1)
        state_seq = torch.stack(rollout_states, dim=1)

        target_frames.append(target_seq.cpu())
        predicted_frames.append(pred_seq)
        latent_states.append(latent_seq)
        state_preds.append(state_seq)
        last_context_frames.append(input_seq[:, -1].cpu())

        if pos is not None and vel is not None:
            positions.append(pos.cpu())
            velocities.append(vel.cpu())

    target_frames = torch.cat(target_frames, dim=0)
    predicted_frames = torch.cat(predicted_frames, dim=0)
    latent_states = torch.cat(latent_states, dim=0)
    state_preds = torch.cat(state_preds, dim=0)
    last_context_frames = torch.cat(last_context_frames, dim=0)

    if positions:
        positions = torch.cat(positions, dim=0)
        velocities = torch.cat(velocities, dim=0)
    else:
        positions, velocities = None, None

    return target_frames, predicted_frames, latent_states, state_preds, positions, velocities, last_context_frames


def compute_state_metrics_one_step(state_preds, positions, velocities):
    pred_pos = state_preds[:, :2]
    pred_vel = state_preds[:, 2:]
    return {
        "position_r2": r2_score(positions.numpy(), pred_pos.numpy()),
        "velocity_r2": r2_score(velocities.numpy(), pred_vel.numpy()),
        "position_aee": torch.norm(pred_pos - positions, dim=1).mean().item(),
        "velocity_aee": torch.norm(pred_vel - velocities, dim=1).mean().item(),
    }


def compute_state_metrics_rollout(state_preds, positions, velocities):
    pred_pos = state_preds[:, :, :2]
    pred_vel = state_preds[:, :, 2:]
    per_step = {}
    for step in range(state_preds.shape[1]):
        step_metrics = {
            "position_r2": r2_score(positions[:, step].numpy(), pred_pos[:, step].numpy()),
            "position_aee": torch.norm(pred_pos[:, step] - positions[:, step], dim=1).mean().item(),
        }
        if step < velocities.shape[1]:
            step_metrics["velocity_r2"] = r2_score(velocities[:, step].numpy(), pred_vel[:, step].numpy())
            step_metrics["velocity_aee"] = torch.norm(pred_vel[:, step] - velocities[:, step], dim=1).mean().item()
        per_step[str(step + 1)] = step_metrics

    return {
        "aggregate": {
            "position_aee": torch.norm(pred_pos - positions, dim=2).mean().item(),
            "velocity_aee": torch.norm(pred_vel - velocities, dim=2).mean().item(),
        },
        "per_step": per_step,
    }


def physics_from_observed_frames_one_step(
    target_frames, predicted_frames, last_context_frames, **kwargs
):
    pred_positions = extract_positions_from_frames_robust(predicted_frames, **kwargs)
    tgt_positions = extract_positions_from_frames_robust(target_frames, **kwargs)
    ctx_positions = extract_positions_from_frames_robust(last_context_frames, **kwargs)

    pred_velocities = pred_positions - ctx_positions
    tgt_velocities = tgt_positions - ctx_positions
    position_dists = torch.norm(pred_positions - tgt_positions, dim=1)
    velocity_dists = torch.norm(pred_velocities - tgt_velocities, dim=1)

    return {
        "position_aee": torch.nanmean(position_dists).item(),
        "velocity_aee": torch.nanmean(velocity_dists).item(),
        "position_failures": torch.isnan(position_dists).sum().item(),
        "velocity_failures": torch.isnan(velocity_dists).sum().item(),
        "position_total": len(position_dists),
        "velocity_total": len(velocity_dists),
    }


def physics_from_observed_frames_rollout(
    target_frames, predicted_frames, **kwargs
):
    _, h = target_frames.shape[:2]
    per_step = {}
    all_pred_pos, all_tgt_pos = [], []

    for step in range(h):
        pred_positions = extract_positions_from_frames_robust(predicted_frames[:, step], **kwargs)
        tgt_positions = extract_positions_from_frames_robust(target_frames[:, step], **kwargs)
        
        position_dists = torch.norm(pred_positions - tgt_positions, dim=1)
        all_pred_pos.append(pred_positions)
        all_tgt_pos.append(tgt_positions)

        per_step[str(step + 1)] = {
            "position_aee": torch.nanmean(position_dists).item(),
            "position_failures": torch.isnan(position_dists).sum().item(),
            "position_total": len(position_dists),
        }

    all_pred_pos = torch.stack(all_pred_pos, dim=1)
    all_tgt_pos = torch.stack(all_tgt_pos, dim=1)

    pred_vel = all_pred_pos[:, 1:] - all_pred_pos[:, :-1]
    tgt_vel = all_tgt_pos[:, 1:] - all_tgt_pos[:, :-1]

    vel_dists = torch.norm(pred_vel - tgt_vel, dim=2)
    pos_dists = torch.norm(all_pred_pos - all_tgt_pos, dim=2)

    for step in range(1, h):
        per_step[str(step + 1)]["velocity_aee"] = torch.nanmean(vel_dists[:, step - 1]).item()
        per_step[str(step + 1)]["velocity_failures"] = torch.isnan(vel_dists[:, step - 1]).sum().item()
        per_step[str(step + 1)]["velocity_total"] = vel_dists[:, step - 1].numel()

    return {
        "aggregate": {
            "position_aee": torch.nanmean(pos_dists).item(),
            "velocity_aee": torch.nanmean(vel_dists).item(),
            "position_failures": torch.isnan(pos_dists).sum().item(),
            "velocity_failures": torch.isnan(vel_dists).sum().item(),
            "position_total": pos_dists.numel(),
            "velocity_total": vel_dists.numel(),
        },
        "per_step": per_step,
    }


def physics_from_latent_probe(train_hidden, train_pos, train_vel, test_hidden, test_pos, test_vel):
    X_train = train_hidden.reshape(train_hidden.shape[0], -1).numpy()
    X_test = test_hidden.reshape(test_hidden.shape[0], -1).numpy()

    y_train_pos, y_train_vel = train_pos.numpy(), train_vel.numpy()
    y_test_pos, y_test_vel = test_pos.numpy(), test_vel.numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    reg_pos = LinearRegression().fit(X_train, y_train_pos)
    reg_vel = LinearRegression().fit(X_train, y_train_vel)

    y_pred_pos = reg_pos.predict(X_test)
    y_pred_vel = reg_vel.predict(X_test)

    return {
        "position_r2": r2_score(y_test_pos, y_pred_pos),
        "velocity_r2": r2_score(y_test_vel, y_pred_vel),
        "position_aee": torch.norm(torch.tensor(y_pred_pos) - torch.tensor(y_test_pos), dim=1).mean().item(),
        "velocity_aee": torch.norm(torch.tensor(y_pred_vel) - torch.tensor(y_test_vel), dim=1).mean().item(),
    }


def physics_from_latent_probe_rollout(train_hidden, train_pos, train_vel, test_hidden, test_pos, test_vel):
    X_train = train_hidden.reshape(train_hidden.shape[0], -1).numpy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    reg_pos = LinearRegression().fit(X_train, train_pos.numpy())
    reg_vel = LinearRegression().fit(X_train, train_vel.numpy())

    h = test_hidden.shape[1]
    per_step = {}
    pred_pos_all, pred_vel_all = [], []

    for step in range(h):
        X_test = scaler.transform(test_hidden[:, step].reshape(test_hidden.shape[0], -1).numpy())
        
        y_pred_pos = reg_pos.predict(X_test)
        pred_pos_all.append(torch.tensor(y_pred_pos))
        y_test_pos = test_pos[:, step].numpy()

        metrics = {
            "position_r2": r2_score(y_test_pos, y_pred_pos),
            "position_aee": torch.norm(torch.tensor(y_pred_pos) - torch.tensor(y_test_pos), dim=1).mean().item(),
        }

        if step < test_vel.shape[1]:
            y_pred_vel = reg_vel.predict(X_test)
            pred_vel_all.append(torch.tensor(y_pred_vel))
            y_test_vel = test_vel[:, step].numpy()
            
            metrics["velocity_r2"] = r2_score(y_test_vel, y_pred_vel)
            metrics["velocity_aee"] = torch.norm(torch.tensor(y_pred_vel) - torch.tensor(y_test_vel), dim=1).mean().item()

        per_step[str(step + 1)] = metrics

    pred_pos_all = torch.stack(pred_pos_all, dim=1)
    results = {
        "aggregate": {"position_aee": torch.norm(pred_pos_all - test_pos, dim=2).mean().item()},
        "per_step": per_step,
    }

    if pred_vel_all:
        pred_vel_all = torch.stack(pred_vel_all, dim=1)
        results["aggregate"]["velocity_aee"] = torch.norm(pred_vel_all - test_vel, dim=2).mean().item()

    return results


def build_frame_loader(sequence_dirs, context, invert, grayscale, num_workers, batch_size=64, shuffle=False, stride=1):
    dataset = FramePredictionDataset(
        sequence_dirs=sequence_dirs,
        context=context,
        rollout=1,
        stride=stride,
        grayscale=grayscale,
        invert=invert,
        return_state=True
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def build_rollout_loader(sequence_dirs, context, invert, grayscale, rollout_steps, num_workers, batch_size=32, stride=5):
    dataset = FixedRolloutDataset(
        sequence_dirs=sequence_dirs,
        context=context,
        rollout_steps=rollout_steps,
        grayscale=grayscale,
        invert=invert,
        return_state=True,
        stride=stride
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def resolve_eval_dirs(args):
    test_dirs = get_sequence_dirs(args.data_dir)
    if not test_dirs:
        raise ValueError(f"No valid sequences found in test data dir: {args.data_dir}")

    if args.probe_train_dir is not None:
        probe_train_dirs = get_sequence_dirs(args.probe_train_dir)
        split_mode = "explicit_probe_train_dir"
    else:
        probe_train_dirs, test_dirs = split_sequence_dirs(test_dirs, val_ratio=args.val_ratio, seed=args.seed)
        split_mode = "legacy_internal_split"

    return probe_train_dirs, test_dirs, split_mode


def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    model, ckpt_args = load_model(args.model_dir, args.ckpt_name, device)
    context = ckpt_args.get("context", args.context)
    invert = ckpt_args.get("invert", args.invert)
    grayscale = ckpt_args.get("grayscale", True)

    probe_train_dirs, test_dirs, split_mode = resolve_eval_dirs(args)

    print(f"Evaluation split mode: {split_mode}")
    print(f"Probe-train trajectories: {len(probe_train_dirs)}")
    print(f"Test trajectories:        {len(test_dirs)}")

    print("\n=== Evaluating mode: one_step ===")
    
    # 1. Estrarre i Probe dal Train Set
    probe_train_loader = build_frame_loader(
        sequence_dirs=probe_train_dirs, context=context, invert=invert, grayscale=grayscale,
        num_workers=args.num_workers, batch_size=args.frame_batch_size, shuffle=True, stride=args.eval_stride
    )
    
    _, _, train_hidden_states, train_state_preds, train_positions, train_velocities, _ = frame_prediction(
        model, probe_train_loader, device, args.fm_steps, extract_only=True
    )

    # 2. Valutare il One-Step sul Test Set
    test_loader = build_frame_loader(
        sequence_dirs=test_dirs, context=context, invert=invert, grayscale=grayscale,
        num_workers=args.num_workers, batch_size=args.frame_batch_size, shuffle=False, stride=args.eval_stride
    )

    target_frames, predicted_frames, hidden_states, state_preds, positions, velocities, last_context_frames = frame_prediction(
        model, test_loader, device, args.fm_steps, extract_only=False
    )

    kwargs = {
        "threshold": args.threshold, "invert": invert, "min_mass": args.min_mass,
        "fallback_thresholds": tuple(args.fallback_thresholds),
        "use_topk_fallback": not args.disable_topk_fallback,
        "topk_ratio": args.topk_ratio, "debug": args.debug_threshold
    }

    observed_metrics = physics_from_observed_frames_one_step(
        target_frames, predicted_frames, last_context_frames, **kwargs
    )
    latent_probe_metrics = physics_from_latent_probe(
        train_hidden_states, train_positions, train_velocities, hidden_states, positions, velocities
    )
    state_head_metrics = compute_state_metrics_one_step(state_preds, positions, velocities)

    print("Observed physics:")
    print(f"Position AEE: {observed_metrics['position_aee']:.4f}")
    
    print("State head physics:")
    print(f"Position AEE: {state_head_metrics['position_aee']:.4f}")

    print("\n=== Evaluating mode: rollout ===")
    rollout_test_loader = build_rollout_loader(
        sequence_dirs=test_dirs, context=context, invert=invert, grayscale=grayscale,
        rollout_steps=args.rollout_steps, num_workers=args.num_workers, 
        batch_size=args.rollout_batch_size, stride=args.eval_stride
    )

    target_frames, predicted_frames, hidden_states, state_preds, positions, velocities, _ = rollout_prediction(
        model, rollout_test_loader, device, args.rollout_steps, args.fm_steps
    )

    observed_rollout_metrics = physics_from_observed_frames_rollout(target_frames, predicted_frames, **kwargs)
    latent_rollout_metrics = physics_from_latent_probe_rollout(
        train_hidden_states, train_positions, train_velocities, hidden_states, positions, velocities
    )
    state_head_rollout_metrics = compute_state_metrics_rollout(state_preds, positions, velocities)

    print("Observed rollout physics:")
    print(f"Position AEE: {observed_rollout_metrics['aggregate']['position_aee']:.4f}")

    results = {
        "metadata": {
            "model_dir": args.model_dir, "ckpt_name": args.ckpt_name, "data_dir": args.data_dir,
            "probe_train_dir": args.probe_train_dir, "split_mode": split_mode, "context": context,
            "invert": invert, "rollout_steps": args.rollout_steps, "fm_steps": args.fm_steps,
            "eval_stride": args.eval_stride
        },
        "one_step": {
            "from_observed": observed_metrics,
            "from_latent_probe": latent_probe_metrics,
            "from_state_head": state_head_metrics,
        },
        "rollout": {
            "from_observed": observed_rollout_metrics,
            "from_latent_probe": latent_rollout_metrics,
            "from_state_head": state_head_rollout_metrics,
        }
    }

    save_dir = os.path.join(args.model_dir, "physics")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"evaluation_results_{os.path.basename(os.path.normpath(args.data_dir))}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="best.pt")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--probe_train_dir", type=str, default=None)

    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fallback_thresholds", type=float, nargs="+", default=[0.4, 0.3, 0.2, 0.15, 0.1, 0.05])
    parser.add_argument("--min_mass", type=int, default=3)
    parser.add_argument("--topk_ratio", type=float, default=0.01)
    parser.add_argument("--disable_topk_fallback", action="store_true")

    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--rollout_steps", type=int, default=10)
    parser.add_argument("--fm_steps", type=int, default=20)
    parser.add_argument("--eval_stride", type=int, default=5, help="Stride per evitare RAM OOM")

    parser.add_argument("--frame_batch_size", type=int, default=16)
    parser.add_argument("--rollout_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_threshold", action="store_true")

    args = parser.parse_args()
    main(args)