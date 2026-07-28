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
        state_dim=args_dict.get("state_dim", 4),
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
    def __init__(self, sequence_dirs, context=5, rollout_steps=10, grayscale=True, invert=False, return_state=True,
                 stride=1):
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
                if target_seq.shape[0] == rollout_steps and pos.shape[0] == rollout_steps and vel.shape[
                    0] == rollout_steps:
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
    """Estimate the center of one bright foreground object.

    This is kept for single-ball environments. The current main datasets use
    dark background + bright ball(s), so we always look for bright pixels.
    """
    centers = estimate_object_centers_robust(
        frame_tensor=frame_tensor,
        num_objects=1,
        invert=invert,
        threshold=threshold,
        min_mass=min_mass,
        fallback_thresholds=fallback_thresholds,
        use_topk_fallback=use_topk_fallback,
        topk_ratio=topk_ratio,
        debug=debug,
    )
    return centers[0]


def _connected_components(mask):
    """Return connected components from a 2D boolean numpy mask."""
    mask = mask.astype(bool)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or visited[y0, x0]:
                continue

            stack = [(y0, x0)]
            visited[y0, x0] = True
            ys = []
            xs = []

            while stack:
                y, x = stack.pop()
                ys.append(y)
                xs.append(x)

                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            components.append((np.asarray(ys), np.asarray(xs)))

    return components


def _centers_from_components(frame_np, mask_np, num_objects, min_mass):
    components = _connected_components(mask_np)

    centers = []
    for ys, xs in components:
        if len(xs) < min_mass:
            continue
        vals = frame_np[ys, xs].astype(np.float64)
        weight_sum = vals.sum()
        if weight_sum <= 1e-8:
            cx = float(xs.mean())
            cy = float(ys.mean())
        else:
            cx = float((xs * vals).sum() / weight_sum)
            cy = float((ys * vals).sum() / weight_sum)
        centers.append((len(xs), cx, cy))

    centers.sort(key=lambda item: item[0], reverse=True)
    centers = centers[:num_objects]

    if len(centers) < num_objects:
        return None

    result = torch.tensor([[cx, cy] for _, cx, cy in centers], dtype=torch.float32)
    # Deterministic order for the initial estimate. Matching code later handles identity.
    result = result[torch.argsort(result[:, 0])]
    return result


def _topk_fallback_single_or_two(frame, num_objects, topk_ratio, min_mass):
    """Fallback for 1-ball / 2-ball cases using the brightest pixels.

    For one object, return the mean of the brightest pixels.
    For two objects, run a tiny 2-cluster k-means over the brightest pixels.
    """
    if num_objects not in (1, 2):
        raise ValueError(f"Only num_objects=1 or num_objects=2 are supported, got {num_objects}")

    flat = frame.reshape(-1)
    k = max(num_objects * max(min_mass, 10), int(topk_ratio * flat.numel()))
    k = min(k, flat.numel())

    vals, idx = torch.topk(flat, k=k, largest=True)
    valid = vals > 0
    idx = idx[valid]

    if idx.numel() < num_objects * min_mass:
        return torch.full((num_objects, 2), float("nan"), dtype=torch.float32)

    h, w = frame.shape
    points = torch.stack([(idx % w).float(), (idx // w).float()], dim=1)

    if num_objects == 1:
        center = points.mean(dim=0, keepdim=True)
        return center.cpu().float()

    # num_objects == 2. Initialize centers as leftmost and rightmost bright pixels.
    order = torch.argsort(points[:, 0])
    centers = torch.stack([points[order[0]], points[order[-1]]], dim=0).clone()

    for _ in range(10):
        dists = torch.cdist(points, centers)
        labels = torch.argmin(dists, dim=1)

        new_centers = []
        for obj_idx in range(2):
            cluster = points[labels == obj_idx]
            if cluster.numel() == 0:
                new_centers.append(centers[obj_idx])
            else:
                new_centers.append(cluster.mean(dim=0))
        centers = torch.stack(new_centers, dim=0)

    centers = centers[torch.argsort(centers[:, 0])]
    return centers.cpu().float()


def estimate_object_centers_robust(
        frame_tensor, num_objects=1, invert=False, threshold=0.5, min_mass=3,
        fallback_thresholds=(0.4, 0.3, 0.2, 0.15, 0.1, 0.05),
        use_topk_fallback=True, topk_ratio=0.01, debug=False,
):
    """Estimate centers for supported observed-space cases.

    Supported cases:
        num_objects=1: single bright ball, used for baseline and magnetic_wells.
        num_objects=2: two bright balls, used for billiard.

    Returns:
        Tensor of shape (num_objects, 2), with coordinates [x, y].

    The current main datasets use dark background + bright ball(s), so this
    extractor always looks for bright foreground pixels. The invert argument is
    kept only for compatibility with older call sites.
    """
    if num_objects not in (1, 2):
        raise ValueError(f"Only num_objects=1 or num_objects=2 are supported, got {num_objects}")

    frame = _to_single_channel(frame_tensor).float().clamp(0, 1).cpu()
    thresholds = [threshold] + [t for t in fallback_thresholds if t != threshold]

    frame_np = frame.numpy()

    for thr in thresholds:
        # Current main datasets: bright ball(s) on dark background.
        mask_np = frame_np > thr
        centers = _centers_from_components(frame_np, mask_np, num_objects, min_mass)
        if centers is not None:
            return centers

    if use_topk_fallback:
        return _topk_fallback_single_or_two(frame, num_objects, topk_ratio, min_mass)

    return torch.full((num_objects, 2), float("nan"), dtype=torch.float32)


def extract_positions_from_frames_robust(frames, num_objects=1, **kwargs):
    centers = [
        estimate_object_centers_robust(frame, num_objects=num_objects, **kwargs)
        for frame in frames
    ]
    centers = torch.stack(centers, dim=0)
    if num_objects == 1:
        return centers[:, 0]
    return centers


def infer_num_objects_from_state(positions):
    """Infer whether evaluation should use the 1-ball or 2-ball extractor."""
    if positions is None:
        return 1

    # One-step single ball: (B, 2)
    # One-step billiard:    (B, 2, 2)
    if positions.dim() == 3 and positions.shape[-1] == 2:
        n = int(positions.shape[-2])
        return n if n in (1, 2) else 1

    # Rollout billiard: (B, H, 2, 2)
    if positions.dim() == 4 and positions.shape[-1] == 2:
        n = int(positions.shape[-2])
        return n if n in (1, 2) else 1

    return 1


def _best_permutation_indices(ref, points):
    """Return indices that reorder points to best match ref. Supports small N."""
    n = ref.shape[0]
    if n == 1:
        return torch.tensor([0], dtype=torch.long)

    import itertools

    best_perm = None
    best_dist = None
    for perm in itertools.permutations(range(n)):
        perm_tensor = torch.tensor(perm, dtype=torch.long)
        dist = torch.norm(points[perm_tensor] - ref, dim=1).mean()
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_perm = perm_tensor
    return best_perm


def match_points_to_reference(points, reference):
    """Reorder object centers for each sample to match reference centers."""
    if points.dim() == 2:
        return points

    matched = []
    for p, r in zip(points, reference):
        if torch.isnan(p).any() or torch.isnan(r).any():
            matched.append(p)
            continue
        perm = _best_permutation_indices(r, p)
        matched.append(p[perm])
    return torch.stack(matched, dim=0)


def object_set_position_distances(pred_positions, tgt_positions):
    """Mean per-object distance after optimal matching per sample."""
    if pred_positions.dim() == 2:
        return torch.norm(pred_positions - tgt_positions, dim=1)

    matched_pred = match_points_to_reference(pred_positions, tgt_positions)
    return torch.norm(matched_pred - tgt_positions, dim=2).mean(dim=1)


def order_sequence_by_previous_step(positions):
    """Keep object identity approximately consistent over time.

    positions: (B, H, N, 2)
    """
    if positions.dim() != 4:
        return positions

    ordered = positions.clone()
    for step in range(1, ordered.shape[1]):
        ordered[:, step] = match_points_to_reference(ordered[:, step], ordered[:, step - 1])
    return ordered


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


def _flatten_state_tensor(x):
    """
    Flatten physical state dimensions while preserving batch/time axes.

    One-step single ball:   (B, 2)       -> (B, 2)
    One-step multi ball:    (B, N, 2)    -> (B, 2*N)
    Rollout single ball:    (B, H, 2)    -> (B, H, 2)
    Rollout multi ball:     (B, H, N, 2) -> (B, H, 2*N)
    """
    if x is None:
        return None
    if x.dim() <= 2:
        return x
    if x.dim() == 3:
        # Ambiguous case:
        #   one-step multi-ball: (B, N, 2)
        #   rollout single-ball: (B, H, 2)
        # The caller decides by context.
        return x
    return x.reshape(*x.shape[:2], -1)


def _flatten_one_step_state(x):
    if x is None:
        return None
    return x.reshape(x.shape[0], -1)


def _flatten_rollout_state(x):
    if x is None:
        return None
    return x.reshape(x.shape[0], x.shape[1], -1)


def compute_state_metrics_one_step(state_preds, positions, velocities):
    positions_flat = _flatten_one_step_state(positions)
    velocities_flat = _flatten_one_step_state(velocities)

    pos_dim = positions_flat.shape[1]
    vel_dim = velocities_flat.shape[1]

    pred_pos = state_preds[:, :pos_dim]
    pred_vel = state_preds[:, pos_dim:pos_dim + vel_dim]

    return {
        "position_r2": r2_score(positions_flat.numpy(), pred_pos.numpy()),
        "velocity_r2": r2_score(velocities_flat.numpy(), pred_vel.numpy()),
        "position_aee": torch.norm(pred_pos - positions_flat, dim=1).mean().item(),
        "velocity_aee": torch.norm(pred_vel - velocities_flat, dim=1).mean().item(),
    }


def compute_state_metrics_rollout(state_preds, positions, velocities):
    positions_flat = _flatten_rollout_state(positions)
    velocities_flat = _flatten_rollout_state(velocities)

    pos_dim = positions_flat.shape[2]
    vel_dim = velocities_flat.shape[2]

    pred_pos = state_preds[:, :, :pos_dim]
    pred_vel = state_preds[:, :, pos_dim:pos_dim + vel_dim]

    per_step = {}
    for step in range(state_preds.shape[1]):
        step_metrics = {
            "position_r2": r2_score(positions_flat[:, step].numpy(), pred_pos[:, step].numpy()),
            "position_aee": torch.norm(pred_pos[:, step] - positions_flat[:, step], dim=1).mean().item(),
        }
        if step < velocities_flat.shape[1]:
            step_metrics["velocity_r2"] = r2_score(velocities_flat[:, step].numpy(), pred_vel[:, step].numpy())
            step_metrics["velocity_aee"] = torch.norm(pred_vel[:, step] - velocities_flat[:, step], dim=1).mean().item()
        per_step[str(step + 1)] = step_metrics

    return {
        "aggregate": {
            "position_aee": torch.norm(pred_pos - positions_flat, dim=2).mean().item(),
            "velocity_aee": torch.norm(pred_vel - velocities_flat, dim=2).mean().item(),
        },
        "per_step": per_step,
    }


def physics_from_latent_probe(train_hidden, train_pos, train_vel, test_hidden, test_pos, test_vel):
    X_train = train_hidden.reshape(train_hidden.shape[0], -1).numpy()
    X_test = test_hidden.reshape(test_hidden.shape[0], -1).numpy()

    y_train_pos = _flatten_one_step_state(train_pos).numpy()
    y_train_vel = _flatten_one_step_state(train_vel).numpy()
    y_test_pos = _flatten_one_step_state(test_pos).numpy()
    y_test_vel = _flatten_one_step_state(test_vel).numpy()

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

    train_pos_flat = _flatten_one_step_state(train_pos)
    train_vel_flat = _flatten_one_step_state(train_vel)
    test_pos_flat = _flatten_rollout_state(test_pos)
    test_vel_flat = _flatten_rollout_state(test_vel)

    reg_pos = LinearRegression().fit(X_train, train_pos_flat.numpy())
    reg_vel = LinearRegression().fit(X_train, train_vel_flat.numpy())

    h = test_hidden.shape[1]
    per_step = {}
    pred_pos_all, pred_vel_all = [], []

    for step in range(h):
        X_test = scaler.transform(test_hidden[:, step].reshape(test_hidden.shape[0], -1).numpy())

        y_pred_pos = reg_pos.predict(X_test)
        pred_pos_all.append(torch.tensor(y_pred_pos))
        y_test_pos = test_pos_flat[:, step].numpy()

        metrics = {
            "position_r2": r2_score(y_test_pos, y_pred_pos),
            "position_aee": torch.norm(torch.tensor(y_pred_pos) - torch.tensor(y_test_pos), dim=1).mean().item(),
        }

        if step < test_vel_flat.shape[1]:
            y_pred_vel = reg_vel.predict(X_test)
            pred_vel_all.append(torch.tensor(y_pred_vel))
            y_test_vel = test_vel_flat[:, step].numpy()

            metrics["velocity_r2"] = r2_score(y_test_vel, y_pred_vel)
            metrics["velocity_aee"] = torch.norm(torch.tensor(y_pred_vel) - torch.tensor(y_test_vel),
                                                 dim=1).mean().item()

        per_step[str(step + 1)] = metrics

    pred_pos_all = torch.stack(pred_pos_all, dim=1)
    results = {
        "aggregate": {"position_aee": torch.norm(pred_pos_all - test_pos_flat, dim=2).mean().item()},
        "per_step": per_step,
    }

    if pred_vel_all:
        pred_vel_all = torch.stack(pred_vel_all, dim=1)
        results["aggregate"]["velocity_aee"] = torch.norm(pred_vel_all - test_vel_flat, dim=2).mean().item()

    return results


def physics_from_observed_frames_one_step(
        target_frames, predicted_frames, last_context_frames, num_objects=1, **kwargs
):
    pred_positions = extract_positions_from_frames_robust(predicted_frames, num_objects=num_objects, **kwargs)
    tgt_positions = extract_positions_from_frames_robust(target_frames, num_objects=num_objects, **kwargs)
    ctx_positions = extract_positions_from_frames_robust(last_context_frames, num_objects=num_objects, **kwargs)

    if num_objects > 1:
        pred_positions = match_points_to_reference(pred_positions, ctx_positions)
        tgt_positions = match_points_to_reference(tgt_positions, ctx_positions)

    pred_velocities = pred_positions - ctx_positions
    tgt_velocities = tgt_positions - ctx_positions

    position_dists = object_set_position_distances(pred_positions, tgt_positions)

    if num_objects > 1:
        velocity_dists = torch.norm(pred_velocities - tgt_velocities, dim=2).mean(dim=1)
    else:
        velocity_dists = torch.norm(pred_velocities - tgt_velocities, dim=1)

    return {
        "position_aee": torch.nanmean(position_dists).item(),
        "velocity_aee": torch.nanmean(velocity_dists).item(),
        "position_failures": torch.isnan(position_dists).sum().item(),
        "velocity_failures": torch.isnan(velocity_dists).sum().item(),
        "position_total": len(position_dists),
        "velocity_total": len(velocity_dists),
        "num_objects": num_objects,
    }


def physics_from_observed_frames_rollout(
        target_frames, predicted_frames, num_objects=1, **kwargs
):
    _, h = target_frames.shape[:2]
    per_step = {}
    all_pred_pos, all_tgt_pos = [], []

    for step in range(h):
        pred_positions = extract_positions_from_frames_robust(
            predicted_frames[:, step], num_objects=num_objects, **kwargs
        )
        tgt_positions = extract_positions_from_frames_robust(
            target_frames[:, step], num_objects=num_objects, **kwargs
        )

        position_dists = object_set_position_distances(pred_positions, tgt_positions)
        all_pred_pos.append(pred_positions)
        all_tgt_pos.append(tgt_positions)

        per_step[str(step + 1)] = {
            "position_aee": torch.nanmean(position_dists).item(),
            "position_failures": torch.isnan(position_dists).sum().item(),
            "position_total": len(position_dists),
        }

    all_pred_pos = torch.stack(all_pred_pos, dim=1)
    all_tgt_pos = torch.stack(all_tgt_pos, dim=1)

    if num_objects > 1:
        all_pred_pos = order_sequence_by_previous_step(all_pred_pos)
        all_tgt_pos = order_sequence_by_previous_step(all_tgt_pos)

    if num_objects > 1:
        # Align prediction object order to target object order at each step.
        aligned_pred_pos = []
        for step in range(h):
            aligned_pred_pos.append(match_points_to_reference(all_pred_pos[:, step], all_tgt_pos[:, step]))
        aligned_pred_pos = torch.stack(aligned_pred_pos, dim=1)
        pos_dists = torch.norm(aligned_pred_pos - all_tgt_pos, dim=3).mean(dim=2)

        pred_vel = aligned_pred_pos[:, 1:] - aligned_pred_pos[:, :-1]
        tgt_vel = all_tgt_pos[:, 1:] - all_tgt_pos[:, :-1]
        vel_dists = torch.norm(pred_vel - tgt_vel, dim=3).mean(dim=2)
    else:
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
            "num_objects": num_objects,
        },
        "per_step": per_step,
    }


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


def build_rollout_loader(sequence_dirs, context, invert, grayscale, rollout_steps, num_workers, batch_size=32,
                         stride=5):
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

    num_objects = args.num_objects
    if num_objects is None:
        num_objects = infer_num_objects_from_state(positions)
    print(f"Observed extractor num_objects: {num_objects}")

    kwargs = {
        "threshold": args.threshold, "invert": invert, "min_mass": args.min_mass,
        "fallback_thresholds": tuple(args.fallback_thresholds),
        "use_topk_fallback": not args.disable_topk_fallback,
        "topk_ratio": args.topk_ratio, "debug": args.debug_threshold
    }

    observed_metrics = physics_from_observed_frames_one_step(
        target_frames, predicted_frames, last_context_frames, num_objects=num_objects, **kwargs
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

    observed_rollout_metrics = physics_from_observed_frames_rollout(
        target_frames, predicted_frames, num_objects=num_objects, **kwargs
    )
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
            "eval_stride": args.eval_stride, "num_objects": num_objects
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
    parser.add_argument("--num_objects", type=int, choices=[1, 2], default=None,
                        help="Observed-space extractor mode. Use 1 for baseline/magnetic_wells, 2 for billiard. "
                             "Default: infer from state shape.")

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