import argparse
import json
import os
import random


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


from models.rnn_baseline import BaselineVideoPredictor
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



def split_sequence_dirs(sequence_dirs, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    sequence_dirs = sequence_dirs.copy()
    rng.shuffle(sequence_dirs)


    val_size = max(1, int(len(sequence_dirs) * val_ratio))
    train_dirs = sequence_dirs[:-val_size]
    val_dirs = sequence_dirs[-val_size:]
    return train_dirs, val_dirs



def estimate_ball_center(frame_tensor, threshold=0.5, invert=False):
    if frame_tensor.dim() == 3:
        frame = frame_tensor.squeeze(0)
    else:
        frame = frame_tensor


    if invert:
        mask = frame > threshold
    else:
        mask = frame < threshold


    mass = mask.sum()


    if mass == 0:
        return torch.tensor([torch.nan, torch.nan], dtype=torch.float32)


    h, w = frame.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij"
    )


    center_x = (xs * mask.float()).sum() / mass.float()
    center_y = (ys * mask.float()).sum() / mass.float()


    return torch.tensor([center_x, center_y], dtype=torch.float32)



def extract_positions_from_frames(frames, threshold=0.5, invert=False):
    positions = [estimate_ball_center(frame, threshold=threshold, invert=invert) for frame in frames]
    return torch.stack(positions, dim=0)



def compute_velocity_from_positions(positions):
    return torch.diff(positions, dim=0)



class FixedRolloutDataset(Dataset):
    def __init__(self, sequence_dirs, context=5, rollout_steps=10, grayscale=True, invert=False, return_state=True):
        self.samples = []
        self.context = context
        self.rollout_steps = rollout_steps
        self.grayscale = grayscale
        self.invert = invert
        self.return_state = return_state


        base_dataset = FramePredictionDataset(
            sequence_dirs=sequence_dirs,
            context=context,
            rollout=rollout_steps,
            stride=1,
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



@torch.no_grad()
def frame_prediction(model, dataloader, device):
    target_frames = []
    predicted_frames = []
    hidden_states = []
    positions = []
    velocities = []


    for batch in tqdm.tqdm(dataloader, desc="One-step evaluation"):
        if len(batch) == 4:
            input_seq, target_seq, pos, vel = batch
            positions.append(pos[:, 0])
            velocities.append(vel[:, 0])
        else:
            input_seq, target_seq = batch


        input_seq = input_seq.to(device)
        target_next = target_seq[:, 0]


        pred_next, h_list = model(input_seq, return_hidden=True)
        last_hidden = h_list[-1]


        target_frames.append(target_next.cpu())
        predicted_frames.append(pred_next.clamp(0, 1).cpu())
        hidden_states.append(F.adaptive_avg_pool2d(last_hidden, (2, 2)).cpu())


    target_frames = torch.cat(target_frames, dim=0)
    predicted_frames = torch.cat(predicted_frames, dim=0)
    hidden_states = torch.cat(hidden_states, dim=0)


    if positions:
        positions = torch.cat(positions, dim=0)
        velocities = torch.cat(velocities, dim=0)
    else:
        positions, velocities = None, None


    return target_frames, predicted_frames, hidden_states, positions, velocities



@torch.no_grad()
def rollout_prediction(model, dataloader, device, rollout_steps):
    target_frames = []
    predicted_frames = []
    hidden_states = []
    positions = []
    velocities = []


    for batch in tqdm.tqdm(dataloader, desc="Rollout evaluation"):
        if len(batch) == 4:
            input_seq, target_seq, pos, vel = batch
        else:
            input_seq, target_seq = batch
            pos, vel = None, None


        input_seq = input_seq.to(device)


        current_context = input_seq.clone()
        rollout_preds = []
        rollout_hidden = []


        for _ in range(rollout_steps):
            pred_next, h_list = model(current_context, return_hidden=True)
            last_hidden = h_list[-1]


            rollout_preds.append(pred_next.clamp(0, 1).cpu())
            rollout_hidden.append(F.adaptive_avg_pool2d(last_hidden, (2, 2)).cpu())


            current_context = torch.cat([current_context[:, 1:], pred_next.unsqueeze(1)], dim=1)


        pred_seq = torch.stack(rollout_preds, dim=1)
        hidden_seq = torch.stack(rollout_hidden, dim=1)


        target_frames.append(target_seq.cpu())
        predicted_frames.append(pred_seq)
        hidden_states.append(hidden_seq)


        if pos is not None and vel is not None:
            positions.append(pos.cpu())
            velocities.append(vel.cpu())


    target_frames = torch.cat(target_frames, dim=0)
    predicted_frames = torch.cat(predicted_frames, dim=0)
    hidden_states = torch.cat(hidden_states, dim=0)


    if positions:
        positions = torch.cat(positions, dim=0)
        velocities = torch.cat(velocities, dim=0)
    else:
        positions, velocities = None, None


    return target_frames, predicted_frames, hidden_states, positions, velocities



def physics_from_observed_frames_one_step(target_frames, predicted_frames, threshold=0.5, invert=False):
    pred_positions = extract_positions_from_frames(predicted_frames, threshold=threshold, invert=invert)
    tgt_positions = extract_positions_from_frames(target_frames, threshold=threshold, invert=invert)


    pred_velocities = compute_velocity_from_positions(pred_positions)
    tgt_velocities = compute_velocity_from_positions(tgt_positions)


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



def physics_from_observed_frames_rollout(target_frames, predicted_frames, threshold=0.5, invert=False):
    _, h = target_frames.shape[:2]


    per_step = {}
    all_pred_pos = []
    all_tgt_pos = []


    for step in range(h):
        pred_step = predicted_frames[:, step]
        tgt_step = target_frames[:, step]


        pred_positions = extract_positions_from_frames(pred_step, threshold=threshold, invert=invert)
        tgt_positions = extract_positions_from_frames(tgt_step, threshold=threshold, invert=invert)


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


    aggregate = {
        "position_aee": torch.nanmean(pos_dists).item(),
        "velocity_aee": torch.nanmean(vel_dists).item(),
        "position_failures": torch.isnan(pos_dists).sum().item(),
        "velocity_failures": torch.isnan(vel_dists).sum().item(),
        "position_total": pos_dists.numel(),
        "velocity_total": vel_dists.numel(),
    }


    return {
        "aggregate": aggregate,
        "per_step": per_step,
    }, all_pred_pos, all_tgt_pos, pred_vel, tgt_vel



def physics_from_latent(train_hidden, train_pos, train_vel, test_hidden, test_pos, test_vel):
    X_train = train_hidden.reshape(train_hidden.shape[0], -1).numpy()
    X_test = test_hidden.reshape(test_hidden.shape[0], -1).numpy()


    y_train_pos = train_pos.numpy()
    y_train_vel = train_vel.numpy()
    y_test_pos = test_pos.numpy()
    y_test_vel = test_vel.numpy()


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



def physics_from_latent_rollout(train_hidden, train_pos, train_vel, test_hidden, test_pos, test_vel):
    X_train = train_hidden.reshape(train_hidden.shape[0], -1).numpy()
    y_train_pos = train_pos.numpy()
    y_train_vel = train_vel.numpy()


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)


    reg_pos = LinearRegression().fit(X_train, y_train_pos)
    reg_vel = LinearRegression().fit(X_train, y_train_vel)


    h = test_hidden.shape[1]
    per_step = {}
    pred_pos_all = []
    pred_vel_all = []


    for step in range(h):
        X_test = test_hidden[:, step].reshape(test_hidden.shape[0], -1).numpy()
        X_test = scaler.transform(X_test)


        y_test_pos = test_pos[:, step].numpy()
        y_pred_pos = reg_pos.predict(X_test)


        pred_pos_all.append(torch.tensor(y_pred_pos))


        metrics = {
            "position_r2": r2_score(y_test_pos, y_pred_pos),
            "position_aee": torch.norm(torch.tensor(y_pred_pos) - torch.tensor(y_test_pos), dim=1).mean().item(),
        }


        if step < test_vel.shape[1]:
            y_test_vel = test_vel[:, step].numpy()
            y_pred_vel = reg_vel.predict(X_test)
            pred_vel_all.append(torch.tensor(y_pred_vel))


            metrics["velocity_r2"] = r2_score(y_test_vel, y_pred_vel)
            metrics["velocity_aee"] = torch.norm(torch.tensor(y_pred_vel) - torch.tensor(y_test_vel), dim=1).mean().item()


        per_step[str(step + 1)] = metrics


    pred_pos_all = torch.stack(pred_pos_all, dim=1)
    pred_vel_all = torch.stack(pred_vel_all, dim=1) if len(pred_vel_all) > 0 else None


    agg_pos = torch.norm(pred_pos_all - test_pos, dim=2).mean().item()


    results = {
        "aggregate": {
            "position_aee": agg_pos,
        },
        "per_step": per_step,
    }


    if pred_vel_all is not None:
        agg_vel = torch.norm(pred_vel_all - test_vel, dim=2).mean().item()
        results["aggregate"]["velocity_aee"] = agg_vel


    return results



def build_frame_loader(sequence_dirs, context, invert, num_workers, batch_size=64, shuffle=False):
    dataset = FramePredictionDataset(
        sequence_dirs=sequence_dirs,
        context=context,
        rollout=1,
        stride=1,
        grayscale=True,
        invert=invert,
        return_state=True
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)



def build_rollout_loader(sequence_dirs, context, invert, rollout_steps, num_workers, batch_size=32):
    dataset = FixedRolloutDataset(
        sequence_dirs=sequence_dirs,
        context=context,
        rollout_steps=rollout_steps,
        grayscale=True,
        invert=invert,
        return_state=True
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)



def resolve_eval_dirs(args):
    test_dirs = get_sequence_dirs(args.data_dir)
    if len(test_dirs) == 0:
        raise ValueError(f"No valid sequences found in test data dir: {args.data_dir}")


    if args.probe_train_dir is not None:
        probe_train_dirs = get_sequence_dirs(args.probe_train_dir)
        if len(probe_train_dirs) == 0:
            raise ValueError(f"No valid sequences found in probe train dir: {args.probe_train_dir}")
        split_mode = "explicit_probe_train_dir"
    else:
        probe_train_dirs, test_dirs = split_sequence_dirs(
            test_dirs,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        split_mode = "legacy_internal_split"


    return probe_train_dirs, test_dirs, split_mode



def main(args):
    set_seed(args.seed)


    device = get_device()
    print(f"Using device: {device}")


    model, ckpt_args = load_model(args.model_dir, args.ckpt_name, device)


    context = ckpt_args.get("context", args.context)
    invert = ckpt_args.get("invert", args.invert)


    probe_train_dirs, test_dirs, split_mode = resolve_eval_dirs(args)


    print(f"Evaluation split mode: {split_mode}")
    print(f"Probe-train trajectories: {len(probe_train_dirs)}")
    print(f"Test trajectories:        {len(test_dirs)}")


    results = {
        "metadata": {
            "model_dir": args.model_dir,
            "ckpt_name": args.ckpt_name,
            "data_dir": args.data_dir,
            "probe_train_dir": args.probe_train_dir,
            "split_mode": split_mode,
            "context": context,
            "invert": invert,
            "threshold": args.threshold,
            "rollout_steps": args.rollout_steps,
            "seed": args.seed,
        }
    }


    print("\n=== Evaluating mode: one_step ===")
    probe_train_loader = build_frame_loader(
        sequence_dirs=probe_train_dirs,
        context=context,
        invert=invert,
        num_workers=args.num_workers,
        batch_size=64,
        shuffle=True,
    )
    test_loader = build_frame_loader(
        sequence_dirs=test_dirs,
        context=context,
        invert=invert,
        num_workers=args.num_workers,
        batch_size=64,
        shuffle=False,
    )


    target_frames, predicted_frames, hidden_states, positions, velocities = frame_prediction(
        model, test_loader, device
    )


    observed_metrics = physics_from_observed_frames_one_step(
        target_frames=target_frames,
        predicted_frames=predicted_frames,
        threshold=args.threshold,
        invert=invert
    )


    print("Observed physics:")
    print(f"Position AEE: {observed_metrics['position_aee']:.4f}")
    print(f"Velocity AEE: {observed_metrics['velocity_aee']:.4f}")
    print(f"Position failures: {observed_metrics['position_failures']} / {observed_metrics['position_total']}")
    print(f"Velocity failures: {observed_metrics['velocity_failures']} / {observed_metrics['velocity_total']}")


    _, _, train_hidden_states, train_positions, train_velocities = frame_prediction(
        model, probe_train_loader, device
    )


    latent_metrics = physics_from_latent(
        train_hidden=train_hidden_states,
        train_pos=train_positions,
        train_vel=train_velocities,
        test_hidden=hidden_states,
        test_pos=positions,
        test_vel=velocities,
    )


    print("Latent physics:")
    print(f"Position R2: {latent_metrics['position_r2']:.4f}")
    print(f"Velocity R2: {latent_metrics['velocity_r2']:.4f}")
    print(f"Position AEE: {latent_metrics['position_aee']:.4f}")
    print(f"Velocity AEE: {latent_metrics['velocity_aee']:.4f}")


    results["one_step"] = {
        "from_observed": observed_metrics,
        "from_latent": latent_metrics,
    }


    print("\n=== Evaluating mode: rollout ===")
    rollout_test_loader = build_rollout_loader(
        sequence_dirs=test_dirs,
        context=context,
        invert=invert,
        rollout_steps=args.rollout_steps,
        num_workers=args.num_workers,
        batch_size=32,
    )


    target_frames, predicted_frames, hidden_states, positions, velocities = rollout_prediction(
        model, rollout_test_loader, device, args.rollout_steps
    )


    observed_rollout_metrics, _, _, _, _ = physics_from_observed_frames_rollout(
        target_frames=target_frames,
        predicted_frames=predicted_frames,
        threshold=args.threshold,
        invert=invert
    )


    print("Observed physics:")
    print(f"Position AEE: {observed_rollout_metrics['aggregate']['position_aee']:.4f}")
    print(f"Velocity AEE: {observed_rollout_metrics['aggregate']['velocity_aee']:.4f}")


    latent_rollout_metrics = physics_from_latent_rollout(
        train_hidden=train_hidden_states,
        train_pos=train_positions,
        train_vel=train_velocities,
        test_hidden=hidden_states,
        test_pos=positions,
        test_vel=velocities[:, 1:],
    )


    print("Latent physics:")
    print(f"Position AEE: {latent_rollout_metrics['aggregate']['position_aee']:.4f}")
    if "velocity_aee" in latent_rollout_metrics["aggregate"]:
        print(f"Velocity AEE: {latent_rollout_metrics['aggregate']['velocity_aee']:.4f}")


    results["rollout"] = {
        "from_observed": observed_rollout_metrics,
        "from_latent": latent_rollout_metrics,
    }


    split_name = os.path.basename(os.path.normpath(args.data_dir))
    save_dir = os.path.join(args.model_dir, "physics")
    os.makedirs(save_dir, exist_ok=True)


    save_path = os.path.join(save_dir, f"evaluation_results_{split_name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


    print(f"\nSaved results to: {save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate physics learning of recurrent video model with fixed rollout horizon"
    )


    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="best.pt")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--probe_train_dir",
        type=str,
        default=None,
        help="Directory used to fit the linear latent probes. If omitted, falls back to legacy internal split."
    )


    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--rollout_steps", type=int, default=10)


    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)


    args = parser.parse_args()
    main(args)