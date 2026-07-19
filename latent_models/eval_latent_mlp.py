import torch
import torch.nn as nn
import os
import sys
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import r2_score

sys.path.append('..')

from dataset import FramePredictionDataset
from autoencoder import ConvAutoencoder

CONTEXT = 5
LATENT_DIM = 128
HIDDEN_DIM = 512

# Must match training script's held-out eval range exactly.
EVAL_TRAJ_START, EVAL_TRAJ_END = 50, 80


class LatentMLP(nn.Module):
    """Must match train_latent_mlp.py exactly (delta prediction)."""
    def __init__(self, latent_dim=128, hidden_dim=512, context=5):
        super().__init__()
        self.context = context
        self.net = nn.Sequential(
            nn.Linear(latent_dim * context, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, latent_seq):
        B, T, D = latent_seq.shape
        x = latent_seq.reshape(B, T * D)
        delta = self.net(x)
        last_latent = latent_seq[:, -1, :]
        return last_latent + delta


def get_position_from_frame(frame):
    """Extract the ball's (x, y) position from a decoded frame via an
    intensity-weighted centroid — NOT frame.mean(), which just measures
    overall brightness and has nothing to do with location.

    frame: tensor (C, H, W) or (H, W), values expected roughly in [0, 1].
    Returns: np.array([x, y]) in pixel coordinates.
    """
    if frame.dim() == 3:
        frame = frame.squeeze(0)  # assume single channel -> (H, W)
    H, W = frame.shape

    weights = frame.clamp(min=0)
    total = weights.sum()
    if total <= 1e-6:
        # decoder produced a blank/near-zero frame; fall back to image center
        # rather than returning garbage/NaN.
        return np.array([W / 2.0, H / 2.0])

    ys = torch.arange(H, device=frame.device, dtype=frame.dtype).view(H, 1)
    xs = torch.arange(W, device=frame.device, dtype=frame.dtype).view(1, W)

    cy = (weights * ys).sum() / total
    cx = (weights * xs).sum() / total
    return np.array([cx.item(), cy.item()])


@torch.no_grad()
def encode_sequence(ae, frames):
    # frames: (B, T, C, H, W) -> (B, T, D)
    B, T, C, H, W = frames.shape
    return torch.stack([ae.encode(frames[:, t]) for t in range(T)], dim=1)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    ae = ConvAutoencoder(latent_dim=LATENT_DIM).to(device)
    ae.load_state_dict(torch.load("../models/autoencoder/best.pth", map_location=device, weights_only=True))
    ae.eval()

    mlp = LatentMLP(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, context=CONTEXT).to(device)
    # NOTE: filename now matches what train_latent_mlp.py actually saves (best.pth).
    mlp.load_state_dict(torch.load("../models/latent_mlp/best.pth", map_location=device, weights_only=True))
    mlp.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])

    data_dirs = [os.path.join("..", "data_id", f'traj-{i}') for i in range(EVAL_TRAJ_START, EVAL_TRAJ_END)]
    print(f"Evaluating Latent + MLP on {len(data_dirs)} held-out trajectories "
          f"(traj-{EVAL_TRAJ_START} to traj-{EVAL_TRAJ_END - 1})...")

    short_term_preds = []
    short_term_gts = []
    rollout_errors_per_traj = []

    for traj_dir in tqdm(data_dirs):
        try:
            positions_path = os.path.join(traj_dir, "positions.npy")
            if not os.path.exists(positions_path):
                continue
            gt_positions = np.load(positions_path)  # (T_total, 2)

            n_frames = len(gt_positions)
            rollout_len = n_frames - CONTEXT
            if rollout_len < 1:
                continue

            # context=CONTEXT, rollout=rollout_len -> predict every remaining
            # frame in the trajectory, autoregressively.
            dataset = FramePredictionDataset([traj_dir], context=CONTEXT, rollout=rollout_len, transform=transform)
            if len(dataset) == 0:
                continue

            input_seq, target_seq = dataset[0]           # (CONTEXT,C,H,W), (rollout_len,C,H,W)
            input_seq = input_seq.unsqueeze(0).to(device)  # (1, CONTEXT, C, H, W)

            with torch.no_grad():
                window = encode_sequence(ae, input_seq)   # (1, CONTEXT, D)

                traj_errors = []
                for step in range(rollout_len):
                    pred_latent = mlp(window)              # (1, D)
                    pred_frame = ae.decode(pred_latent)[0]  # (C, H, W)

                    pred_pos = get_position_from_frame(pred_frame)
                    # Index fix: with context=CONTEXT frames (indices 0..CONTEXT-1) as
                    # input, the first predicted frame corresponds to positions[CONTEXT],
                    # not positions[CONTEXT + 1]. This was the off-by-one bug.
                    gt_pos = gt_positions[CONTEXT + step]

                    err = np.abs(pred_pos - gt_pos).mean()
                    traj_errors.append(err)

                    if step == 0:
                        short_term_preds.append(pred_pos)
                        short_term_gts.append(gt_pos)

                    # autoregressive: feed prediction back in for next step
                    window = torch.cat([window[:, 1:, :], pred_latent.unsqueeze(1)], dim=1)

            rollout_errors_per_traj.append(np.mean(traj_errors))

        except Exception as e:
            print(f"  skipped {traj_dir}: {e}")
            continue

    if rollout_errors_per_traj:
        short_term_preds = np.array(short_term_preds)
        short_term_gts = np.array(short_term_gts)

        # R^2 computed per-coordinate then averaged (matches typical convention
        # for 2D position regression; sklearn r2_score on flattened arrays
        # would also be reasonable — pick whichever your teammates used).
        r2_x = r2_score(short_term_gts[:, 0], short_term_preds[:, 0])
        r2_y = r2_score(short_term_gts[:, 1], short_term_preds[:, 1])
        short_term_r2 = (r2_x + r2_y) / 2

        short_term_aee = np.abs(short_term_preds - short_term_gts).mean()
        rollout_aee = np.mean(rollout_errors_per_traj)

        print(f"\n=== Latent + MLP Results (ID Split, held-out trajectories) ===")
        print(f"Short-term Pos R^2:   {short_term_r2:.4f}")
        print(f"Short-term Pos AEE:   {short_term_aee:.4f}")
        print(f"Rollout Pos AEE:      {rollout_aee:.4f}")
        print(f"Evaluated trajectories: {len(rollout_errors_per_traj)}")
    else:
        print("No valid trajectories evaluated.")


if __name__ == "__main__":
    main()