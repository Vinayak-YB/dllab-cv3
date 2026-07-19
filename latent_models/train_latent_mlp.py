import torch
import torch.nn as nn
import random
import os
import sys
from tqdm import tqdm
from torchvision import transforms

sys.path.append('..')

from dataset import FramePredictionDataset
from autoencoder import ConvAutoencoder

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONTEXT = 5
MAX_ROLLOUT = 10          # dataset always returns this many target frames;
                           # training curriculum uses a growing slice of it
LATENT_DIM = 128
HIDDEN_DIM = 512
EPOCHS = 100
BATCH_SIZE = 64
LR = 2e-4

# Trajectories 50-79 are reserved for eval_latent_mlp.py. If they're in
# training too, your eval numbers are meaningless (this was happening before).
EVAL_TRAJ_START, EVAL_TRAJ_END = 50, 80

def discover_trajectories(data_root="../data_id"):
    """Scan disk for trajectories that actually exist, instead of assuming a
    hardcoded count (e.g. 600) that may not match reality. A mismatch here
    silently produces an EMPTY dataset -- which is exactly what happened
    before: val loss was fake (0.0 from an empty loader), checkpoint
    selection broke on epoch 1, and the "trained" model was never actually
    the best one.

    NOTE: kept as a plain function (not module-level code) deliberately.
    On Windows, DataLoader(num_workers>0) spawns worker processes that
    re-import this whole script. Anything executed at module level (not
    inside a function or the __main__ guard) reruns once per worker per
    epoch phase -- which is why you saw "Discovered 250 trajectories"
    printed 4x per epoch. Keeping this as a function called only from
    main() avoids that.
    """
    existing = []
    i = 0
    misses_in_a_row = 0
    while misses_in_a_row < 5 and i < 5000:
        d = os.path.join(data_root, f'traj-{i}')
        if os.path.isdir(d):
            existing.append(i)
            misses_in_a_row = 0
        else:
            misses_in_a_row += 1
        i += 1
    return existing


def get_train_val_split():
    all_traj = discover_trajectories()
    assert len(all_traj) > 0, "No trajectories found under ../data_id -- check the path"
    print(f"Discovered {len(all_traj)} trajectories on disk (range {min(all_traj)}-{max(all_traj)})")

    train_traj = [i for i in all_traj if not (EVAL_TRAJ_START <= i < EVAL_TRAJ_END)]
    n_val = max(1, int(0.1 * len(train_traj)))   # 10% of remaining for validation
    val_traj = train_traj[-n_val:]
    train_traj = train_traj[:-n_val]

    assert len(val_traj) > 0, "No validation trajectories -- check data_id path/contents"
    assert len(train_traj) > 0, "No training trajectories -- check data_id path/contents"
    return train_traj, val_traj


class LatentMLP(nn.Module):
    """Predicts a DELTA in latent space, not the absolute next latent.

    z_{t+1} = z_t + f(window of latents)

    This is much easier to learn than absolute latents because the model
    only has to model the *change*, and it means "predict nothing" (delta=0)
    is a sane initial guess, rather than random noise.
    """
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
        # latent_seq: (B, T, D) — T should equal self.context (sliding window)
        B, T, D = latent_seq.shape
        x = latent_seq.reshape(B, T * D)
        delta = self.net(x)
        last_latent = latent_seq[:, -1, :]
        return last_latent + delta


def teacher_forcing_prob(epoch, total_epochs, max_prob=0.75):
    """Probability of feeding the model's OWN prediction back in (instead of
    ground truth) at each rollout step, ramped up over training.

    Early on: mostly teacher forcing (stable gradients, learns basic dynamics).
    Later on: mostly autoregressive (matches how the model is actually used
    at eval/rollout time, so it learns to be robust to its own errors).
    """
    ramp_epochs = total_epochs * 0.6
    return min(max_prob, max_prob * epoch / ramp_epochs)


def rollout_curriculum(epoch, total_epochs, min_r=2, max_r=MAX_ROLLOUT):
    """How many steps ahead to unroll and backprop through this epoch.
    Starts short (easier optimization), grows to the full horizon.
    """
    ramp_epochs = total_epochs * 0.6
    r = min_r + (max_r - min_r) * min(1.0, epoch / ramp_epochs)
    return max(min_r, min(max_r, int(round(r))))


@torch.no_grad()
def encode_sequence(ae, frames):
    # frames: (B, T, C, H, W) -> (B, T, D)
    B, T, C, H, W = frames.shape
    return torch.stack([ae.encode(frames[:, t]) for t in range(T)], dim=1)


def run_epoch(mlp, ae, loader, optimizer, criterion, device, epoch, total_epochs, train=True):
    mlp.train(train)
    total_loss = 0.0
    n_batches = 0

    current_rollout = rollout_curriculum(epoch, total_epochs)
    tf_prob = teacher_forcing_prob(epoch, total_epochs)

    assert len(loader) > 0, (
        f"Loader for epoch {epoch+1} ({'train' if train else 'val'}) has zero batches. "
        f"This previously caused a silent fake val_loss=0.0 that corrupted checkpoint "
        f"selection -- failing loudly instead. Check dataset size / paths."
    )

    for input_seq, target_seq in tqdm(loader, desc=f"Epoch {epoch+1} ({'train' if train else 'val'})"):
        input_seq = input_seq.to(device)          # (B, CONTEXT, C, H, W)
        target_seq = target_seq.to(device)         # (B, MAX_ROLLOUT, C, H, W)

        with torch.no_grad():
            window = encode_sequence(ae, input_seq)                      # (B, CONTEXT, D)
            target_latents = encode_sequence(ae, target_seq[:, :current_rollout])  # (B, R, D)

        if train:
            optimizer.zero_grad()

        step_loss = 0.0
        cur_window = window
        for step in range(current_rollout):
            pred_latent = mlp(cur_window)                     # (B, D)
            step_loss = step_loss + criterion(pred_latent, target_latents[:, step])

            # scheduled sampling: feed own prediction vs. ground truth
            use_pred = train and (random.random() < tf_prob)
            next_latent = pred_latent if use_pred else target_latents[:, step]
            cur_window = torch.cat([cur_window[:, 1:, :], next_latent.detach().unsqueeze(1)
                                     if use_pred else next_latent.unsqueeze(1)], dim=1)
            # note: when NOT using own prediction, next_latent (=ground truth) doesn't need
            # detaching since it isn't part of the graph; when using own prediction, we
            # deliberately detach it from the window so gradients only flow through the
            # *current* step's prediction, keeping memory bounded (truncated BPTT).

        loss = step_loss / current_rollout

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])

    train_traj, val_traj = get_train_val_split()
    train_dirs = [os.path.join("..", "data_id", f'traj-{i}') for i in train_traj]
    val_dirs = [os.path.join("..", "data_id", f'traj-{i}') for i in val_traj]

    train_dataset = FramePredictionDataset(train_dirs, context=CONTEXT, rollout=MAX_ROLLOUT, transform=transform)
    val_dataset = FramePredictionDataset(val_dirs, context=CONTEXT, rollout=MAX_ROLLOUT, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                              num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    ae = ConvAutoencoder(latent_dim=LATENT_DIM).to(device)
    ae.load_state_dict(torch.load("../models/autoencoder/best.pth", map_location=device, weights_only=True))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    mlp = LatentMLP(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, context=CONTEXT).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()

    os.makedirs("../models/latent_mlp", exist_ok=True)
    best_val_loss = float('inf')

    print("Training Latent + MLP (rollout curriculum + scheduled sampling + delta prediction)...")

    for epoch in range(EPOCHS):
        train_loss = run_epoch(mlp, ae, train_loader, optimizer, criterion, device, epoch, EPOCHS, train=True)
        val_loss = run_epoch(mlp, ae, val_loader, optimizer, criterion, device, epoch, EPOCHS, train=False)

        r = rollout_curriculum(epoch, EPOCHS)
        tf = teacher_forcing_prob(epoch, EPOCHS)
        print(f"Epoch {epoch+1} - train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f} "
              f"| rollout={r} | tf_prob={tf:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(mlp.state_dict(), "../models/latent_mlp/best.pth")
            print(f"  -> new best (val {val_loss:.6f}), saved to best.pth")

    torch.save(mlp.state_dict(), "../models/latent_mlp/last.pth")
    print("✅ Training finished! Best checkpoint: ../models/latent_mlp/best.pth")


if __name__ == "__main__":
    main()