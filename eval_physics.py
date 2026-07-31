import argparse
import random
import json
import os

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import torch
import tqdm
import cv2
from scipy.optimize import linear_sum_assignment

from dataset import FramePredictionDataset, TrajectoryPredictionDataset
from model import SequenceEncoderDecoder


def frame_prediction(model, val_loader, device):
    """
    Predict next frame given previous groundtruth frames.
    """
    target_frames = []
    positions = []
    velocities = []

    predicted_frames = []
    hidden_states = []

    for input_seq, target_frame, position, velocity in tqdm.tqdm(val_loader, desc='Frame'):
        target_frames.append(target_frame.squeeze(1))
        positions.append(position.squeeze(1))
        velocities.append(velocity.squeeze(1))

        with torch.inference_mode():
            predicted_frame, hidden_state = model(input_seq.to(device), return_last_hidden_state=True)
            predicted_frames.append(predicted_frame.clamp(0, 1).cpu())
            hidden_states.append(F.adaptive_avg_pool2d(hidden_state, (2, 2)).cpu())

    target_frames = torch.cat(target_frames, dim=0)
    positions = torch.cat(positions, dim=0)
    velocities = torch.cat(velocities, dim=0)

    predicted_frames = torch.cat(predicted_frames, dim=0)
    hidden_states = torch.cat(hidden_states, dim=0)

    return target_frames, predicted_frames, hidden_states, positions, velocities


def trajectory_prediction(model, val_loader, device):
    """
    Predict next frame given previous predicted frames.
    """
    target_frames = []
    positions = []
    velocities = []

    predicted_frames = []
    hidden_states = []

    for input_seq, target_seq, position, velocity in tqdm.tqdm(val_loader, desc='Trajectory'):
        target_frames.append(target_seq.squeeze(0))
        positions.append(position.squeeze(0))
        velocities.append(velocity.squeeze(0))

        with torch.inference_mode():
            input_seq = input_seq.to(device)
            for _ in range(target_seq.shape[1]):
                predicted_frame, hidden_state = model(input_seq, return_last_hidden_state=True)
                predicted_frames.append(predicted_frame.clamp(0, 1).cpu())
                hidden_states.append(F.adaptive_avg_pool2d(hidden_state, (2, 2)).cpu())
                input_seq = torch.cat([input_seq[:, 1:], predicted_frame.unsqueeze(0)], dim=1)

    target_frames = torch.cat(target_frames, dim=0)
    positions = torch.cat(positions, dim=0)
    velocities = torch.cat(velocities, dim=0)

    predicted_frames = torch.cat(predicted_frames, dim=0)
    hidden_states = torch.cat(hidden_states, dim=0)

    return target_frames, predicted_frames, hidden_states, positions, velocities


def find_circle_center_com(image_tensor, threshold=0.35):
    """
    Find center of mass (COM) for single foreground object.
    Dynamically auto-detects background brightness.
    """
    if image_tensor.dim() == 3 and image_tensor.size(0) == 1:
        image = image_tensor.squeeze(0).cpu().numpy()
    else:
        image = image_tensor.cpu().numpy()

    is_dark_bg = image.mean() < 0.5
    inverted = (image > threshold) if is_dark_bg else (image < (1.0 - threshold))

    total_mass = inverted.sum()
    if total_mass == 0:
        return torch.tensor([[torch.nan, torch.nan]])

    y_indices, x_indices = torch.meshgrid(
        torch.arange(image.shape[0]), torch.arange(image.shape[1]), indexing='ij'
    )
    y_indices, x_indices = y_indices.numpy(), x_indices.numpy()

    center_y = (y_indices * inverted).sum() / total_mass
    center_x = (x_indices * inverted).sum() / total_mass

    return torch.tensor([[center_x + 0.5, 127.5 - center_y]])


def find_circle_center_hough(image_tensor, use_com_fallback=True, num_objects=1):
    """
    Find centers of N objects in an image tensor using Hough Circle Transform.
    Pads missing detections with NaNs to prevent distorted distance metrics.
    """
    if image_tensor.dim() == 3 and image_tensor.size(0) == 1:
        image = image_tensor.squeeze(0).cpu().numpy()
    else:
        image = image_tensor.cpu().numpy()

    image_8bit = (image * 255).astype(np.uint8)

    is_dark_bg = image.mean() < 0.5
    if is_dark_bg:
        binary = (image > 0.35).astype(np.uint8) * 255
    else:
        binary = (image < 0.65).astype(np.uint8) * 255

    blurred = cv2.GaussianBlur(binary, (5, 5), 0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=0.1,
        minDist=10 if num_objects > 1 else 50,
        param1=50,
        param2=10,
        minRadius=3,
        maxRadius=15
    )

    extracted_centers = []
    if circles is not None:
        for c in circles[0, :num_objects]:
            x, y, _ = c
            mask = np.zeros_like(image_8bit, dtype=np.uint8)
            cv2.circle(mask, (int(x), int(y)), 10, 255, -1)
            
            moments = cv2.moments(mask * binary)
            if moments["m00"] != 0:
                refined_x = moments["m10"] / moments["m00"]
                refined_y = moments["m01"] / moments["m00"]
            else:
                refined_x, refined_y = x, y

            extracted_centers.append([refined_x + 0.5, 127.5 - refined_y])

    # Single-object COM fallback
    if len(extracted_centers) == 0 and num_objects == 1 and use_com_fallback:
        com_center = find_circle_center_com(image_tensor)
        if not torch.isnan(com_center).any():
            extracted_centers.append(com_center[0].tolist())

    # FIXED: Pad missing detection slots with NaNs instead of duplicating
    while len(extracted_centers) < num_objects:
        extracted_centers.append([np.nan, np.nan])

    res = torch.tensor(extracted_centers).double().round(decimals=1)
    return res


def compute_matched_distance(pred_pos, target_pos):
    """
    Computes Euclidean distance using Hungarian minimal matching.
    Returns NaN if any object detection in the frame failed (contains NaN).
    """
    if torch.isnan(pred_pos).any() or torch.isnan(target_pos).any():
        return torch.tensor(torch.nan)

    p_np = pred_pos.detach().cpu().numpy().reshape(-1, 2)
    t_np = target_pos.detach().cpu().numpy().reshape(-1, 2)

    cost_matrix = np.linalg.norm(p_np[:, None, :] - t_np[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return torch.tensor(cost_matrix[row_ind, col_ind].mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate physic accuracy of pretrained model.')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the pretrained model.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to evaluation dataset directory.')
    parser.add_argument('--probe_train_dir', type=str, default=None, help='Optional separate path to train linear probes.')
    parser.add_argument('--val_pct', type=float, default=0.1, help='Percentage of data to use for validation.')
    parser.add_argument('--num_objects', type=int, default=None, help='Number of objects. Auto-detected if None.')
    args = parser.parse_args()

    # Auto-detect object count
    if args.num_objects is None:
        check_path = f"{args.data_dir} {args.model_dir}".lower()
        if 'billiard' in check_path:
            args.num_objects = 2
            print("🔹 Auto-configured for multi-object environment: Billiards (num_objects = 2)")
        else:
            args.num_objects = 1
            print("🔹 Auto-configured for single-object environment (num_objects = 1)")

    # Load model configuration
    with open(os.path.join(args.model_dir, 'args.json')) as f:
        config = json.load(f)

    # Set up evaluation validation directories
    n_trajectories = len(os.listdir(args.data_dir))
    sequence_dirs = [os.path.join(args.data_dir, d) for d in sorted(os.listdir(args.data_dir)) if os.path.isdir(os.path.join(args.data_dir, d))]
    num_val_trajectories = max(1, int(len(sequence_dirs) * args.val_pct))
    val_dirs = sequence_dirs[-num_val_trajectories:]

    # Set up probing training directories (external or default dataset)
    if args.probe_train_dir:
        print(f"🔹 Using explicit probing training data: {args.probe_train_dir}")
        n_probe_trajectories = len(os.listdir(args.probe_train_dir))
        train_dirs = [os.path.join(args.probe_train_dir, d) for d in sorted(os.listdir(args.probe_train_dir)) if os.path.isdir(os.path.join(args.probe_train_dir, d))]
    else:
        train_dirs = sequence_dirs[:-num_val_trajectories]

    # Subsample probe training data if larger than validation
    if len(train_dirs) > len(val_dirs):
        train_dirs = random.sample(train_dirs, min(len(train_dirs), 2 * len(val_dirs)))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
    ])

    results = {}
    for mode in ['frame', 'trajectory']:
        if mode == 'frame':
            train_dataset = FramePredictionDataset(train_dirs, context=5, transform=transform, return_state=True)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
            val_dataset = FramePredictionDataset(val_dirs, context=5, transform=transform, return_state=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        elif mode == 'trajectory':
            train_dataset = TrajectoryPredictionDataset(train_dirs, context=5, transform=transform, return_state=True)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
            val_dataset = TrajectoryPredictionDataset(val_dirs, context=5, transform=transform, return_state=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SequenceEncoderDecoder.from_pretrained(args.model_dir, device=device)

        results[mode] = {}

        compute_prediction = frame_prediction if mode == 'frame' else trajectory_prediction
        target_frames, predicted_frames, hidden_states, positions, velocities = compute_prediction(model, val_loader, device)

        # Estimate predicted and target positions/velocities
        predicted_positions = torch.stack([find_circle_center_hough(frame, num_objects=args.num_objects) for frame in predicted_frames], dim=0)
        predicted_velocities = torch.diff(predicted_positions, dim=0)

        target_positions = torch.stack([find_circle_center_hough(frame, num_objects=args.num_objects) for frame in target_frames], dim=0)
        target_velocities = torch.diff(target_positions, dim=0)

        # Compute matched endpoint errors
        position_dists = torch.stack([compute_matched_distance(p, t) for p, t in zip(predicted_positions, target_positions)])
        velocity_dists = torch.stack([compute_matched_distance(p, t) for p, t in zip(predicted_velocities, target_velocities)])

        position_failures = position_dists.isnan().sum().item()
        velocity_failures = velocity_dists.isnan().sum().item()
        position_aee = position_dists.nanmean().item()
        velocity_aee = velocity_dists.nanmean().item()

        results[mode] = {
            'from_observed': {
                'position_aee': position_aee,
                'velocity_aee': velocity_aee,
                'position_total': len(predicted_positions),
                'velocity_total': len(predicted_velocities),
                'position_failures': position_failures,
                'velocity_failures': velocity_failures,
            }
        }
        print('Observed physics:')
        print(f'Average Position Endpoint Error ({mode}): {position_aee}')
        print(f'Average Velocity Endpoint Error ({mode}): {velocity_aee}')
        print(f'Position Failures ({mode}): {position_failures} / {len(predicted_positions)}')
        print(f'Velocity Failures ({mode}): {velocity_failures} / {len(predicted_velocities)}')
        print()

        # Evaluate latent physics
        *_, hidden_states_train, positions_train, velocities_train = compute_prediction(model, train_loader, device)

        X_train = hidden_states_train.view(hidden_states_train.shape[0], -1).detach().numpy()
        y_train_pos = positions_train.detach().numpy()
        y_train_vel = velocities_train.detach().numpy()

        X_test = hidden_states.view(hidden_states.shape[0], -1).detach().numpy()
        y_test_pos = positions.detach().numpy()
        y_test_vel = velocities.detach().numpy()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        reg_pos = Ridge(alpha=100.0).fit(X_train, y_train_pos)
        reg_vel = Ridge(alpha=100.0).fit(X_train, y_train_vel)

        X_test = scaler.transform(X_test)
        y_pred_pos = reg_pos.predict(X_test)
        y_pred_vel = reg_vel.predict(X_test)

        position_r2 = r2_score(y_test_pos, y_pred_pos)
        velocity_r2 = r2_score(y_test_vel, y_pred_vel)
        position_aee = (torch.tensor(y_pred_pos) - torch.tensor(y_test_pos)).norm(p=2, dim=1).mean().item()
        velocity_aee = (torch.tensor(y_pred_vel) - torch.tensor(y_test_vel)).norm(p=2, dim=1).mean().item()

        results[mode]['from_latent'] = {
            'position_r2': position_r2,
            'velocity_r2': velocity_r2,
            'position_aee': position_aee,
            'velocity_aee': velocity_aee,
        }
        print('Latent physics:')
        print(f'Average Position R2 ({mode}): {position_r2}')
        print(f'Average Velocity R2 ({mode}): {velocity_r2}')
        print(f'Average Position Endpoint Error ({mode}): {position_aee}')
        print(f'Average Velocity Endpoint Error ({mode}): {velocity_aee}')

        y_pred_pos = reg_pos.predict(X_train)
        y_pred_vel = reg_vel.predict(X_train)
        position_r2 = r2_score(y_train_pos, y_pred_pos)
        velocity_r2 = r2_score(y_train_vel, y_pred_vel)
        position_aee = (torch.tensor(y_pred_pos) - torch.tensor(y_train_pos)).norm(p=2, dim=1).mean().item()
        velocity_aee = (torch.tensor(y_pred_vel) - torch.tensor(y_train_vel)).norm(p=2, dim=1).mean().item()
        print(f'Average Train Position R2 ({mode}): {position_r2}')
        print(f'Average Train Velocity R2 ({mode}): {velocity_r2}')
        print(f'Average Train Position Endpoint Error ({mode}): {position_aee}')
        print(f'Average Train Velocity Endpoint Error ({mode}): {velocity_aee}')
        print()

    # Save results to JSON
    os.makedirs(os.path.join(args.model_dir, 'physics'), exist_ok=True)
    with open(os.path.join(args.model_dir, 'physics', 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)