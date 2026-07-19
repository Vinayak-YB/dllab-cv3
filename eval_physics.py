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
from model import SequenceEncoderDecoder, SpatioTemporalTransformera
from dataset import FramePredictionDataset, TrajectoryPredictionDataset
from model import SequenceEncoderDecoder


def frame_prediction(model, val_loader, device):
    """
    Predict next frame given previous groundtruth frames.
    Supports both SequenceEncoderDecoder (LSTM/GRU) and SpatioTemporalTransformer.
    """
    target_frames = []
    positions = []
    velocities = []

    predicted_frames = []
    hidden_states = []

    is_stt = isinstance(model, SpatioTemporalTransformer)

    for input_seq, target_frame, position, velocity in tqdm.tqdm(val_loader, desc='Frame'):
        target_frames.append(target_frame.squeeze(1))
        positions.append(position.squeeze(1))
        velocities.append(velocity.squeeze(1))

        with torch.inference_mode():
            input_seq = input_seq.to(device)

            if is_stt:
                predicted_frame = model(input_seq)
                predicted_frames.append(predicted_frame.clamp(0, 1).cpu())

                # Dummy hidden state so downstream code still works
                hidden = torch.zeros(
                    predicted_frame.shape[0], 1, 2, 2,
                    device=predicted_frame.device
                )
                hidden_states.append(hidden.cpu())

            else:
                predicted_frame, hidden_state = model(
                    input_seq,
                    return_last_hidden_state=True
                )

                predicted_frames.append(predicted_frame.clamp(0, 1).cpu())
                hidden_states.append(
                    F.adaptive_avg_pool2d(hidden_state, (2, 2)).cpu()
                )

    target_frames = torch.cat(target_frames, dim=0)
    positions = torch.cat(positions, dim=0)
    velocities = torch.cat(velocities, dim=0)

    predicted_frames = torch.cat(predicted_frames, dim=0)
    hidden_states = torch.cat(hidden_states, dim=0)

    return target_frames, predicted_frames, hidden_states, positions, velocities


def trajectory_prediction(model, val_loader, device):
    """
    Predict next frame given previous predicted frames.
    Supports both SequenceEncoderDecoder (LSTM/GRU) and SpatioTemporalTransformer.
    """
    target_frames = []
    positions = []
    velocities = []

    predicted_frames = []
    hidden_states = []

    is_stt = isinstance(model, SpatioTemporalTransformer)

    for input_seq, target_seq, position, velocity in tqdm.tqdm(val_loader, desc='Trajectory'):
        target_frames.append(target_seq.squeeze(0))
        positions.append(position.squeeze(0))
        velocities.append(velocity.squeeze(0))

        with torch.inference_mode():
            input_seq = input_seq.to(device)

            for _ in range(target_seq.shape[1]):

                if is_stt:
                    predicted_frame = model(input_seq)
                    predicted_frames.append(predicted_frame.clamp(0, 1).cpu())

                    hidden = torch.zeros(
                        predicted_frame.shape[0], 1, 2, 2,
                        device=predicted_frame.device
                    )
                    hidden_states.append(hidden.cpu())

                else:
                    predicted_frame, hidden_state = model(
                        input_seq,
                        return_last_hidden_state=True
                    )

                    predicted_frames.append(predicted_frame.clamp(0, 1).cpu())
                    hidden_states.append(
                        F.adaptive_avg_pool2d(hidden_state, (2, 2)).cpu()
                    )

                input_seq = torch.cat(
                    [input_seq[:, 1:], predicted_frame.unsqueeze(1)],
                    dim=1
                )

    target_frames = torch.cat(target_frames, dim=0)
    positions = torch.cat(positions, dim=0)
    velocities = torch.cat(velocities, dim=0)

    predicted_frames = torch.cat(predicted_frames, dim=0)
    hidden_states = torch.cat(hidden_states, dim=0)

    return target_frames, predicted_frames, hidden_states, positions, velocities


def find_circle_center_com(image_tensor):
    """
    Find the center of a dark circle in a grayscale image tensor with white background.
    
    Args:
        image_tensor: A PyTorch tensor of shape (1, H, W) representing a grayscale image
                     with values in [0, 1] where 0 is black and 1 is white
    
    Returns:
        tuple: (y, x) coordinates of the circle center
    """
    # Convert to numpy for easier processing
    if image_tensor.dim() == 3 and image_tensor.size(0) == 1:
        # Remove channel dimension if present
        image = image_tensor.squeeze(0).cpu().numpy()
    else:
        image = image_tensor.cpu().numpy()
    
    # Invert the image so the circle becomes bright (easier to find centroid)
    # inverted = 1 - image
    inverted = image < 0.65
    
    # Calculate the center of mass (centroid)
    total_mass = inverted.sum()
    if total_mass == 0:
        return torch.tensor([torch.nan, torch.nan])  # No circle found
    
    # Calculate weighted coordinates
    y_indices, x_indices = torch.meshgrid(torch.arange(image.shape[0]), torch.arange(image.shape[1]), indexing='ij')
    y_indices, x_indices = y_indices.numpy(), x_indices.numpy()
    
    center_y = (y_indices * inverted).sum() / total_mass
    center_x = (x_indices * inverted).sum() / total_mass
    
    return torch.tensor([center_x + 0.5, 127.5 - center_y])


def find_circle_center_hough(image_tensor, use_com_fallback=True):
    """
    Find the center of a dark circle in a grayscale image tensor using Hough Circle Transform with sub-pixel accuracy.
    
    Args:
        image_tensor: A PyTorch tensor of shape (1, H, W) representing a grayscale image
                     with values in [0, 1] where 0 is black and 1 is white
    
    Returns:
        tuple: (x, y) coordinates of the circle center with sub-pixel accuracy
    """
    # Convert to numpy for OpenCV processing
    if image_tensor.dim() == 3 and image_tensor.size(0) == 1:
        # Remove channel dimension if present
        image = image_tensor.squeeze(0).cpu().numpy()
    else:
        image = image_tensor.cpu().numpy()
    
    # Convert to 8-bit image and invert so circle is dark on light background
    image_8bit = (image * 255).astype(np.uint8)
    # Create a binary image with a threshold
    binary = (image < 0.65).astype(np.uint8) * 255
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    # blurred = binary
    
    # Use Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=0.1,
        minDist=50,
        param1=50,
        param2=10,
        minRadius=4,
        maxRadius=10
    )
    
    # If circles are found
    if circles is not None:
        # Take the first circle
        x, y, _ = circles[0, 0]
        
        # Refine the circle center using cv2.cornerSubPix for sub-pixel accuracy
        mask = np.zeros_like(image_8bit, dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), 10, 255, -1)  # Create a mask around the detected circle
        moments = cv2.moments(mask * image_8bit)
        if moments["m00"] != 0:
            refined_x = moments["m10"] / moments["m00"]
            refined_y = moments["m01"] / moments["m00"]
        else:
            refined_x, refined_y = x, y
        
        return torch.tensor([refined_x + 0.5, 127.5 - refined_y]).double().round(decimals=1)  # Adjust y coordinate as in the original function
    elif use_com_fallback:
        # Fallback to the original method if no circles are found
        return find_circle_center_com(image_tensor)
    
    return torch.tensor([torch.nan, torch.nan])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate physic accuracy of pretrained model.')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the pretrained model.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--val_pct', type=float, default=0.1, help='Percentage of data to use for validation.')
    args = parser.parse_args()

    # Load the model
    with open(os.path.join(args.model_dir, 'args.json')) as f:
        config = json.load(f)

    if args.data_dir:
        n_trajectories = len(os.listdir(args.data_dir))
        sequence_dirs = [os.path.join(args.data_dir, f'traj-{i}') for i in range(n_trajectories)]
        num_val_trajectories = int(len(sequence_dirs) * args.val_pct)
        train_dirs = sequence_dirs[:-num_val_trajectories]
        val_dirs = sequence_dirs[-num_val_trajectories:]
    else:
        args.data_dir = config['data_dir']
        train_dirs = config['train_dirs']
        val_dirs = config['val_dirs']

    # subsample training set to avoid overhead
    assert len(val_dirs) <= len(train_dirs), f'Validation set is larger than training set: {len(val_dirs)} > {len(train_dirs)}'
    train_dirs = random.sample(train_dirs, 2 * len(val_dirs))

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
        else:
            raise ValueError(f'Invalid mode: {mode}')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SequenceEncoderDecoder.from_pretrained(args.model_dir, device=device)

        results[mode] = {}

        compute_prediction = frame_prediction if mode == 'frame' else trajectory_prediction
        target_frames, predicted_frames, hidden_states, positions, velocities = compute_prediction(model, val_loader, device)
        # print(f'Predicted frames shape: {predicted_frames.shape}')
        # print(f'Target frames shape: {target_frames.shape}')
        # print(f'Hidden states shape: {hidden_states.shape}')
        # print(f'Positions shape: {positions.shape}')
        # print(f'Velocities shape: {velocities.shape}')

        # estimate predicted positions and velocities from circle centers
        predicted_positions = torch.stack([find_circle_center_hough(frame) for frame in predicted_frames], dim=0)
        predicted_velocities = torch.diff(predicted_positions, dim=0)

        # also estimate target positions and velocities instead of look-up to avoid inconsistencies between simulated
        # and observed velocities (e.g., the ball might be really fast before hitting the ground but the observed
        # velocity is low due to only a small observed change in position)
        target_positions = torch.stack([find_circle_center_hough(frame) for frame in target_frames], dim=0)
        target_velocities = torch.diff(target_positions, dim=0)

        position_dists = (predicted_positions - target_positions).norm(p=2, dim=1)
        velocity_dists = (predicted_velocities - target_velocities).norm(p=2, dim=1)
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

        # evaluate to which extent the simulated positions and velocities are decodable from the hidden states
        *_, hidden_states_train, positions_train, velocities_train = compute_prediction(model, train_loader, device)
        
        # train data for regression
        X_train = hidden_states_train.view(hidden_states_train.shape[0], -1).detach().numpy()
        # X_train = hidden_states_train.detach().numpy()
        y_train_pos = positions_train.detach().numpy()
        y_train_vel = velocities_train.detach().numpy()

        # test data for regression
        X_test = hidden_states.view(hidden_states.shape[0], -1).detach().numpy()
        # X_test = hidden_states.detach().numpy()
        y_test_pos = positions.detach().numpy()
        y_test_vel = velocities.detach().numpy()

        # print(f'X_train shape: {X_train.shape}')
        # print(f'X_test shape: {X_test.shape}')

        # train regression model
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        # pca = PCA(n_components=512)
        # X_train = pca.fit_transform(X_train)
        # reg_pos = Ridge(alpha=50).fit(X_train, y_train_pos)
        # reg_vel = Ridge(alpha=50).fit(X_train, y_train_vel)

        reg_pos = LinearRegression().fit(X_train, y_train_pos)
        reg_vel = LinearRegression().fit(X_train, y_train_vel)

        X_test = scaler.transform(X_test)
        # X_test = pca.transform(X_test)
        y_pred_pos = reg_pos.predict(X_test)
        y_pred_vel = reg_vel.predict(X_test)

        # calculate metrics
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

    # Save results to a JSON file
    os.makedirs(os.path.join(args.model_dir, 'physics'), exist_ok=True)
    with open(os.path.join(args.model_dir, 'physics', 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
