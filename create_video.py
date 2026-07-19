from model import SequenceEncoderDecoder
from dataset import FramePredictionDataset, TrajectoryPredictionDataset
from matplotlib.animation import FFMpegWriter
from torchvision import transforms
import torch.nn.functional as F
import torchshow as ts
import torch
import argparse
import json
import os


def frame_prediction(model, val_loader, device):
    """
    Predict next frame given previous groundtruth frames.
    """
    target_frames = []
    predicted_frames = []
    for input_seq, target_frame in val_loader:
        with torch.inference_mode():
            target_frames.append(target_frame.squeeze(1))
            predicted_frames.append(model(input_seq.to(device)).clamp(0, 1).cpu())

    target_frames = torch.cat(target_frames, dim=0)
    predicted_frames = torch.cat(predicted_frames, dim=0)
    return target_frames, predicted_frames


def trajectory_prediction(model, val_loader, device):
    """
    Predict next frame given previous predicted frames.
    """
    target_frames = []
    predicted_frames = []
    for input_seq, target_seq in val_loader:
        with torch.inference_mode():
            target_frames.append(target_seq.squeeze(0))
            input_seq = input_seq.to(device)
            for _ in range(target_seq.shape[1]):
                predicted_frame = model(input_seq)
                predicted_frames.append(predicted_frame.clamp(0, 1).cpu())
                input_seq = torch.cat([input_seq[:, 1:], predicted_frame.unsqueeze(0)], dim=1)

    target_frames = torch.cat(target_frames, dim=0)
    predicted_frames = torch.cat(predicted_frames, dim=0)
    return target_frames, predicted_frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model for frame prediction.')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory containing the model.')
    parser.add_argument('--ckpt_name', type=str, default='last.ckpt', help='Name of the model checkpoint.')
    parser.add_argument('--data_dir', type=str, help='Directory containing the dataset.')
    parser.add_argument('--val_pct', type=float, default=0.1, help='Percentage of data to use for validation.')
    parser.add_argument('--mode', type=str, choices=['frame', 'trajectory'], default='frame', help='Evaluation mode to use.')
    args = parser.parse_args()

    with open(os.path.join(args.model_dir, 'args.json')) as f:
        config = json.load(f)

    if args.data_dir:
        n_trajectories = len(os.listdir(args.data_dir))
        sequence_dirs = [os.path.join(args.data_dir, f'traj-{i}') for i in range(n_trajectories)]
        num_val_trajectories = int(len(sequence_dirs) * args.val_pct)
        val_dirs = sequence_dirs[-num_val_trajectories:]
    else:
        args.data_dir = config['data_dir']
        val_dirs = config['val_dirs']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
    ])
    if args.mode == 'frame':
        val_dataset = FramePredictionDataset(val_dirs, context=5, transform=transform)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    elif args.mode == 'trajectory':
        val_dataset = TrajectoryPredictionDataset(val_dirs, context=5, transform=transform)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    else:
        raise ValueError(f'Invalid mode: {args.mode}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SequenceEncoderDecoder.from_pretrained(args.model_dir, ckpt_name=args.ckpt_name, device=device)

    compute_prediction = frame_prediction if args.mode == 'frame' else trajectory_prediction
    target_frames, predicted_frames = compute_prediction(model, val_loader, device)
    print(f'Predicted frames shape: {predicted_frames.shape}')
    print(f'Target frames shape: {target_frames.shape}')

    # convert to blue / red
    target_frames_rgb = torch.cat([target_frames, target_frames, torch.ones_like(target_frames)], dim=1)
    predicted_frames_rgb = torch.cat([torch.ones_like(predicted_frames), predicted_frames, predicted_frames], dim=1)

    # pad with black border
    target_frames_rgb = F.pad(target_frames_rgb, (2, 2, 2, 2), value=0)
    predicted_frames_rgb = F.pad(predicted_frames_rgb, (2, 2, 2, 2), value=0)

    os.makedirs(os.path.join(args.model_dir, 'videos'), exist_ok=True)
    writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    ani = ts.show_video([predicted_frames_rgb, target_frames_rgb])
    ani.save(os.path.join(args.model_dir, 'videos', f'{args.data_dir}-{os.path.splitext(args.ckpt_name)[0]}-{args.mode}.mp4'), writer=writer)
