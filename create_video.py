from model import SequenceEncoderDecoder
from dataset import FramePredictionDataset, TrajectoryPredictionDataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import torch
import argparse
import json
import os
from pathlib import Path


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def frame_prediction(model, val_loader, device):
    """Predict next frames from ground-truth context windows."""
    target_frames = []
    predicted_frames = []

    for input_seq, target_frame in val_loader:
        with torch.inference_mode():
            prediction = model(input_seq.to(device)).clamp(0, 1).cpu()
            target_frames.append(target_frame.cpu())
            predicted_frames.append(prediction)

    if not target_frames:
        raise RuntimeError("The frame dataset produced no samples.")

    return (
        torch.cat(target_frames, dim=0),
        torch.cat(predicted_frames, dim=0),
    )


def trajectory_prediction(model, val_loader, device, max_frames):
    """Run an autoregressive rollout on one selected trajectory."""
    for input_seq, target_seq in val_loader:
        with torch.inference_mode():
            steps = min(target_seq.shape[1], max_frames)

            target_frames = target_seq[:, :steps].squeeze(0).cpu()
            input_seq = input_seq.to(device)
            predicted_frames = []

            for _ in range(steps):
                predicted_frame = model(input_seq).clamp(0, 1)
                predicted_frames.append(predicted_frame.cpu())

                input_seq = torch.cat(
                    [
                        input_seq[:, 1:],
                        predicted_frame.unsqueeze(1),
                    ],
                    dim=1,
                )

            return target_frames, torch.cat(predicted_frames, dim=0)

    raise RuntimeError("The trajectory dataset produced no samples.")


def find_trajectory_dirs(data_dir):
    sequence_dirs = sorted(
        str(path)
        for path in Path(data_dir).glob("traj-*")
        if path.is_dir()
    )

    if not sequence_dirs:
        raise ValueError(f"No trajectory directories found in: {data_dir}")

    return sequence_dirs


def tensor_to_pil(frame_tensor):
    """
    Convert a [1,H,W] or [H,W] tensor in [0,1] to a grayscale PIL image.
    """
    frame = frame_tensor.detach().cpu()

    if frame.ndim == 3:
        frame = frame.squeeze(0)

    frame = frame.clamp(0, 1)
    array = (frame.numpy() * 255).astype("uint8")
    return Image.fromarray(array, mode="L")


def make_side_by_side_frame(pred_frame, target_frame, frame_idx, mode):
    """
    Create one GIF frame:
    black background, white text, prediction and target both shown in grayscale.
    """
    pred_img = tensor_to_pil(pred_frame)
    target_img = tensor_to_pil(target_frame)

    font = ImageFont.load_default()

    pad = 16
    title_h = 24
    label_h = 18
    footer_h = 20

    panel_w, panel_h = pred_img.size
    canvas_w = panel_w * 2 + pad * 3
    canvas_h = title_h + label_h + panel_h + footer_h + pad * 3

    canvas = Image.new("L", (canvas_w, canvas_h), color=0)
    draw = ImageDraw.Draw(canvas)

    title = (
        "Trajectory rollout"
        if mode == "trajectory"
        else "Frame prediction"
    )
    draw.text((pad, pad // 2), title, fill=255, font=font)

    pred_x = pad
    target_x = panel_w + pad * 2
    img_y = title_h + label_h + pad

    draw.text((pred_x, title_h), "Prediction", fill=255, font=font)
    draw.text((target_x, title_h), "Target", fill=255, font=font)

    canvas.paste(pred_img, (pred_x, img_y))
    canvas.paste(target_img, (target_x, img_y))

    draw.rectangle(
        [pred_x - 1, img_y - 1, pred_x + panel_w, img_y + panel_h],
        outline=255,
        width=1,
    )
    draw.rectangle(
        [target_x - 1, img_y - 1, target_x + panel_w, img_y + panel_h],
        outline=255,
        width=1,
    )

    draw.text(
        (pad, canvas_h - footer_h),
        f"Frame {frame_idx}",
        fill=255,
        font=font,
    )

    return canvas


def save_gif(predicted_frames, target_frames, save_path, fps, mode):
    frames = []

    n_frames = min(len(predicted_frames), len(target_frames))
    if n_frames == 0:
        raise RuntimeError("No frames available to save.")

    for idx in range(n_frames):
        canvas = make_side_by_side_frame(
            predicted_frames[idx],
            target_frames[idx],
            idx,
            mode,
        )
        frames.append(canvas)

    duration_ms = int(1000 / fps)

    # GIF wants mode "P" or "L". We already use grayscale "L".
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create GIFs for recurrent models without ffmpeg."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Run directory containing args.json and checkpoints/.",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="best.ckpt",
        help="Checkpoint filename, e.g. best.ckpt.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Dataset split directory containing traj-* folders.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["frame", "trajectory"],
        default="trajectory",
        help="frame = one-step from GT context, trajectory = autoregressive rollout",
    )
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=10)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--trajectory_index", type=int, default=0)
    parser.add_argument("--frame_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    args_path = Path(args.model_dir) / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(f"Run configuration not found: {args_path}")

    with args_path.open("r", encoding="utf-8") as file:
        _ = json.load(file)

    sequence_dirs = find_trajectory_dirs(args.data_dir)

    if not 0 <= args.trajectory_index < len(sequence_dirs):
        raise ValueError(
            f"trajectory_index={args.trajectory_index}, "
            f"but only {len(sequence_dirs)} trajectories are available"
        )

    # For qualitative visualization, select one concrete held-out trajectory.
    selected_dirs = [sequence_dirs[args.trajectory_index]]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
        ]
    )

    if args.mode == "frame":
        val_dataset = FramePredictionDataset(
            selected_dirs,
            context=args.context,
            transform=transform,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.frame_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        val_dataset = TrajectoryPredictionDataset(
            selected_dirs,
            context=args.context,
            transform=transform,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SequenceEncoderDecoder.from_pretrained(
        args.model_dir,
        ckpt_name=args.ckpt_name,
        device=device,
    )
    model.eval()

    if args.mode == "frame":
        target_frames, predicted_frames = frame_prediction(
            model,
            val_loader,
            device,
        )
        target_frames = target_frames[: args.max_frames]
        predicted_frames = predicted_frames[: args.max_frames]
    else:
        target_frames, predicted_frames = trajectory_prediction(
            model,
            val_loader,
            device,
            max_frames=args.max_frames,
        )

    print(f"Predicted frames shape: {predicted_frames.shape}")
    print(f"Target frames shape: {target_frames.shape}")

    video_dir = Path(args.model_dir) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    env_name = Path(args.data_dir).parent.name
    checkpoint_stem = Path(args.ckpt_name).stem
    save_path = (
        video_dir
        / f"{env_name}_{checkpoint_stem}_{args.mode}_traj{args.trajectory_index:03d}.gif"
    )

    save_gif(
        predicted_frames=predicted_frames,
        target_frames=target_frames,
        save_path=save_path,
        fps=args.fps,
        mode=args.mode,
    )

    print(f"Saved GIF to: {save_path}")


if __name__ == "__main__":
    main()