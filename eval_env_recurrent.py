import argparse
import json
import os
import random

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from eval_latent_flow import (
    physics_from_observed_frames_one_step,
    physics_from_observed_frames_rollout,
)
from model import SequenceEncoderDecoder
from utils.dataset import FramePredictionDataset, get_sequence_dirs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(
    model_dir: str,
    ckpt_name: str,
    device: torch.device,
) -> SequenceEncoderDecoder:
    checkpoint_path = os.path.join(
        model_dir,
        "checkpoints",
        ckpt_name,
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    args_path = os.path.join(
        model_dir,
        "args.json",
    )

    if not os.path.isfile(args_path):
        raise FileNotFoundError(
            f"Model configuration not found: {args_path}"
        )

    model = SequenceEncoderDecoder.from_pretrained(
        ckpt_dir=model_dir,
        ckpt_name=ckpt_name,
        device=str(device),
    )

    model.eval()

    return model


def build_loader(
    sequence_dirs,
    context: int,
    rollout: int,
    eval_stride: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = FramePredictionDataset(
        sequence_dirs=sequence_dirs,
        context=context,
        rollout=rollout,
        stride=eval_stride,
        grayscale=True,
        return_state=True,
    )

    if len(dataset) == 0:
        raise ValueError(
            "No valid evaluation windows were created. "
            f"context={context}, "
            f"rollout={rollout}, "
            f"stride={eval_stride}"
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict_one_step(
    model,
    dataloader,
    device,
):
    target_frames = []
    predicted_frames = []
    last_context_frames = []

    for input_seq, target_seq, _, _ in tqdm.tqdm(
        dataloader,
        desc="One-step evaluation",
    ):
        input_seq = input_seq.to(
            device,
            non_blocking=True,
        )

        pred_next = model(
            input_seq
        ).clamp(0, 1)

        target_frames.append(
            target_seq[:, 0].cpu()
        )

        predicted_frames.append(
            pred_next.cpu()
        )

        last_context_frames.append(
            input_seq[:, -1].cpu()
        )

    return (
        torch.cat(target_frames, dim=0),
        torch.cat(predicted_frames, dim=0),
        torch.cat(last_context_frames, dim=0),
    )


@torch.no_grad()
def predict_rollout(
    model,
    dataloader,
    device,
    rollout_steps: int,
):
    target_frames = []
    predicted_frames = []

    for input_seq, target_seq, _, _ in tqdm.tqdm(
        dataloader,
        desc=f"{rollout_steps}-step rollout evaluation",
    ):
        current_context = input_seq.to(
            device,
            non_blocking=True,
        )

        rollout_predictions = []

        for _ in range(rollout_steps):
            pred_next = model(
                current_context
            ).clamp(0, 1)

            rollout_predictions.append(
                pred_next.cpu()
            )

            current_context = torch.cat(
                [
                    current_context[:, 1:],
                    pred_next.unsqueeze(1),
                ],
                dim=1,
            )

        target_frames.append(
            target_seq.cpu()
        )

        predicted_frames.append(
            torch.stack(
                rollout_predictions,
                dim=1,
            )
        )

    return (
        torch.cat(target_frames, dim=0),
        torch.cat(predicted_frames, dim=0),
    )


def main(args) -> None:
    set_seed(args.seed)

    device = get_device()

    print(f"Using device: {device}")

    test_dirs = get_sequence_dirs(
        args.data_dir
    )

    if not test_dirs:
        raise ValueError(
            f"No valid trajectories found in: "
            f"{args.data_dir}"
        )

    print(f"Model directory:     {args.model_dir}")
    print(f"Checkpoint:          {args.ckpt_name}")
    print(f"Test directory:      {args.data_dir}")
    print(f"Test trajectories:   {len(test_dirs)}")
    print(f"Context:             {args.context}")
    print(f"Rollout steps:       {args.rollout_steps}")
    print(f"Evaluation stride:   {args.eval_stride}")
    print(f"Number of objects:   {args.num_objects}")

    model = load_model(
        model_dir=args.model_dir,
        ckpt_name=args.ckpt_name,
        device=device,
    )

    extractor_kwargs = {
        "threshold": args.threshold,
        "min_mass": args.min_mass,
        "fallback_thresholds": tuple(
            args.fallback_thresholds
        ),
        "use_topk_fallback": (
            not args.disable_topk_fallback
        ),
        "topk_ratio": args.topk_ratio,
        "debug": args.debug_threshold,
    }

    # ---------------------------------------------------------
    # One-step evaluation
    # ---------------------------------------------------------

    one_step_loader = build_loader(
        sequence_dirs=test_dirs,
        context=args.context,
        rollout=1,
        eval_stride=args.eval_stride,
        batch_size=args.frame_batch_size,
        num_workers=args.num_workers,
    )

    (
        target_frames,
        predicted_frames,
        last_context_frames,
    ) = predict_one_step(
        model=model,
        dataloader=one_step_loader,
        device=device,
    )

    one_step_metrics = (
        physics_from_observed_frames_one_step(
            target_frames=target_frames,
            predicted_frames=predicted_frames,
            last_context_frames=last_context_frames,
            num_objects=args.num_objects,
            **extractor_kwargs,
        )
    )

    print("\nOne-step observed-space metrics:")

    print(
        f"  Position AEE:      "
        f"{one_step_metrics['position_aee']:.4f}"
    )

    print(
        "  Position failures: "
        f"{one_step_metrics['position_failures']} / "
        f"{one_step_metrics['position_total']}"
    )

    del (
        target_frames,
        predicted_frames,
        last_context_frames,
        one_step_loader,
    )

    # ---------------------------------------------------------
    # Fixed-horizon rollout evaluation
    # ---------------------------------------------------------

    rollout_loader = build_loader(
        sequence_dirs=test_dirs,
        context=args.context,
        rollout=args.rollout_steps,
        eval_stride=args.eval_stride,
        batch_size=args.rollout_batch_size,
        num_workers=args.num_workers,
    )

    (
        target_frames,
        predicted_frames,
    ) = predict_rollout(
        model=model,
        dataloader=rollout_loader,
        device=device,
        rollout_steps=args.rollout_steps,
    )

    rollout_metrics = (
        physics_from_observed_frames_rollout(
            target_frames=target_frames,
            predicted_frames=predicted_frames,
            num_objects=args.num_objects,
            **extractor_kwargs,
        )
    )

    rollout_aggregate = rollout_metrics[
        "aggregate"
    ]

    print(
        f"\n{args.rollout_steps}-step rollout "
        "observed-space metrics:"
    )

    print(
        f"  Position AEE:      "
        f"{rollout_aggregate['position_aee']:.4f}"
    )

    print(
        "  Position failures: "
        f"{rollout_aggregate['position_failures']} / "
        f"{rollout_aggregate['position_total']}"
    )

    # ---------------------------------------------------------
    # Save in the same main schema as eval_latent_flow.py
    # ---------------------------------------------------------

    results = {
        "metadata": {
            "model_type": "sequence_encoder_decoder",
            "model_dir": args.model_dir,
            "ckpt_name": args.ckpt_name,
            "data_dir": args.data_dir,
            "context": args.context,
            "rollout_steps": args.rollout_steps,
            "eval_stride": args.eval_stride,
            "num_objects": args.num_objects,
            "threshold": args.threshold,
            "min_mass": args.min_mass,
            "fallback_thresholds": (
                args.fallback_thresholds
            ),
            "topk_fallback": (
                not args.disable_topk_fallback
            ),
            "topk_ratio": args.topk_ratio,
            "seed": args.seed,
            "test_trajectories": len(test_dirs),
        },
        "one_step": {
            "from_observed": one_step_metrics,
        },
        "rollout": {
            "from_observed": rollout_metrics,
        },
    }

    split_name = os.path.basename(
        os.path.normpath(args.data_dir)
    )

    save_dir = os.path.join(
        args.model_dir,
        "physics",
    )

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    save_path = os.path.join(
        save_dir,
        f"evaluation_results_{split_name}.json",
    )

    with open(
        save_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print(
        f"\nSaved results to: {save_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the original SequenceEncoderDecoder "
            "LSTM/GRU models with the same fixed-window "
            "observed-space protocol used for Latent Flow."
        )
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="best.ckpt",
    )

    parser.add_argument(
        "--context",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--eval_stride",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--num_objects",
        type=int,
        choices=[1, 2],
        required=True,
    )

    parser.add_argument(
        "--frame_batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--min_mass",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--fallback_thresholds",
        type=float,
        nargs="+",
        default=[
            0.4,
            0.3,
            0.2,
            0.15,
            0.1,
            0.05,
        ],
    )

    parser.add_argument(
        "--disable_topk_fallback",
        action="store_true",
    )

    parser.add_argument(
        "--topk_ratio",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--debug_threshold",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    main(
        parser.parse_args()
    )