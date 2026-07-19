from glob import glob
import argparse
import json
import os

import torch
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms


from model import SequenceEncoderDecoder, SpatioTemporalTransformer
from dataset import FramePredictionDataset
from learner import WorldModelLearner


def global_to_local_(args: argparse.Namespace) -> None:
    """
    Translate from global batch size and num workers to local, i.e., to the batch size and num workers to use per GPU.
    This function modifies its argument in-place.
    """
    # no translation needed if training on CPU
    if args.accelerator == 'cpu' or not torch.cuda.is_available():
        args.devices = args.devices if args.devices != -1 else 1
        return

    # compute world size
    devices = args.devices if args.devices != -1 else torch.cuda.device_count()
    num_nodes = args.num_nodes
    world_size = num_nodes * devices

    # divide global batch size by world size
    global_batch_size = args.batch_size
    if global_batch_size % world_size != 0:
        raise ValueError(
            'Global batch size needs to be divisible by world size but got '
            f'global_batch_size={global_batch_size} and world_size={world_size}.'
        )
    args.batch_size = global_batch_size // world_size
    args.global_batch_size = global_batch_size

    # divide global num workers by number of devices
    global_num_workers = args.num_workers if args.num_workers else 0
    if global_num_workers % devices != 0:
        raise ValueError(
            'Global num workers size needs to be divisible by number of devices but got '
            f'global_num_workers={global_num_workers} and devices={devices}.'
        )
    args.num_workers = global_num_workers // devices
    args.global_num_workers = global_num_workers


def train(args: argparse.Namespace):
    # Data preparation
    sequence_dirs = [os.path.join(args.data_dir, f'traj-{i}') for i in range(args.n_trajectories)]
    num_val_trajectories = int(len(sequence_dirs) * args.val_pct)
    train_dirs = sequence_dirs[:-num_val_trajectories]
    val_dirs = sequence_dirs[-num_val_trajectories:]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
    ])
    train_dataset = FramePredictionDataset(train_dirs, context=args.context, rollout=args.rollout, transform=transform)
    val_dataset = FramePredictionDataset(val_dirs, context=args.context, rollout=args.rollout, transform=transform)
    print(f'Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.rnn_type == 'stt':
        net = SpatioTemporalTransformer(
            input_channels=1,
            n_layers=args.transformer_layers,
            n_heads=args.transformer_heads,
            d_model=args.transformer_dim,
            patch_size=args.patch_size,
            img_size=args.image_size,
            context=args.context,
        )
    else:
        net = SequenceEncoderDecoder(
            rnn_type=args.rnn_type,
            input_channels=1,
            hidden_channels=args.hidden_channels,
            kernel_size=args.kernel_size,
            encoder_channels=args.encoder_channels,
            norm_type=args.norm_type,
            norm_groups=args.norm_groups,
        )

    def schedule_fn(epoch: int) -> float:
        decay_start = int(args.max_epochs * 0.4)
        if epoch < decay_start:
            return 1.0
        else:
            return max(0.5 * (1 + np.cos(np.pi * (epoch - decay_start) / (args.max_epochs - decay_start))), 0.2)


    learner = WorldModelLearner(
        net,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        teacher_forcing_schedule=schedule_fn,
    )

    previous_versions = [int(exp.split('_')[-1]) for exp in glob(os.path.join('models', f'{args.rnn_type}_*'))]
    version = 0 if len(previous_versions) == 0 else max(previous_versions) + 1
    exp_dir = os.path.join('models', f'{args.rnn_type}_{version}')

    logger = TensorBoardLogger(save_dir=exp_dir, name='tensorboard', version='')

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=os.path.join(exp_dir, 'checkpoints'),
            save_top_k=-1,
            save_last=True,
            every_n_epochs=args.save_epochs,
            save_weights_only=True,
        ),
    ]

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        max_epochs=args.max_epochs,
        max_steps=args.max_epochs * len(train_loader),
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        detect_anomaly=args.detect_anomaly,
    )

    additional_args = {
        'train_dirs': train_dirs,
        'val_dirs': val_dirs,
        'input_channels': 1,
    }

    if trainer.is_global_zero:
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, 'args.json'), 'w') as f:
            json.dump({**vars(args), **additional_args}, f, indent=4)

    trainer.fit(learner, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model for frame prediction.')
    parser.add_argument('--data_dir', type=str, default='physics-data', help='Directory containing the dataset.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers for data loading.')
    parser.add_argument('--context', type=int, default=5, help='Number of input frames before target.')
    parser.add_argument('--rollout', type=int, default=1, help='Number of frames to roll out during training.')
    parser.add_argument('--val_pct', type=float, default=0.1, help='Percentage of data to use for validation.')
    parser.add_argument('--n_trajectories', type=int, default=1000, help='Number of trajectories to use.')

    parser.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'gru', 'stt'], help='Type of RNN to use.')
    parser.add_argument('--hidden_channels', type=int, nargs='+', default=[32], help='Hidden channels of the RNN.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for the convolutional layers.')
    parser.add_argument('--encoder_channels', type=int, default=16, help='Number of channels in the encoder.')
    parser.add_argument('--norm_type', type=str, choices=['group', 'layer'], help='Normalization type.')
    parser.add_argument('--norm_groups', type=int, default=4, help='Number of groups for group normalization.')

    parser.add_argument('--transformer_layers', type=int, default=2, help='Number of transformer layers.')
    parser.add_argument('--transformer_heads', type=int, default=4, help='Random seed for reproducibility.')
    parser.add_argument('--transformer_dim', type=int, default=128, help='Dimension of transformer layers.')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for transformer.')
    parser.add_argument('--image_size', type=int, default=128, help='Image size for transformer.')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for AdamW.')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps for lr scheduler.')

    parser.add_argument(
        '--accelerator',
        type=str,
        default='auto',
        choices=('cpu', 'gpu', 'auto'),
        help='the accelerator used for training',
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='auto',
        choices=('ddp', 'fsdp', 'auto'),
        help='parallelization strategy for the trainer',
    )
    parser.add_argument('--devices', type=int, default=-1, help='the number of devices to train on')
    parser.add_argument('--num_nodes', type=int, default=1, help='the number of gpu nodes to train on')
    parser.add_argument(
        '--precision',
        type=str,
        default='16-mixed',
        choices=('16-mixed', 'bf16-mixed', '32-true', '64-true'),
        help='the floating point precision to use',
    )
    parser.add_argument('--max_epochs', type=int, default=100, help='the maximum number of epochs')
    parser.add_argument('--save_epochs', type=int, default=10, help='the number of epochs between checkpoints')
    parser.add_argument('--gradient_clip_val', type=float, default=None, help='value for clipping global gradient norm')
    parser.add_argument('--detect_anomaly', action='store_true', help='enables torch.autograd anomaly detection')

    args = parser.parse_args()
    global_to_local_(args)
    train(args)
