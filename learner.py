from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl


class WorldModelLearner(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        teacher_forcing_schedule: Callable[[int], float] = lambda _: 1.0,
    ) -> None:
        super().__init__()
        self.net = net

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.teacher_forcing_schedule = teacher_forcing_schedule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def compute_and_log_loss(self, batch: tuple, suffix: str) -> torch.Tensor:
        """
        Handle forward pass, loss computation, and logging. Used for both training and validation steps.
        """
        input_seqs, target_frames = batch
        assert input_seqs.ndim == 5, f'Expected input shape (B, T, C, H, W), got {input_seqs.shape}'
        assert target_frames.ndim == 5, f'Expected target shape (B, R, C, H, W), got {target_frames.shape}'

        rollout_length = target_frames.shape[1]
        teacher_forcing_ratio = self.teacher_forcing_schedule(self.current_epoch)

        if rollout_length > 1:
            self.log(f'TeacherForcing/{suffix}', teacher_forcing_ratio, rank_zero_only=True)

        predicted_frames = []
        for i in range(rollout_length):
            predicted_frame = self.forward(input_seqs).unsqueeze(1)  # (B, 1, C, H, W)
            predicted_frames.append(predicted_frame)

            if i < rollout_length - 1:
                if torch.rand(1).item() < teacher_forcing_ratio:
                    # Use the target frame for the next step
                    next_frame = target_frames[:, i:i + 1]
                else:
                    # Use the predicted frame for the next step
                    next_frame = predicted_frame.detach()
                input_seqs = torch.cat((input_seqs[:, 1:], next_frame), dim=1)

        predicted_frames = torch.cat(predicted_frames, dim=1)  # (B, R, C, H, W)
        assert predicted_frames.shape == target_frames.shape, \
            f'Expected predicted shape (B, R, C, H, W), got {predicted_frames.shape}'

        loss = F.mse_loss(predicted_frames, target_frames)
        self.log(f'Loss/{suffix}', loss, sync_dist=True)

        return loss

    def training_step(self, batch: tuple, _) -> torch.Tensor:
        assert self.training
        return self.compute_and_log_loss(batch, suffix='train')

    def validation_step(self, batch: tuple, _) -> torch.Tensor:
        return self.compute_and_log_loss(batch, suffix='valid')

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameter_groups(), lr=self.lr, weight_decay=self.weight_decay)

        total_steps = self.num_training_steps
        warmup_steps = self.warmup_steps
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-15, total_iters=warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def parameter_groups(self) -> list[dict]:
        """Group parameters into weight decay and no weight decay."""

        def exclude(n: str, p: nn.Parameter) -> bool:
            return p.ndim < 2 or 'norm' in n or 'bias' in n

        def include(n: str, p: nn.Parameter) -> bool:
            return not exclude(n, p)

        named_parameters = list(self.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        return [{'params': gain_or_bias_params, 'weight_decay': 0}, {'params': rest_params}]

    @property
    def num_training_steps(self) -> int:
        # return max steps directly if specified
        if hasattr(self.trainer, 'max_steps') and self.trainer.max_steps != -1:
            return self.trainer.max_steps

        # if max steps is not provided, we require max epochs to be given
        assert self.trainer.max_epochs, \
            'self.trainer.max_steps=-1 and self.trainer.max_epochs=None. Cannot compute num_training_steps.'

        # compute number of total training batches
        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())  # type: ignore
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        # account for multi gpu setups and gradient accumulation
        effective_accum = self.trainer.accumulate_grad_batches * self.trainer.num_devices
        return (batches // effective_accum) * self.trainer.max_epochs
