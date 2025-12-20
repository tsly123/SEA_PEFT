import math
import warnings
import torch

from typing import List, Optional
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                1 +
                math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


class CosineAnnealingWarmRestarts(_LRScheduler):
    """Cosine annealing with warm restarts for adapter parameter groups.

    This scheduler restarts the cosine annealing at regular intervals (T_0),
    providing periodic LR resets that can help escape local minima during
    adapter training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        param_group_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_0 (int): Number of iterations for the first restart.
            T_mult (int): A factor increases T_i after a restart. Default: 1.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
            param_group_indices (List[int], optional): Indices of param groups to schedule.
                If None, applies to all groups.
        """
        self.T_0 = max(1, T_0)
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = self.T_0
        self.param_group_indices = param_group_indices
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
        # _LRScheduler.__init__ invokes `step()` once, which increments T_cur.
        # Reset back to zero so that the first real training step counts as iteration 1.
        self.T_cur = 0

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        lrs = []
        for group_idx, base_lr in enumerate(self.base_lrs):
            # Only apply to specified param groups, or all if not specified
            if self.param_group_indices is None or group_idx in self.param_group_indices:
                lr = self.eta_min + (base_lr - self.eta_min) * (
                    1 + math.cos(math.pi * self.T_cur / self.T_i)
                ) / 2
                lrs.append(lr)
            else:
                # Keep current LR for groups not managed by this scheduler
                lrs.append(self.optimizer.param_groups[group_idx]["lr"])
        return lrs

    def step(self, epoch: Optional[int] = None) -> None:
        """Step with restart detection."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = 0
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = epoch
        super(CosineAnnealingWarmRestarts, self).step(epoch)

    def is_restart_step(self) -> bool:
        """Check if current step is a restart point."""
        return self.T_cur == 0
