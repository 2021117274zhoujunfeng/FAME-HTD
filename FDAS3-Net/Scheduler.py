import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.warm_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.finished:
            if self.after_scheduler:
                if not hasattr(self.after_scheduler, "get_lr"):
                    return [group["lr"] for group in self.optimizer.param_groups]
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        progress = min(1.0, self.last_epoch / self.warm_epoch)
        return [
            base_lr * ((self.multiplier - 1) * progress + 1)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None, metrics=None):
        if self.finished:
            if self.after_scheduler:
                if isinstance(
                    self.after_scheduler, optim.lr_scheduler.ReduceLROnPlateau
                ):
                    if metrics is None:
                        raise ValueError("ReduceLROnPlateau requires metrics")
                    self.after_scheduler.step(metrics, epoch)
                else:
                    self.after_scheduler.step(epoch)
        else:
            super().step(epoch)
            if self.last_epoch >= self.warm_epoch:
                self.finished = True
                if self.after_scheduler:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
