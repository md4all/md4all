import torch.nn as nn

from losses.MultiViewPhotometricLoss import MultiViewPhotometricLoss
from losses.AverageMeter import AverageMeter
from losses.SmoothnessLoss import SmoothnessLoss
from losses.SupervisedLoss import SupervisedLoss
from losses.VelocityLoss import VelocityLoss


class TotalLoss(nn.Module):
    def __init__(self, cfg, is_train):
        super(TotalLoss, self).__init__()

        assert (cfg.LOSS.PHOTOMETRIC.WEIGHT > 0.0 and cfg.LOSS.SMOOTHNESS_WEIGHT >= 0.0 and cfg.LOSS.VELOCITY_WEIGHT >= 0.0) or cfg.LOSS.SUPERVISED.WEIGHT > 0.0
        self.photometric_weight = cfg.LOSS.PHOTOMETRIC.WEIGHT
        self.smoothness_weight = cfg.LOSS.SMOOTHNESS_WEIGHT
        self.velocity_weight = cfg.LOSS.VELOCITY_WEIGHT
        self.supervised_weight = cfg.LOSS.SUPERVISED.WEIGHT
        self.loss_history = {"total_loss": AverageMeter("Total Loss Baseline", ":6.3f")}

        if self.photometric_weight > 0.0:
            self.photometric_loss = MultiViewPhotometricLoss(cfg)
            self.loss_history["photometric_loss"] = AverageMeter("Photometric Loss Baseline", ":6.3f")

        if self.smoothness_weight > 0.0:
            self.smoothness_loss = SmoothnessLoss(cfg)
            self.loss_history["smoothness_loss"] = AverageMeter("Smoothness Loss Baseline", ":6.3f")

        if self.velocity_weight > 0.0:
            self.velocity_loss = VelocityLoss(cfg)
            self.loss_history["velocity_loss"] = AverageMeter("Velocity Loss Baseline", ":6.3f")

        if self.supervised_weight > 0.0:
            self.supervised_loss = SupervisedLoss(cfg, is_train)
            self.loss_history["supervised_loss"] = AverageMeter("Supervised Loss", ":6.3f")

    def running_avg_loss(self, loss_name):
        return self.loss_history[loss_name].avg

    @property
    def running_avg_total_loss(self):
        return self.loss_history["total_loss"].avg

    @property
    def running_avg_photometric_loss(self):
        return self.loss_history["photometric_loss"].avg

    @property
    def running_avg_smoothness_loss(self):
        return self.loss_history["smoothness_loss"].avg

    @property
    def running_avg_velocity_loss(self):
        return self.loss_history["velocity_loss"].avg

    @property
    def running_avg_supervised_loss(self):
        return self.loss_history["supervised_loss"].avg

    @property
    def activated_losses(self):
        return self.loss_history.keys()

    def reset_losses(self):
        for _, avgmeter in self.loss_history.items():
            avgmeter.reset()

    def forward(self, inputs, outputs):
        total_loss = 0

        if self.photometric_weight > 0.0:
            photometric_loss = self.photometric_weight * self.photometric_loss(inputs, outputs)
            self.loss_history["photometric_loss"].update(photometric_loss.item())
            total_loss += photometric_loss

        if self.smoothness_weight > 0.0:
            smoothness_loss = self.smoothness_weight * self.smoothness_loss(inputs, outputs)
            self.loss_history["smoothness_loss"].update(smoothness_loss.item())
            total_loss += smoothness_loss

        if self.velocity_weight > 0.0:
            velocity_loss = self.velocity_weight * self.velocity_loss(inputs, outputs)
            self.loss_history["velocity_loss"].update(velocity_loss.item())
            total_loss += velocity_loss

        if self.supervised_weight > 0.0:
            supervised_loss = self.supervised_weight * self.supervised_loss(inputs, outputs)
            self.loss_history["supervised_loss"].update(supervised_loss.item())
            total_loss += supervised_loss

        self.loss_history["total_loss"].update(total_loss.item())
        return total_loss
