# Adapted from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/losses/supervised_loss.py

import torch
import torch.nn as nn

from config.config import get_cfg
from models.depth_net import DepthNet


class BerHuLoss(nn.Module):
    """Class implementing the BerHu loss."""
    def __init__(self, threshold=0.2):
        """
        Initializes the BerHuLoss class.
        Parameters
        ----------
        threshold : float
            Mask parameter
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, gt):
        """
        Calculates the BerHu loss.
        Parameters
        ----------
        pred : torch.Tensor [B,1,H,W]
            Predicted inverse depth map
        gt : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth map
        Returns
        -------
        loss : torch.Tensor [1]
            BerHu loss
        """
        huber_c = torch.max(pred - gt)
        huber_c = self.threshold * huber_c
        diff = (pred - gt).abs()

        # Remove
        # mask = (gt > 0).detach()
        # diff = gt - pred
        # diff = diff[mask]
        # diff = diff.abs()

        huber_mask = (diff > huber_c).detach()
        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2
        return torch.cat((diff, diff2)).mean()


class SilogLoss(nn.Module):
    def __init__(self, ratio=10, ratio2=0.85):
        super().__init__()
        self.ratio = ratio
        self.ratio2 = ratio2

    def forward(self, pred, gt):
        log_diff = torch.log(pred * self.ratio) - \
                   torch.log(gt * self.ratio)
        silog1 = torch.mean(log_diff ** 2)
        silog2 = self.ratio2 * (log_diff.mean() ** 2)
        silog_loss = torch.sqrt(silog1 - silog2) * self.ratio
        return silog_loss


def get_loss_func(supervised_method):
    """Determines the supervised loss to be used, given the supervised method."""
    if supervised_method.endswith('l1'):
        return nn.L1Loss()
    elif supervised_method.endswith('mse'):
        return nn.MSELoss()
    elif supervised_method.endswith('berhu'):
        return BerHuLoss()
    elif supervised_method.endswith('silog'):
        return SilogLoss()
    elif supervised_method.endswith('abs_rel'):
        return lambda x, y: torch.mean(torch.abs(x - y) / x)
    else:
        raise ValueError('Unknown supervised loss {}'.format(supervised_method))


class SupervisedLoss(nn.Module):
    """
    Supervised loss for inverse depth maps.
    Parameters
    """
    def __init__(self, cfg, is_train):
        super().__init__()
        self.supervised_method = cfg.LOSS.SUPERVISED.METHOD
        self.loss_func = get_loss_func(self.supervised_method)
        self.scales = cfg.DATASET.SCALES
        self.teacher_net = DepthNet(cfg)
        if is_train:
            self.teacher_net.load_state_dict(state_dict={key.replace("depth_model.", ""): weight for key, weight in torch.load(cfg.LOAD.DAYTIME_TRANSLATION_TEACHER_PATH)['state_dict'].items() if 'pose_model' not in key})
        for param in self.teacher_net.parameters():
            param.requires_grad = False

    def calculate_loss(self, inv_depths, gt_inv_depths):
        """
        Calculate the supervised loss.
        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            List of predicted inverse depth maps
        gt_inv_depths : list of torch.Tensor [B,1,H,W]
            List of ground-truth inverse depth maps
        Returns
        -------
        loss : torch.Tensor [1]
            Average supervised loss for all scales
        """
        # If using a sparse loss, mask invalid pixels for all scales
        if self.supervised_method.startswith('sparse'):
            for i in self.scales:
                mask = (gt_inv_depths[i] > 0.).detach()
                inv_depths[i] = inv_depths[i][mask]
                gt_inv_depths[i] = gt_inv_depths[i][mask]
        # Return per-scale average loss
        return sum([self.loss_func(inv_depths[i], gt_inv_depths[i]) for i in self.scales]) / len(self.scales)

    def forward(self, inputs, outputs):
        """
        Calculates training supervised loss.
        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        gt_inv_depth : torch.Tensor [B,1,H,W]
            Ground-truth depth map for the original image
        """
        inv_depths = [outputs[("disp", 0, scale)] for scale in self.scales]
        with torch.no_grad():
            gt_inv_depth = self.teacher_net(inputs[("color", 0)], inputs['weather'])
            gt_inv_depths = [gt_inv_depth[("disp", 0, scale)] for scale in self.scales]
        # Calculate supervised loss
        loss = self.calculate_loss(inv_depths, gt_inv_depths)
        # Return losses and metrics
        return loss
