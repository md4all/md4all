# Adapted from https://github.com/TRI-ML/packnet-sfm/blob/de53b310533ff6b01eaa23a8ba5ac01bac5587b1/packnet_sfm/losses/multiview_photometric_loss.py

import torch
import torch.nn as nn

from utils.image import match_scales, interpolate_scales
from utils.camera import Warping


def SSIM(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMilarity (SSIM) distance between two images.

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim


class MultiViewPhotometricLoss(nn.Module):
    """
    Self-Supervised multiview photometric loss.
    It takes two images, a depth map and a pose transformation to produce a
    reconstruction of one image from the perspective of the other, and calculates
    the difference between them
    """
    def __init__(self, cfg):
        super().__init__()

        self.ssim_loss_weight = cfg.LOSS.PHOTOMETRIC.SSIM.WEIGHT
        self.C1 = cfg.LOSS.PHOTOMETRIC.SSIM.C1
        self.C2 = cfg.LOSS.PHOTOMETRIC.SSIM.C2
        self.photometric_reduce_op = cfg.LOSS.PHOTOMETRIC.REDUCE_OP
        self.padding_mode = cfg.LOSS.PHOTOMETRIC.PADDING_MODE
        self.clip_loss = cfg.LOSS.PHOTOMETRIC.CLIP
        self.automask_loss = cfg.LOSS.PHOTOMETRIC.AUTOMASK
        self.upsample_depth = cfg.LOSS.DEPTH_UPSAMPLE
        self.min_depth = cfg.MODEL.DEPTH.MIN_DEPTH
        self.max_depth = cfg.MODEL.DEPTH.MAX_DEPTH

        self.res_height = cfg.DATASET.AUGMENTATION.RESIZE.HEIGHT
        self.res_width = cfg.DATASET.AUGMENTATION.RESIZE.WIDTH
        self.scales = cfg.DATASET.SCALES
        self.temp_context = cfg.DATASET.TEMP_CONTEXT

        self.warping = Warping(self.scales)

        # Asserts
        if self.automask_loss:
            assert self.photometric_reduce_op == 'min', \
                'For automasking only the min photometric_reduce_op is supported.'

    def SSIM(self, x, y, kernel_size=3):
        """
        Calculates the SSIM (Structural SIMilarity) loss

        Parameters
        ----------
        x,y : torch.Tensor [B,3,H,W]
            Input images
        kernel_size : int
            Convolutional parameter

        Returns
        -------
        ssim : torch.Tensor [1]
            SSIM loss
        """
        ssim_value = SSIM(x, y, C1=self.C1, C2=self.C2, kernel_size=kernel_size)
        return torch.clamp((1. - ssim_value) / 2., 0., 1.)

    def calc_photometric_loss(self, t_est, images):
        """
        Calculates the photometric loss (L1 + SSIM)
        Parameters
        ----------
        t_est : list of torch.Tensor [B,3,H,W]
            List of warped reference images in multiple scales
        images : list of torch.Tensor [B,3,H,W]
            List of original images in multiple scales

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Photometric loss
        """
        # L1 loss
        l1_loss = [torch.abs(t_est[i] - images[i])
                   for i in self.scales]
        # SSIM loss
        if self.ssim_loss_weight > 0.0:
            ssim_loss = [self.SSIM(t_est[i], images[i], kernel_size=3)
                         for i in self.scales]
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss = [self.ssim_loss_weight * ssim_loss[i].mean(1, True) +
                                (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True)
                                for i in self.scales]
        else:
            photometric_loss = l1_loss
        # Clip loss
        if self.clip_loss > 0.0:
            for i in self.scales:
                mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
                photometric_loss[i] = torch.clamp(
                    photometric_loss[i], max=float(mean + self.clip_loss * std))
        # Return total photometric loss
        return photometric_loss

    def reduce_photometric_loss(self, photometric_losses):
        """
        Combine the photometric loss from all context images

        Parameters
        ----------
        photometric_losses : list of torch.Tensor [B,3,H,W]
            Pixel-wise photometric losses from the entire context

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Reduced photometric loss
        """
        # Reduce function
        def reduce_function(losses):
            if self.photometric_reduce_op == 'mean':
                return sum([l.mean() for l in losses]) / len(losses)
            elif self.photometric_reduce_op == 'min':
                return torch.cat(losses, 1).min(1, True)[0].mean()
            else:
                raise NotImplementedError(
                    'Unknown photometric_reduce_op: {}'.format(self.photometric_reduce_op))
        # Reduce photometric loss
        photometric_loss = sum([reduce_function(photometric_losses[i])
                                for i in self.scales]) / len(self.scales)
        # Return reduced photometric loss
        return photometric_loss

    @staticmethod
    def compute_auto_masks(photometric_losses_scale):
        photometric_losses_scale = torch.cat(photometric_losses_scale, 1)
        idxs = torch.argmin(photometric_losses_scale, dim=1, keepdim=True)
        photometric_loss_mask_scale = (idxs % 2 == 0).float().detach()
        return photometric_loss_mask_scale

    def forward(self, inputs, outputs):
        """
        Calculates training photometric loss.
        """
        disps = [outputs[("disp", 0, scale)] for scale in self.scales]
        if self.upsample_depth:
            disps = interpolate_scales(disps, mode="nearest")
        # Loop over all reference images
        photometric_losses = [[] for _ in self.scales]
        images = match_scales(inputs[("color", 0)], disps, self.scales)
        K = inputs['K']
        for frame_id, (ref_image, pose) in zip(self.temp_context[1:], [(inputs[("color", frame_id)], outputs[("pose", frame_id)]) for frame_id in self.temp_context[1:]]):
            # Calculate warped images
            ref_warped, _, _ = self.warping.warp_ref_image(disps, ref_image, K, K, pose, padding_mode=self.padding_mode)
            outputs[("color", frame_id, 0)] = ref_warped[0]
            # Calculate and store image loss
            photometric_loss = self.calc_photometric_loss(ref_warped, images)
            for i in self.scales:
                photometric_losses[i].append(photometric_loss[i])
            # If using automask
            if self.automask_loss:
                # Calculate and store unwarped image loss
                ref_images = match_scales(ref_image, disps, self.scales)
                unwarped_image_loss = self.calc_photometric_loss(ref_images, images)
                for i in self.scales:
                    photometric_losses[i].append(unwarped_image_loss[i])

        # Calculate reduced photometric loss
        loss = self.reduce_photometric_loss(photometric_losses)

        # Return losses
        return loss
