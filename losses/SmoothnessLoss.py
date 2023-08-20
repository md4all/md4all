import torch
import torch.nn as nn

from utils.image import interpolate_scales, match_scales


def inv_depths_normalize(inv_depths):
    """
    Inverse depth normalization

    Parameters
    ----------
    inv_depths : list of torch.Tensor [B,1,H,W]
        Inverse depth maps

    Returns
    -------
    norm_inv_depths : list of torch.Tensor [B,1,H,W]
        Normalized inverse depth maps
    """
    mean_inv_depths = [inv_depth.mean(2, True).mean(3, True) for inv_depth in inv_depths]
    return [inv_depth / mean_inv_depth.clamp(min=1e-6)
            for inv_depth, mean_inv_depth in zip(inv_depths, mean_inv_depths)]


def gradient_x(image):
    """
    Calculates the gradient of an image in the x dimension
    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Input image

    Returns
    -------
    gradient_x : torch.Tensor [B,3,H,W-1]
        Gradient of image with respect to x
    """
    return image[:, :, :, :-1] - image[:, :, :, 1:]


def gradient_y(image):
    """
    Calculates the gradient of an image in the y dimension
    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Input image

    Returns
    -------
    gradient_y : torch.Tensor [B,3,H-1,W]
        Gradient of image with respect to y
    """
    return image[:, :, :-1, :] - image[:, :, 1:, :]


class SmoothnessLoss(nn.Module):
    def __init__(self, cfg):
        super(SmoothnessLoss, self).__init__()

        self.upsample_depth = cfg.LOSS.DEPTH_UPSAMPLE
        self.scales = cfg.DATASET.SCALES

    @staticmethod
    def calc_smoothness(inv_depths, images, scales):
        """
        Calculate smoothness values for inverse depths

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Inverse depth maps
        images : list of torch.Tensor [B,3,H,W]
            Inverse depth maps
        scales : list
            Scales considered

        Returns
        -------
        smoothness_x : list of torch.Tensor [B,1,H,W]
            Smoothness values in direction x
        smoothness_y : list of torch.Tensor [B,1,H,W]
            Smoothness values in direction y
        """
        inv_depths_norm = inv_depths_normalize(inv_depths)
        inv_depth_gradients_x = [gradient_x(d) for d in inv_depths_norm]
        inv_depth_gradients_y = [gradient_y(d) for d in inv_depths_norm]

        image_gradients_x = [gradient_x(image) for image in images]
        image_gradients_y = [gradient_y(image) for image in images]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

        # Note: Fix gradient addition
        smoothness_x = [inv_depth_gradients_x[i] * weights_x[i] for i in scales]
        smoothness_y = [inv_depth_gradients_y[i] * weights_y[i] for i in scales]
        return smoothness_x, smoothness_y

    def calc_smoothness_loss(self, inv_depths, images):
        """
        Calculates the smoothness loss for inverse depth maps.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted inverse depth maps for all scales
        images : list of torch.Tensor [B,3,H,W]
            Original images for all scales

        Returns
        -------
        smoothness_loss : torch.Tensor [1]
            Smoothness loss
        """
        # Calculate smoothness gradients
        smoothness_x, smoothness_y = self.calc_smoothness(inv_depths, images, self.scales)
        # Calculate smoothness loss
        smoothness_loss = sum([(smoothness_x[i].abs().mean() +
                                smoothness_y[i].abs().mean()) / 2 ** i
                               for i in self.scales]) / len(self.scales)
        # Return smoothness loss
        return smoothness_loss

    def forward(self, inputs, outputs):
        disps = [outputs[("disp", 0, scale)] for scale in self.scales]
        if self.upsample_depth:
            disps = interpolate_scales(disps, mode="nearest")
        images = match_scales(inputs[("color", 0)], disps, self.scales)

        smoothess_loss = self.calc_smoothness_loss(disps, images)

        return smoothess_loss
