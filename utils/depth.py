# Adapted from https://github.com/TRI-ML/packnet-sfm/blob/de53b310533ff6b01eaa23a8ba5ac01bac5587b1/packnet_sfm/utils/depth.py

import numpy as np
import torch
from matplotlib.cm import get_cmap
from torchvision.utils import save_image

from utils.image import load_image, flip_lr, interpolate_image
from utils.types import is_seq, is_tensor


def load_depth(file):
    """
    Load a depth map from file
    Parameters
    ----------
    file : str
        Depth map filename (.npz or .png)

    Returns
    -------
    depth : np.array [H,W]
        Depth map (invalid pixels are 0)
    """
    if file.endswith('npz'):
        return np.load(file)['depth']
    elif file.endswith('png'):
        depth_png = np.array(load_image(file), dtype=int)
        assert (np.max(depth_png) > 255), 'Wrong .png depth file'
        return depth_png.astype(np.float) / 256.
    else:
        raise NotImplementedError('Depth extension not supported.')


def write_depth(filename, depth, intrinsics=None):
    """
    Write a depth map to file, and optionally its corresponding intrinsics.

    Parameters
    ----------
    filename : str
        File where depth map will be saved (.npz or .png)
    depth : np.array [H,W]
        Depth map
    intrinsics : np.array [3,3]
        Optional camera intrinsics matrix
    """
    # If depth is a tensor
    if is_tensor(depth):
        depth = depth.detach().squeeze().cpu()
    # If intrinsics is a tensor
    if is_tensor(intrinsics):
        intrinsics = intrinsics.detach().cpu()
    # If we are saving as a .npz
    if filename.endswith('.npz'):
        np.savez_compressed(filename, depth=depth, intrinsics=intrinsics)
    # If we are saving as a .png
    elif filename.endswith('.png'):
        save_image(depth, filename)
    # Something is wrong
    else:
        raise NotImplementedError('Depth filename not valid.')


def viz_inv_depth(inv_depth, normalizer=None, percentile=95,
                  colormap='plasma', filter_zeros=False):
    """
    Converts an inverse depth map to a colormap for visualization.

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map to be converted
    normalizer : float
        Value for inverse depth map normalization
    percentile : float
        Percentile value for automatic normalization
    colormap : str
        Colormap to be used
    filter_zeros : bool
        If True, do not consider zero values during normalization

    Returns
    -------
    colormap : np.array [H,W,3]
        Colormap generated from the inverse depth map
    """
    # If a tensor is provided, convert to numpy
    if is_tensor(inv_depth):
        # Squeeze if depth channel exists
        if len(inv_depth.shape) == 3:
            inv_depth = inv_depth.squeeze(0)
        inv_depth = inv_depth.detach().cpu().numpy()
    cm = get_cmap(colormap)
    if normalizer is None:
        normalizer = np.percentile(
            inv_depth[inv_depth > 0] if filter_zeros else inv_depth, percentile)
    inv_depth /= (normalizer + 1e-6)
    return cm(np.clip(inv_depth, 0., 1.0))[:, :, :3]


def viz_disps(disps, shape):
    disps_det = disps.clone().detach().cpu()
    disps_vis = torch.zeros(shape)
    for b in range(len(disps)):
        disps_vis[b, :, :, :] = torch.from_numpy(viz_inv_depth(disps_det[b, :, :, :]).transpose(2, 0, 1))
    return disps_vis


def inv2depth(inv_depth):
    """
    Invert an inverse depth map to produce a depth map

    Parameters
    ----------
    inv_depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
        Inverse depth map

    Returns
    -------
    depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
        Depth map
    """
    if is_seq(inv_depth):
        return [inv2depth(item) for item in inv_depth]
    else:
        return 1. / inv_depth.clamp(min=1e-6)


def depth2inv(depth):
    """
    Invert a depth map to produce an inverse depth map

    Parameters
    ----------
    depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
        Depth map

    Returns
    -------
    inv_depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
        Inverse depth map

    """
    if is_seq(depth):
        return [depth2inv(item) for item in depth]
    else:
        inv_depth = 1. / depth.clamp(min=1e-6)
        inv_depth[depth <= 0.] = 0.
        return inv_depth


def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    """
    Fuse inverse depth and flipped inverse depth maps

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_hat : torch.Tensor [B,1,H,W]
        Flipped inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    fused_inv_depth : torch.Tensor [B,1,H,W]
        Fused inverse depth map
    """
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))


def post_process_inv_depth(inv_depth, inv_depth_flipped, method='mean'):
    """
    Post-process an inverse and flipped inverse depth map

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_flipped : torch.Tensor [B,1,H,W]
        Inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    inv_depth_pp : torch.Tensor [B,1,H,W]
        Post-processed inverse depth map
    """
    B, C, H, W = inv_depth.shape
    inv_depth_hat = flip_lr(inv_depth_flipped)
    inv_depth_fused = fuse_inv_depth(inv_depth, inv_depth_hat, method=method)
    xs = torch.linspace(0., 1., W, device=inv_depth.device,
                        dtype=inv_depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)
    return mask_hat * inv_depth + mask * inv_depth_hat + \
           (1.0 - mask - mask_hat) * inv_depth_fused


def compute_depth_metrics(gt, pred, weather, metrics, mode, min_depth, max_depth, use_gt_scale=True):
    # Initialize variables
    batch_size, _, gt_height, gt_width = gt.shape
    # Interpolate predicted depth to ground-truth resolution
    pred = interpolate_image(pred, gt.shape, mode='bilinear', align_corners=True)
    # For each depth map
    for pred_i, gt_i, weather_i in zip(pred, gt, weather):
        gt_i, pred_i = torch.squeeze(gt_i), torch.squeeze(pred_i)
        # Keep valid pixels (min/max depth and crop)
        valid = (gt_i > min_depth) & (gt_i < max_depth)
        # Stop if there are no remaining valid pixels
        if valid.sum() == 0:
            continue
        # Keep only valid pixels
        gt_i, pred_i = gt_i[valid], pred_i[valid]
        # Ground-truth median scaling if needed
        if use_gt_scale:
            pred_i = pred_i * torch.median(gt_i) / torch.median(pred_i)
        # Clamp predicted depth values to min/max values
        pred_i = pred_i.clamp(min_depth, max_depth)

        # Calculate depth metrics
        thresh = torch.max((gt_i / pred_i), (pred_i / gt_i))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        diff_i = gt_i - pred_i
        abs_diff = torch.mean(torch.abs(diff_i))
        abs_rel = torch.mean(torch.abs(diff_i) / gt_i)
        sq_rel = torch.mean(diff_i ** 2 / gt_i)
        rmse = torch.sqrt(torch.mean(diff_i ** 2))
        rmse_log = torch.sqrt(torch.mean((torch.log(gt_i) - torch.log(pred_i)) ** 2))

        for condition in metrics.keys():
            if condition in weather_i or condition == 'all-conditions':
                metrics[condition][mode]['abs_rel'] += abs_rel.item()
                metrics[condition][mode]['sq_rel'] += sq_rel.item()
                metrics[condition][mode]['rmse'] += rmse.item()
                metrics[condition][mode]['rmse_log'] += rmse_log.item()
                metrics[condition][mode]['a1'] += a1.item()
                metrics[condition][mode]['a2'] += a2.item()
                metrics[condition][mode]['a3'] += a3.item()


def scale_depth(pred, gt, scale_fn):
    """
    Match depth maps to ground-truth resolution

    Parameters
    ----------
    pred : torch.Tensor
        Predicted depth maps [B,1,w,h]
    gt : torch.tensor
        Ground-truth depth maps [B,1,H,W]
    scale_fn : str
        How to scale output to GT resolution
            Resize: Nearest neighbors interpolation
            top-center: Pad the top of the image and left-right corners with zeros

    Returns
    -------
    pred : torch.tensor
        Uncropped predicted depth maps [B,1,H,W]
    """
    if scale_fn == 'resize':
        # Resize depth map to GT resolution
        return interpolate_image(pred, gt.shape, mode='bilinear', align_corners=True)
    else:
        # Create empty depth map with GT resolution
        pred_uncropped = torch.zeros(gt.shape, dtype=pred.dtype, device=pred.device)
        # Uncrop top vertically and center horizontally
        if scale_fn == 'top-center':
            top, left = gt.shape[2] - pred.shape[2], (gt.shape[3] - pred.shape[3]) // 2
            pred_uncropped[:, :, top:(top + pred.shape[2]), left:(left + pred.shape[3])] = pred
        else:
            raise NotImplementedError('Depth scale function {} not implemented.'.format(scale_fn))
        # Return uncropped depth map
        return pred_uncropped