# Adapted from https://github.com/TRI-ML/packnet-sfm/blob/de53b310533ff6b01eaa23a8ba5ac01bac5587b1/packnet_sfm/geometry/camera.py

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth import inv2depth
from .pose import Pose
from .image import image_grid, match_scales


class Camera(nn.Module):
    """
    Differentiable camera class implementing reconstruction and projection
    functions for a pinhole model.
    """
    def __init__(self, K, Tcw=None):
        """
        Initializes the Camera class

        Parameters
        ----------
        K : torch.Tensor [3,3]
            Camera intrinsics
        Tcw : Pose
            Camera -> World pose transformation
        """
        super().__init__()
        self.K = K
        self.Tcw = Pose.identity(len(K), K.device, K.dtype) if Tcw is None else Tcw

    def __len__(self):
        """Batch size of the camera intrinsics"""
        return len(self.K)

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.K = self.K.to(*args, **kwargs)
        self.Tcw = self.Tcw.to(*args, **kwargs)
        return self

    @property
    def fx(self):
        """Focal length in x"""
        return self.K[:, 0, 0]

    @property
    def fy(self):
        """Focal length in y"""
        return self.K[:, 1, 1]

    @property
    def cx(self):
        """Principal point in x"""
        return self.K[:, 0, 2]

    @property
    def cy(self):
        """Principal point in y"""
        return self.K[:, 1, 2]

    @property
    @lru_cache()
    def Twc(self):
        """World -> Camera pose transformation (inverse of Tcw)"""
        return self.Tcw.inverse()

    @property
    @lru_cache()
    def Kinv(self):
        """Inverse intrinsics (for lifting)"""
        Kinv = self.K.clone()
        Kinv[:, 0, 0] = 1. / self.fx
        Kinv[:, 1, 1] = 1. / self.fy
        Kinv[:, 0, 2] = -1. * self.cx / self.fx
        Kinv[:, 1, 2] = -1. * self.cy / self.fy
        return Kinv

    @staticmethod
    def scale_intrinsics(K, x_scale, y_scale):
        """Scale intrinsics given x_scale and y_scale factors"""
        K[..., 0, 0] *= x_scale
        K[..., 1, 1] *= y_scale
        K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
        K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
        return K

    def scaled(self, x_scale, y_scale=None):
        """
        Returns a scaled version of the camera (changing intrinsics)

        Parameters
        ----------
        x_scale : float
            Resize scale in x
        y_scale : float
            Resize scale in y. If None, use the same as x_scale

        Returns
        -------
        camera : Camera
            Scaled version of the current cmaera
        """
        # If single value is provided, use for both dimensions
        if y_scale is None:
            y_scale = x_scale
        # If no scaling is necessary, return same camera
        if x_scale == 1. and y_scale == 1.:
            return self
        # Scale intrinsics and return new camera with same Pose
        K = self.scale_intrinsics(self.K.clone(), x_scale, y_scale)
        return Camera(K, Tcw=self.Tcw)

    def reconstruct(self, depth, frame='w', coordinates_grid=None):
        """
        Reconstructs pixel-wise 3D points from a depth map.

        Parameters
        ----------
        depth : torch.Tensor [B,1,H,W]
            Depth map for the camera
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.tensor [B,3,H,W]
            Pixel-wise 3D points
        """
        B, C, H, W = depth.shape
        assert C == 1

        # Create flat index grid
        if coordinates_grid is None:
            grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False)  # [B,3,H,W]
            flat_grid = grid.view(B, 3, -1)  # [B,3,HW]
        else:
            flat_grid = coordinates_grid

        # Estimate the outward rays in the camera frame
        xnorm = (self.Kinv.matmul(flat_grid)).view(B, 3, H, W)
        # Scale rays to metric depth
        Xc = xnorm * depth

        # If in camera frame of reference
        if frame == 'c':
            return Xc
        # If in world frame of reference
        elif frame == 'w':
            return self.Twc @ Xc
        # If none of the above
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

    def project(self, X, frame='w'):
        """
        Projects 3D points onto the image plane

        Parameters
        ----------
        X : torch.Tensor [B,3,H,W]
            3D points to be projected
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.Tensor [B,H,W,2]
            2D projected points that are within the image boundaries
        """
        B, C, H, W = X.shape
        assert C == 3

        # Project 3D points onto the camera image plane
        if frame == 'c':
            Xc = self.K.bmm(X.view(B, C, -1))
        elif frame == 'w':
            Xc = self.K.bmm((self.Tcw @ X).view(B, C, -1))
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

        # Normalize points
        X = Xc[:, 0]
        Y = Xc[:, 1]
        Z = Xc[:, 2]#.clamp(min=1e-5)
        Xnorm = 2 * (X / Z) / (W - 1) - 1.
        Ynorm = 2 * (Y / Z) / (H - 1) - 1.

        # Clamp out-of-bounds pixels
        # Xmask = ((Xnorm > 1) + (Xnorm < -1)).detach()
        # Xnorm[Xmask] = 2.
        # Ymask = ((Ynorm > 1) + (Ynorm < -1)).detach()
        # Ynorm[Ymask] = 2.

        # Return pixel coordinates
        return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2), Z.view(B, H, W, 1)


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Warping(nn.Module):
    def __init__(self, scales):
        super().__init__()
        self.scales = scales

    @staticmethod
    def view_synthesis(ref_image, depth, ref_cam, cam, padding_mode):
        """
        Synthesize an image from another plus a depth map.

        Parameters
        ----------
        ref_image : torch.Tensor [B,3,H,W]
            Reference image to be warped
        depth : torch.Tensor [B,1,H,W]
            Depth map from the original image
        ref_cam : Camera
            Camera class for the reference image
        cam : Camera
            Camera class for the original image
        mode : str
            Interpolation mode
        padding_mode : str
            Padding mode for interpolation

        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image in the original frame of reference
        """
        assert depth.size(1) == 1
        B, C, H, W = depth.shape

        # Reconstruct world points from target_camera
        world_points = cam.reconstruct(depth, frame='w')
        # Project world points onto reference camera
        ref_coords, depth_warped = ref_cam.project(world_points, frame='w')
        depth_warped = depth_warped.view(B, C, H, W)
        # View-synthesis given the projected reference points
        return F.grid_sample(ref_image, ref_coords, mode="bilinear", padding_mode=padding_mode, align_corners=True), depth_warped, ref_coords

    @staticmethod
    def view_synthesis_generic(ref_image, depth, ref_cam, cam,
                               mode='bilinear', padding_mode='zeros', progress=0.0):
        """
        Synthesize an image from another plus a depth map.

        Parameters
        ----------
        ref_image : torch.Tensor [B,3,H,W]
            Reference image to be warped
        depth : torch.Tensor [B,1,H,W]
            Depth map from the original image
        ref_cam : Camera
            Camera class for the reference image
        cam : Camera
            Camera class for the original image
        mode : str
            Interpolation mode
        padding_mode : str
            Padding mode for interpolation

        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image in the original frame of reference
        """
        assert depth.size(1) == 1
        # Reconstruct world points from target_camera
        world_points = cam.reconstruct(depth, frame='w')
        # Project world points onto reference camera
        ref_coords, _ = ref_cam.project(world_points, progress=progress, frame='w')
        # View-synthesis given the projected reference points
        return F.grid_sample(ref_image, ref_coords, mode=mode,
                                 padding_mode=padding_mode, align_corners=True)

    def warp_ref_depth(self, depths, K, ref_K, pose):
        B, _, H, W = depths[0].shape
        ref_coordinates = []
        depths_warped = []

        for scale in self.scales:
            _, _, DH, DW = depths[scale].shape
            scale_factor = DW / float(W)

            # Reconstruct world points from target_camera
            world_points = Camera(K=K.float()).scaled(scale_factor).reconstruct(depths[scale], frame='w')
            # Project world points onto reference camera
            ref_coords, depth_warped = Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).project(world_points, frame='w')
            # Append warped depth maps
            ref_coordinates.append(ref_coords)
            depths_warped.append(depth_warped)

        # Return warped reference depth maps
        return ref_coordinates, depths_warped

    def warp_ref_image(self, disp_depth, ref_image, K, ref_K, pose, is_depth=False, padding_mode="zeros"):
        """
        Warps a reference image to produce a reconstruction of the original one.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Inverse depth map of the original image
        ref_image : torch.Tensor [B,3,H,W]
            Reference RGB image
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        pose : Pose
            Original -> Reference camera transformation

        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image (reconstructing the original one)
        """
        B, _, H, W = ref_image.shape
        # Generate cameras for all scales
        ref_warped, depth_warped, ref_coords = [], [], []
        depth = disp_depth if is_depth else inv2depth(disp_depth)
        ref_image = match_scales(ref_image, depth, self.scales)
        for scale in self.scales:
            _, _, DH, DW = disp_depth[scale].shape
            scale_factor = DW / float(W)
            cam = Camera(K=K).scaled(scale_factor)
            ref_cam = Camera(K=ref_K, Tcw=pose).scaled(scale_factor)
            ref_warp, depth_warp, ref_coord = self.view_synthesis(ref_image[scale], depth[scale], ref_cam, cam, padding_mode=padding_mode)
            ref_warped.append(ref_warp)
            depth_warped.append(depth_warp)
            ref_coords.append(ref_coord)

        # Return warped reference image
        return ref_warped, depth_warped, ref_coords
