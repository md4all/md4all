# Adapt from https://github.com/TRI-ML/packnet-sfm/blob/de53b310533ff6b01eaa23a8ba5ac01bac5587b1/packnet_sfm/geometry/pose_utils.py

import torch
import numpy as np

from models.md2.layers import rot_from_axisangle, get_translation_matrix, euler2mat

"""
def euler2mat(angle):
    # Convert euler angles to rotation matrix
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).view(B, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    return rot_mat
"""


def pose_vec2mat(vec, mode='euler', invert=False):
    """Convert Euler parameters to transformation matrix."""
    if mode is None:
        return vec
    trans, rot = vec[:, :3].unsqueeze(-1), vec[:, 3:]
    if mode == 'euler':
        R = euler2mat(rot)
    elif mode == 'rodrigues':
        R = rot_from_axisangle(rot.unsqueeze(1))
    else:
        raise ValueError('Rotation mode not supported {}'.format(mode))
    t = trans.clone()
    if invert:
        R = R.transpose(1, 2)
        t *= -1
    T = get_translation_matrix(t)
    mat = R @ T if invert else T @ R
    #mat = torch.cat([R, t], dim=2)  # [B,3,4]
    return mat[:, :3, :]


def invert_pose(T):
    """Inverts a [B,4,4] torch.tensor pose"""
    Tinv = torch.eye(4, device=T.device, dtype=T.dtype).repeat([len(T), 1, 1])
    Tinv[:, :3, :3] = torch.transpose(T[:, :3, :3], -2, -1)
    Tinv[:, :3, -1] = torch.bmm(-1. * Tinv[:, :3, :3], T[:, :3, -1].unsqueeze(-1)).squeeze(-1)
    return Tinv


def invert_pose_numpy(T):
    """Inverts a [4,4] np.array pose"""
    Tinv = np.copy(T)
    R, t = Tinv[:3, :3], Tinv[:3, 3]
    Tinv[:3, :3], Tinv[:3, 3] = R.T, - np.matmul(R.T, t)
    return Tinv
