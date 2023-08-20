# Adapted from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/losses/velocity_loss.py

import pytorch_lightning as pl


class VelocityLoss(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        assert cfg.DATASET.LOAD.GT.POSE
        self.temp_context = cfg.DATASET.TEMP_CONTEXT

    def forward(self, inputs, outputs):
        pred_pose = [outputs[("pose", frame_id)] for frame_id in self.temp_context[1:]]
        gt_pose_context = [inputs[('pose_gt', frame_id)] for frame_id in self.temp_context[1:]]

        pred_trans = [pose.mat[:, :3, -1].norm(dim=-1) for pose in pred_pose]
        gt_trans = [pose[:, :3, -1].norm(dim=-1) for pose in gt_pose_context]

        # Calculate velocity supervision loss
        loss = sum([(pred - gt).abs().mean() for pred, gt in zip(pred_trans, gt_trans)]) / len(gt_trans)
        return loss
