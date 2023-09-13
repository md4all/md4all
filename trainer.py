import logging
import time
import os

import numpy as np
import torch
import pytorch_lightning as pl
import pandas as pd

from torch import optim

from evaluation import evaluator
from losses.TotalLoss import TotalLoss
from models.depth_net import DepthNet
from models.pose_net import PoseNet
from utils.depth import inv2depth
from utils.image import flip_lr
from utils.pose import Pose
from visualization.visualize import Visualizer


class Md4All(pl.LightningModule):

    def __init__(self, cfg, is_train):
        super(Md4All, self).__init__()

        self.experiment_name = cfg.EXPERIMENT_NAME

        self.save_hyperparameters()

        self.deterministic = cfg.SYSTEM.DETERMINISTIC

        self.vis_interval = cfg.SAVE.VIS_INTERVAL

        self.depth_estimation_for_ref_frames = False
        self.rotation_mode = cfg.MODEL.POSE.ROTATION_MODE

        self.temp_context = cfg.DATASET.TEMP_CONTEXT
        self.flip_prob = cfg.DATASET.AUGMENTATION.HORIZONTAL_FLIP_PROBABILITY

        self.depth_lr = cfg.OPTIMIZER.DEPTH_LR
        self.pose_lr = cfg.OPTIMIZER.POSE_LR
        self.weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY
        self.step_size = cfg.OPTIMIZER.STEP_SIZE
        self.gamma = cfg.OPTIMIZER.GAMMA

        self.batch_size_training = cfg.TRAINING.BATCH_SIZE
        self.batch_size_validation = cfg.VALIDATION.BATCH_SIZE
        self.batch_size_evaluation = cfg.EVALUATION.BATCH_SIZE

        self.depth_model = DepthNet(cfg)
        self.pose_model = PoseNet(cfg)
        # This is needed otherwise the evaluation of md4allDD would require to load checkpoint for teacher even if it is not used
        self.loss = TotalLoss(cfg, is_train)

        self.evaluator = evaluator.Evaluator(self.depth_model, cfg)
        self.evaluation_quantitative_res_path = cfg.EVALUATION.SAVE.QUANTITATIVE_RES_PATH
        self.evaluation_qualitative_res_path = cfg.EVALUATION.SAVE.QUALITATIVE_RES_PATH
        self.evaluation_visualization_set = cfg.EVALUATION.SAVE.VISUALIZATION_SET
        self.visualizer = Visualizer(self.temp_context, self.loss.photometric_weight > 0.0,
                                     self.evaluation_qualitative_res_path, self.evaluation_visualization_set,
                                     cfg.EVALUATION.SAVE.RGB, cfg.EVALUATION.SAVE.DEPTH.PRED,
                                     cfg.EVALUATION.SAVE.DEPTH.GT, cfg.EVALUATION.DEPTH.MIN_DEPTH,
                                     cfg.EVALUATION.DEPTH.MAX_DEPTH)

    def log_losses(self, is_train):
        if is_train:
            mode = 'train'
            batch_size = self.batch_size_training
        else:
            mode = 'val'
            batch_size = self.batch_size_validation
        for loss in self.loss.activated_losses:
            self.log(f"losses/avg_{mode}_{loss}", self.loss.running_avg_loss(loss), batch_size=batch_size, on_epoch=False,
                     on_step=True, prog_bar=True)

    def depth_estimation(self, inputs, is_train):
        do_flip = is_train and not self.deterministic and torch.rand(1) < self.flip_prob
        cond_flip = lambda x: flip_lr(x) if do_flip else x
        outputs = self.depth_model(cond_flip(inputs[('color_aug', 0)]), inputs['weather_depth'])
        inv_depth = {scale: cond_flip(disp) for scale, disp in outputs.items()}
        return inv_depth

    def pose_estimation(self, inputs):
        pose_vec = self.pose_model(inputs[('color_aug_pose', 0)], [inputs["color_aug_pose", i] for i in self.temp_context[1:]], inputs['weather_pose'])
        poses = {("pose", frame_id): Pose.from_vec(pose_vec[:, i], self.rotation_mode) for i, frame_id in
                 zip(range(pose_vec.shape[1]), self.temp_context[1:])}
        return poses

    def shared_step(self, inputs, is_train):
        disps = self.depth_estimation(inputs, is_train)
        poses = self.pose_estimation(inputs)

        outputs = {**disps, **poses}

        loss = self.loss(inputs, outputs)

        return outputs, loss

    def training_step(self, batch, batch_idx):
        outputs, loss = self.shared_step(batch, is_train=True)

        self.log_losses(is_train=True)
        if self.global_step % self.vis_interval == 0:
            self.visualizer.visualize_train(batch, outputs, self.logger, self.global_step)

        return loss

    def on_training_epoch_end(self):
        self.loss.reset_losses()

    def validation_step(self, batch, batch_idx):
        outputs, loss = self.shared_step(batch, is_train=False)

        self.log_losses(is_train=False)

        self.evaluator.evaluate_depth(batch)

        return loss

    def on_validation_epoch_end(self):
        self.loss.reset_losses()

        metrics = self.evaluator.compute_average_metrics_and_export()
        for condition in metrics.keys():
            for obj_class in metrics[condition].keys():
                for metric_name, metric_value in metrics[condition][obj_class].items():
                    self.log(f"metrics_{condition}/{obj_class}/{metric_name}", metric_value)

        self.evaluator.reset_metrics()

    def test_step(self, batch, batch_idx):
        outputs = self.evaluator.evaluate_batch(batch)

        if self.evaluation_qualitative_res_path is not None:
            self.visualizer.visualize_test(batch, outputs)

    def on_test_epoch_end(self):
        console_logger = logging.getLogger("pytorch_lightning.core")

        metrics = self.evaluator.compute_average_metrics_and_export()
        metrics_all = metrics['all-conditions']
        metrics_day = metrics['day-clear']

        console_logger.info("Finished testing!")
        console_logger.info(f"Metrics: {metrics_all}")
        console_logger.info(f"Day-clear metrics: {metrics_day}")

        # Save metrics to csv
        if self.evaluation_quantitative_res_path is not None:
            metrics_file = os.path.join(self.evaluation_quantitative_res_path, f"{self.experiment_name}_result_metrics_{time.strftime('%d%B%Yat%H:%M:%S%Z')}.csv")
            df = pd.concat({k: pd.DataFrame(v).T for k, v in metrics.items()}, axis=0)
            console_logger.info(f"Saving evaluation metrics to csv file: {metrics_file}")
            df.to_csv(metrics_file)

    def on_predict_start(self):
        def check_path_and_create_dir(path):
            assert path
            if not os.path.isdir(path):
                os.makedirs(path, exist_ok=True)

        check_path_and_create_dir(self.evaluation_quantitative_res_path)
        check_path_and_create_dir(self.evaluation_qualitative_res_path)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.depth_model(batch[('color', 0)], batch.get('weather'))

        # Save depth map
        if self.evaluation_quantitative_res_path:
            depths = inv2depth(outputs[('disp', 0, 0)]).cpu().numpy()
            for j, depth in enumerate(depths):
                filename = os.path.basename(batch[('filename', 0)][j]).split('.')[0]
                np.save(os.path.join(self.evaluation_quantitative_res_path, f"{filename}_depth.npy"), depth)

        # Save visualized depth map
        if self.evaluation_qualitative_res_path:
            self.visualizer.visualize_predict(batch, outputs)

        return outputs

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.depth_model.parameters(), 'lr': self.depth_lr},
            {'params': self.pose_model.parameters(), 'lr': self.pose_lr}
        ],
            weight_decay=self.weight_decay)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        return [optimizer], [scheduler]

    def get_progress_bar_dict(self):
        # Do not show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
