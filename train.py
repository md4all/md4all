import os
import socket
import time

import pytorch_lightning as pl
import torch

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from config.config import get_cfg, get_parser
from data.nuscenes_dataset import NuScenesDataModule
from data.robotcar.robotcar_dataset import RobotCarDataModule

from trainer import Md4All


def train():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    pl.seed_everything(cfg.SYSTEM.SEED, True)

    if cfg.DATASET.NAME == "nuscenes":
        dm = NuScenesDataModule(cfg)
    elif cfg.DATASET.NAME == "robotcar":
        dm = RobotCarDataModule(cfg)
    else:
        raise NotImplementedError(f"The dataset {cfg.DATASET.NAME} is not implemented.")

    model = Md4All(cfg, is_train=True)

    save_dir = os.path.join(
        cfg.SAVE.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.EXPERIMENT_NAME
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_dir)

    checkpoint_callback = ModelCheckpoint(monitor=cfg.SAVE.MONITOR, save_last=cfg.SAVE.LAST, save_top_k=cfg.SAVE.TOP_K, mode='min', dirpath=cfg.SAVE.CHECKPOINT_PATH, filename=cfg.EXPERIMENT_NAME + '-{epoch:02d}')
    #early_stop_callback = EarlyStopping(monitor='metrics_day-clear/abs_rel', min_delta=0.00, patience=3, verbose=False, mode='min')

    trainer = pl.Trainer(
        accelerator=cfg.SYSTEM.ACCELERATOR,
        devices=cfg.SYSTEM.DEVICES,
        accumulate_grad_batches=cfg.TRAINING.ACCUMULATE_GRAD_BATCHES,
        precision=cfg.SYSTEM.PRECISION,
        gradient_clip_val=cfg.TRAINING.GRAD_NORM_CLIP,
        max_epochs=cfg.TRAINING.EPOCHS,
        logger=tb_logger,
        log_every_n_steps=cfg.SAVE.LOGGING_INTERVAL,
        profiler='simple',
        callbacks=[checkpoint_callback], #callbacks=[checkpoint_callback, early_stop_callback],
        deterministic=cfg.SYSTEM.DETERMINISTIC or cfg.SYSTEM.DETERMINISTIC_ALGORITHMS,
        benchmark=cfg.SYSTEM.BENCHMARK
    )

    # Enable warn_only since some modules do not have a deterministic algorithm implemented
    if cfg.SYSTEM.DETERMINISTIC or cfg.SYSTEM.DETERMINISTIC_ALGORITHMS:
        torch.use_deterministic_algorithms(mode=True, warn_only=True)

    trainer.fit(model, dm, ckpt_path=cfg.LOAD.CHECKPOINT_PATH)


if __name__ == '__main__':
    train()
