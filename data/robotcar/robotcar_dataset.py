import json
import logging
import os
import torchvision
import data
import torch
import numpy as np
import pytorch_lightning as pl

from typing import Optional
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from robotcar_dataset_sdk.camera_model import CameraModel
from config.config import convert_to_dict
from PIL import Image


class RobotCarDataset:
    def __init__(self, cfg, transform):
        self.console_logger = logging.getLogger("pytorch_lightning.core")
        self.console_logger.info(f"[Info] Initializing RobotCar database...")
        self.scenes_to_weather = {"2014-12-09-13-21-02": "day-clear", "2014-12-16-18-44-24": "night-clear"}
        self.dataroot = cfg.DATASET.DATAROOT
        self.weather = {'train': cfg.DATASET.WEATHER.TRAIN, 'val': cfg.DATASET.WEATHER.VAL, 'test': cfg.DATASET.WEATHER.TEST}
        self.camera_sensor = "stereo/left"
        self.camera_models = {
            scene: CameraModel(os.path.join(self.dataroot, "models"), os.path.join(self.dataroot, scene, self.camera_sensor)) for
            scene in self.scenes_to_weather.keys()}
        self.temp_context = cfg.DATASET.TEMP_CONTEXT
        self.drop_static = cfg.DATASET.DROP_STATIC
        self.load_split_files_path = cfg.LOAD.SPLIT_FILES_PATH
        self.load_filter_files_path = cfg.LOAD.FILTER_FILES_PATH
        self.point_dim = 3

        self.sample_data = []
        self.split_indices = {'train': [], 'val': [], 'test': []}

        assert self.load_split_files_path is not None, "Please specify a split file path."

        def read_and_unpack_split_file(split_file, index, mode, timestamp_filter_list=None):
            if timestamp_filter_list is None:
                timestamp_filter_list = []
            filtered_samples = 0
            for line in split_file:
                timestamp = line.split(' ')
                color = {-1: timestamp[0].rstrip(), 0: timestamp[1].rstrip(), 1: timestamp[2].rstrip()} if mode != 'test' else {0: timestamp[0]}
                if color[0] in timestamp_filter_list:
                    filtered_samples += 1
                    continue
                self.sample_data.append({'scene': scene, 'color': color})
                self.split_indices[mode].append(index[0])
                index[0] += 1
            if filtered_samples > 0:
                self.console_logger.info(f"Filtered samples from {mode} set: {filtered_samples}")

        index = [0]
        for scene in self.scenes_to_weather.keys():
            if self.scenes_to_weather[scene] in self.weather['train']:
                with open(os.path.join(self.load_split_files_path, f"{scene}_train_stride_1.txt")) as train_split_file:
                    if self.load_filter_files_path is not None:
                        with open(os.path.join(self.load_filter_files_path, "filter_train_samples.txt")) as train_filter_file:
                            train_timestamp_filter_list = [timestamp.rstrip() for timestamp in train_filter_file.readlines()]
                            read_and_unpack_split_file(train_split_file, index, mode='train', timestamp_filter_list=train_timestamp_filter_list)
                    else:
                        read_and_unpack_split_file(train_split_file, index, mode='train')
            if self.scenes_to_weather[scene] in self.weather['val']:
                with open(os.path.join(self.load_split_files_path, f"{scene}_val_stride_1.txt")) as val_split_file:
                    read_and_unpack_split_file(val_split_file, index, mode='val')
            if self.scenes_to_weather[scene] in self.weather['test']:
                with open(os.path.join(self.load_split_files_path, f"{scene}_test.txt")) as test_split_file:
                    read_and_unpack_split_file(test_split_file, index, mode='test')

        self.console_logger.info(f"[Info] Finished initializing RobotCar database!")

        self.original_height = cfg.DATASET.ORIGINAL_SIZE.HEIGHT
        self.original_width = cfg.DATASET.ORIGINAL_SIZE.WIDTH

        self.load_depth_gt = cfg.DATASET.LOAD.GT.DEPTH
        self.load_pose_gt = cfg.DATASET.LOAD.GT.POSE
        self.load_color_full_size = cfg.DATASET.LOAD.COLOR_FULL_SIZE

        if self.load_pose_gt:
            poses = []
            for scene in self.scenes_to_weather.keys():
                with open(os.path.join(self.dataroot, scene, f"poses_synchronized.json")) as scene_pose_file:
                    poses += json.load(scene_pose_file)
            self.timestamp_to_pose = {data['timestamp']: data for data in poses}

        self.transform = transform

    @property
    def train_idcs(self):
        return self.split_indices['train']

    @property
    def val_idcs(self):
        return self.split_indices['val']

    @property
    def test_idcs(self):
        return self.split_indices['test']

    def map_pointcloud_to_image(self, dataroot, scene, camera_model, timestamp, original_height, original_width):
        pointcloud_dir = os.path.join(dataroot, scene, "lms_front_synchronized/vo/time_margin=+-4.0e+06", timestamp)
        pointcloud_path = f"{pointcloud_dir}.pcd.bin"
        pointcloud_3d = np.reshape(np.fromfile(pointcloud_path, dtype=np.float32), (self.point_dim, -1))
        pointcloud = np.vstack([pointcloud_3d, np.ones((1, pointcloud_3d.shape[1]))])
        uv, depth = camera_model.project(pointcloud, (original_height-1, original_width-1, 3))

        return uv, depth, pointcloud

    # Implemented according to https://github.com/w2kun/RNW/issues/2
    @staticmethod
    def generate_image_from_points(uv, depth, imsize):
        # create depth map
        h, w, = imsize
        uv = np.round(uv).astype(int)
        depth_map = np.zeros((h, w), dtype=np.float32)
        for i in range(uv.shape[1]):
            u, v, d = uv[0, i], uv[1, i], depth[i]
            if depth_map[v, u] <= 0:
                depth_map[v, u] = d
            else:
                depth_map[v, u] = min(d, depth_map[v, u])

        return depth_map

    def get_color(self, frame_idx, camera_sensor, temp):
        filename = f"{os.path.join(self.sample_data[frame_idx]['scene'], camera_sensor, self.sample_data[frame_idx]['color'][temp])}.png"
        filepath = os.path.join(self.dataroot, filename)
        return Image.open(filepath), filename

    def get_depth(self, scene, camera_model, index):
        timestamp = self.sample_data[index]['color'][0]
        uv, depth, _ = self.map_pointcloud_to_image(self.dataroot, scene, camera_model, timestamp, self.original_height, self.original_width)
        lidar_proj = self.generate_image_from_points(uv, depth, (self.original_height, self.original_width))

        return np.expand_dims(lidar_proj, axis=0)

    def get_cam_intrinsics(self, camera_model):
        return np.array([
            [camera_model.focal_length[0], 0, camera_model.principal_point[0]],
            [0, camera_model.focal_length[1], camera_model.principal_point[1]],
            [0, 0, 1]
        ])

    # For robotcar, we only support temp_context = [0, -1, 1]
    def get_pose(self, frame_idx, temp_shift):
        timestamp = int(self.sample_data[frame_idx]['color'][0])
        if temp_shift == -1:
            pose = np.vstack(self.timestamp_to_pose[timestamp]['pose_to_prev'])
        elif temp_shift == 1:
            pose = np.vstack(self.timestamp_to_pose[timestamp]['pose_to_next'])
        else:
            raise NotImplementedError(f"The pose for the frame at timestamp {timestamp} is only available for the temporal context -1 and 1.")
        return torch.Tensor(pose)

    def getitem(self, index, mode):
        assert mode in ['train', 'val', 'test']

        if mode in ['train', 'val']:
            temp_context = self.temp_context
        else:
            temp_context = [0]

        scene = self.sample_data[index]['scene']

        # Setup sample data with weather information
        weather = self.scenes_to_weather[scene]
        sample = {'weather': weather,
                  'weather_depth': weather,
                  'weather_pose': weather}

        camera_model = self.camera_models[scene]

        # Insert color image and filename (only for keyframe)
        color_keys = ['color', 'color_aug', 'color_aug_pose']
        if self.load_color_full_size:
            color_keys.append('color_original')
        for temp in temp_context:
            img, filepath = self.get_color(index, self.camera_sensor, temp)
            sample[('filename', temp)] = filepath #os.path.basename(filepath)
            for sample_key in color_keys:
                sample[(sample_key, temp)] = img.copy()

        # Get depth only for samples, not for sweeps
        if self.load_depth_gt and mode in ['val', 'test']:
            sample['depth_gt'] = torch.Tensor(self.get_depth(scene, camera_model, index))

        # Get camera intrinsics
        K = self.get_cam_intrinsics(camera_model)
        sample['K'] = torch.Tensor(K.copy())
        if self.load_color_full_size:
            sample['K_original'] = torch.Tensor(K.copy())

        # Get pose for weak supervision
        if self.load_pose_gt:
            for temp in temp_context[1:]:
                sample[('pose_gt', temp)] = torch.Tensor(self.get_pose(index, temp))

        # Apply color augmentations including resizing, flipping and color jitter
        return self.transform[mode](sample)


class RobotCarDataSubset(Dataset):
    def __init__(self, robotcar_dataset, mode):
        self.robotcar_dataset = robotcar_dataset

        assert mode in ['train', 'val', 'test']
        self.mode = mode

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.robotcar_dataset.getitem(index, self.mode)

    @property
    def indices(self):
        if self.mode == 'train':
            return self.robotcar_dataset.train_idcs
        elif self.mode == 'val':
            return self.robotcar_dataset.val_idcs
        elif self.mode == 'test':
            return self.robotcar_dataset.test_idcs
        else:
            raise NotImplementedError("Dataset mode is not implemented yet")


class RobotCarDataModule(pl.LightningDataModule):
    def __init__(self, cfg):

        super(RobotCarDataModule, self).__init__()

        self.console_logger = logging.getLogger("pytorch_lightning.core")

        self.cfg = cfg

        self.num_workers = cfg.SYSTEM.NUM_WORKERS
        self.deterministic = cfg.SYSTEM.DETERMINISTIC

        self.dataset_crop_top = cfg.DATASET.AUGMENTATION.CROP.TOP
        self.dataset_crop_left = cfg.DATASET.AUGMENTATION.CROP.LEFT
        self.dataset_crop_height = cfg.DATASET.AUGMENTATION.CROP.HEIGHT
        self.dataset_crop_width = cfg.DATASET.AUGMENTATION.CROP.WIDTH

        self.dataset_version = cfg.DATASET.VERSION
        self.resized_height = cfg.DATASET.AUGMENTATION.RESIZE.HEIGHT
        self.resized_width = cfg.DATASET.AUGMENTATION.RESIZE.WIDTH
        self.original_height = cfg.DATASET.ORIGINAL_SIZE.HEIGHT
        self.original_width = cfg.DATASET.ORIGINAL_SIZE.WIDTH
        self.temp_context = cfg.DATASET.TEMP_CONTEXT

        color_jitter_params = convert_to_dict(cfg.DATASET.AUGMENTATION.COLOR_JITTER)
        self.brightness = color_jitter_params.get('BRIGHTNESS')
        self.contrast = color_jitter_params.get('CONTRAST')
        self.saturation = color_jitter_params.get('SATURATION')
        self.hue = color_jitter_params.get('HUE')
        self.color_jitter_prob = cfg.DATASET.AUGMENTATION.COLOR_JITTER.PROBABILITY
        self.color_jitter_keys = cfg.DATASET.AUGMENTATION.COLOR_JITTER.SAMPLE_KEYS

        self.gaussian_noise_rand_min = cfg.DATASET.AUGMENTATION.GAUSSIAN_NOISE.RANDOM_MIN
        self.gaussian_noise_rand_max = cfg.DATASET.AUGMENTATION.GAUSSIAN_NOISE.RANDOM_MAX
        self.gaussian_noise_prob = cfg.DATASET.AUGMENTATION.GAUSSIAN_NOISE.PROBABILITY
        self.gaussian_noise_keys = cfg.DATASET.AUGMENTATION.GAUSSIAN_NOISE.SAMPLE_KEYS

        self.gaussian_blur_kernel_size = cfg.DATASET.AUGMENTATION.GAUSSIAN_BLUR.KERNEL_SIZE
        self.gaussian_blur_sigma = cfg.DATASET.AUGMENTATION.GAUSSIAN_BLUR.SIGMA
        self.gaussian_blur_prob = cfg.DATASET.AUGMENTATION.GAUSSIAN_BLUR.PROBABILITY
        self.gaussian_blur_keys = cfg.DATASET.AUGMENTATION.GAUSSIAN_BLUR.SAMPLE_KEYS

        self.day_night_translation_path = cfg.LOAD.DAY_NIGHT_TRANSLATION_PATH
        self.day_night_translation_direction = cfg.DATASET.AUGMENTATION.DAY_NIGHT_TRANSLATION.DIRECTION
        self.day_night_translation_prob = cfg.DATASET.AUGMENTATION.DAY_NIGHT_TRANSLATION.PROBABILITY
        self.day_night_translation_keys = cfg.DATASET.AUGMENTATION.DAY_NIGHT_TRANSLATION.SAMPLE_KEYS
        self.day_night_translation_key_frame_only = cfg.DATASET.AUGMENTATION.DAY_NIGHT_TRANSLATION.KEY_FRAME_ONLY
        self.evaluation_day_night_translation_enabled = cfg.EVALUATION.DAY_NIGHT_TRANSLATION_ENABLED

        self.train_batch_size = cfg.TRAINING.BATCH_SIZE
        self.train_repeat = cfg.TRAINING.REPEAT
        self.val_batch_size = cfg.VALIDATION.BATCH_SIZE
        self.test_batch_size = cfg.EVALUATION.BATCH_SIZE
        self.test_use_val_set = cfg.EVALUATION.USE_VALIDATION_SET

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):

        crop = data.transforms.Crop(top=self.dataset_crop_top, left=self.dataset_crop_left,
                                    height=self.dataset_crop_height, width=self.dataset_crop_width,
                                    sample_keys=['color', 'color_aug', 'color_aug_pose', 'K', 'depth_gt'], temp_context=self.temp_context)

        resize = data.transforms.Resize(size=(self.resized_height, self.resized_width),
                                        interpolation=torchvision.transforms.InterpolationMode.LANCZOS,
                                        antialias=True, sample_keys=['color', 'color_aug', 'color_aug_pose', 'K'],
                                        temp_context=self.temp_context,
                                        original_size=(self.original_height, self.original_width))

        color_jitter = data.transforms.RandomColorJitter(brightness=self.brightness,
                                                   contrast=self.contrast,
                                                   saturation=self.saturation,
                                                   hue=self.hue,
                                                   p=self.color_jitter_prob,
                                                   sample_keys=self.color_jitter_keys, temp_context=self.temp_context)
        gaussian_noise = data.transforms.RandomGaussianNoise(min_rand=self.gaussian_noise_rand_min,
                                                             max_rand=self.gaussian_noise_rand_max,
                                                             p=self.gaussian_noise_prob,
                                                             sample_size=(3, self.resized_height, self.resized_width),
                                                             sample_keys=self.gaussian_noise_keys,
                                                             temp_context=self.temp_context
                                                             )
        gaussian_blur = data.transforms.RandomGaussianBlur(kernel_size=self.gaussian_blur_kernel_size,
                                                     sigma=self.gaussian_blur_sigma,
                                                     p=self.gaussian_blur_prob,
                                                     sample_keys=self.gaussian_blur_keys, temp_context=self.temp_context)
        to_tensor = data.transforms.ToTensor(sample_keys=['color', 'color_aug', 'color_aug_pose', 'color_original', 'depth_gt'],
                                             temp_context=self.temp_context)

        train_transform_non_deterministic = [crop, resize, to_tensor]
        if self.gaussian_blur_prob > 0.0:
            train_transform_non_deterministic.insert(2, gaussian_blur)
        if self.gaussian_noise_prob > 0.0:
            train_transform_non_deterministic.insert(2, gaussian_noise)
        if self.color_jitter_prob > 0.0:
            train_transform_non_deterministic.insert(2, color_jitter)
        if self.day_night_translation_prob > 0.0:
            day_night_translation_train = data.transforms.DaytimeTranslation(
                path=self.day_night_translation_path,
                direction=self.day_night_translation_direction,
                p=self.day_night_translation_prob,
                sample_keys=self.day_night_translation_keys,
                temp_context=[0] if self.day_night_translation_key_frame_only else self.temp_context)
            train_transform_non_deterministic.insert(2, day_night_translation_train)

        train_transform = train_transform_non_deterministic if not self.deterministic else [crop, resize, to_tensor]

        val_transform = [crop, resize, to_tensor]

        if self.evaluation_day_night_translation_enabled:
            day_night_translation_eval = data.transforms.DaytimeTranslation(
                path=self.day_night_translation_path,
                direction='night->day',
                p=1.0,
                sample_keys=['color'],
                temp_context=self.temp_context)
            test_transform = [crop, resize, day_night_translation_eval, to_tensor]
        else:
            test_transform = [crop, resize, to_tensor]

        transform = {'train': data.transforms.CustomCompose(train_transform),
                     'val': data.transforms.CustomCompose(val_transform),
                     'test': data.transforms.CustomCompose(test_transform)}

        self.dataset_version = self.cfg.DATASET.VERSION
        assert self.dataset_version == "full", f"The dataset version {self.dataset_version} is not implemented yet."

        robotcar_dataset = RobotCarDataset(self.cfg, transform)

        robotcar_dataset_train = RobotCarDataSubset(robotcar_dataset=robotcar_dataset, mode='train')
        if self.train_repeat > 1:
            self.train_dataset = Subset(robotcar_dataset_train, [train_idx for train_idcs in
                                                           [robotcar_dataset_train.indices for _ in
                                                            range(self.train_repeat)]
                                                           for train_idx in train_idcs])
        else:
            self.train_dataset = Subset(robotcar_dataset_train, robotcar_dataset_train.indices)

        robotcar_subset_val = RobotCarDataSubset(robotcar_dataset=robotcar_dataset, mode='val')
        self.val_dataset = Subset(robotcar_subset_val, robotcar_subset_val.indices)

        robotcar_subset_test = RobotCarDataSubset(robotcar_dataset=robotcar_dataset, mode='val' if self.test_use_val_set else 'test')
        self.test_dataset = Subset(robotcar_subset_test, robotcar_subset_test.indices)


    @staticmethod
    def get_mean_and_std(dataloaders):
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0

        for dataloader in dataloaders:
            for data in tqdm(dataloader):
                # Mean over batch, height and width, but not over the channels
                imgs = data[('color', 0)]
                channels_sum += torch.mean(imgs, dim=[0, 2, 3])
                channels_squared_sum += torch.mean(imgs ** 2, dim=[0, 2, 3])
                num_batches += 1

        mean = channels_sum / num_batches

        # std = sqrt(E[X^2] - (E[X])^2)
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

        return mean, std

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.train_batch_size,
                                  shuffle=not self.deterministic, num_workers=self.num_workers,
                                  pin_memory=True, drop_last=False)
        self.console_logger.info(f"Number of samples in train-set: {len(self.train_dataset)}")
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True, drop_last=False)
        self.console_logger.info(f"Number of samples in val-set: {len(self.val_dataset)}")
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                                 num_workers=self.num_workers, pin_memory=True, drop_last=False)
        self.console_logger.info(f"Number of samples in test-set: {len(self.test_dataset)}")
        return test_loader
