import logging
import os
import json
from typing import Optional

import torch
import torchvision
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
from PIL import Image
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

from torch.utils.data import Dataset, Subset, DataLoader
from nuscenes import NuScenes, NuScenesExplorer

import data
from config.config import convert_to_dict


class NuScenesDataset:
    def __init__(self, cfg, transform, dataset_version):
        self.console_logger = logging.getLogger("pytorch_lightning.core")
        self.transform = transform
        self.dataset_version = dataset_version
        self.dataroot = os.path.join(cfg.DATASET.DATAROOT, self.dataset_version)
        self.drop_static = cfg.DATASET.DROP_STATIC

        self.orientation_to_camera = {
            "front": "CAM_FRONT",
            "front_right": "CAM_FRONT_RIGHT",
            "front_left": "CAM_FRONT_LEFT",
            "back_right": "CAM_BACK_RIGHT",
            "back_left": "CAM_BACK_LEFT",
            "back": "CAM_BACK"
        }
        self.camera_sensors = [self.orientation_to_camera[orientation] for orientation in cfg.DATASET.ORIENTATIONS]

        self.console_logger.info(f"[Info] Initializing NuScenes {dataset_version} official database...")
        self.nusc = NuScenes(version="v1.0-" + self.dataset_version, dataroot=self.dataroot, verbose=False)
        if 'trainval' in self.nusc.version and self.drop_static:
            self.drop_static_samples()
        self.nusc_explorer = NuScenesExplorer(self.nusc)
        self.console_logger.info(f"[Info] Finished initializing NuScenes {dataset_version} official database!")

        self.original_width = cfg.DATASET.ORIGINAL_SIZE.WIDTH
        self.original_height = cfg.DATASET.ORIGINAL_SIZE.HEIGHT
        self.res_width = cfg.DATASET.AUGMENTATION.RESIZE.WIDTH
        self.res_height = cfg.DATASET.AUGMENTATION.RESIZE.HEIGHT
        self.scales = cfg.DATASET.SCALES
        self.temp_context = cfg.DATASET.TEMP_CONTEXT
        self.weather_conditions_train = cfg.DATASET.WEATHER.TRAIN
        self.weather_conditions_val = cfg.DATASET.WEATHER.VAL
        self.weather_conditions_eval = cfg.DATASET.WEATHER.TEST

        self.load_depth_gt = cfg.DATASET.LOAD.GT.DEPTH
        self.load_pose_gt = cfg.DATASET.LOAD.GT.POSE
        self.load_color_full_size = cfg.DATASET.LOAD.COLOR_FULL_SIZE

        if dataset_version in ['trainval', 'mini']:
            self.scenes_weather_info = self.get_scenes_weather_info()
            self.train_idcs, self.val_idcs = self.get_train_val_indices()
        elif dataset_version == 'test':
            self.nusc_test_weather_to_scene_range = {
                'day-rain': [("n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537294719112404",
                              'n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295445612404')],
                'night-clear': [("n015-2018-11-14-19-21-41+0800__CAM_FRONT__1542194502662460",
                                 "n015-2018-11-14-19-52-02+0800__CAM_FRONT__1542196694912460")]
            }

            self.modify_weather_descriptions_of_nuscenes_scenes()
            self.scenes_weather_info = self.get_scenes_weather_info()
            self.test_idcs = self.get_test_indices()

        self.sample_to_sensor_sample_data = {}
        for i in range(len(self.nusc.sample)):
            for j, camera_sensor in enumerate(self.camera_sensors):
                self.sample_to_sensor_sample_data[i * len(self.camera_sensors) + j] = (i, camera_sensor)

    def correct_index(self, index):
        # First frame index
        if self.nusc.sample[index]['prev'] == '':
            return index + 1
        # Last frame index
        elif self.nusc.sample[index]['next'] == '':
            return index - 1
        # No correction
        return index

    def getitem(self, index, mode):
        assert mode in ['train', 'val', 'eval']
        is_train = mode == 'train'
        is_val = mode == 'val'

        if is_train or is_val:
            temp_context = self.temp_context
        else:
            temp_context = [0]

        (frame_idx, camera_sensor) = self.sample_to_sensor_sample_data[index]
        if is_train or is_val:
            frame_idx = self.correct_index(frame_idx)

        # Setup sample data with weather information
        weather = self.get_weather(frame_idx)
        sample = {'weather': weather,
                  'weather_depth': weather,
                  'weather_pose': weather}

        # Insert color image and filename (only for keyframe)
        color_keys = ['color', 'color_aug', 'color_aug_pose']
        if self.load_color_full_size:
            color_keys.append('color_original')
        for temp in temp_context:
            img, filepath = self.get_color(frame_idx, camera_sensor, temp)
            sample[('filename', temp)] = filepath
            for sample_key in color_keys:
                sample[(sample_key, temp)] = img.copy()

        # Get depth only for samples, not for sweeps
        if self.check_depth(frame_idx, camera_sensor):
            sample['depth_gt'] = torch.Tensor(self.get_depth(frame_idx, camera_sensor))

        # Get camera intrinsics
        K = self.get_cam_intrinsics(camera_sensor)
        sample['K'] = torch.Tensor(K.copy())
        if self.load_color_full_size:
            sample['K_original'] = torch.Tensor(K.copy())

        # Get pose for weak supervision
        if self.load_pose_gt:
            for temp in temp_context[1:]:
                sample[('pose_gt', temp)] = torch.Tensor(self.get_pose(frame_idx, camera_sensor, temp))

        # Apply color augmentations including resizing, flipping and color jitter
        return self.transform[mode](sample)

    def get_cam_sample_data(self, frame_index, camera_sensor, temp_shift):
        keyframe = self.nusc.get('sample_data', self.nusc.sample[frame_index]['data'][camera_sensor])
        temp_dir = 0

        if camera_sensor in ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]:
            if temp_shift < 0:
                temp_dir = 'prev'
            elif temp_shift > 0:
                temp_dir = 'next'
        elif camera_sensor in ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
            if temp_shift < 0:
                temp_dir = 'next'
            elif temp_shift > 0:
                temp_dir = 'prev'

        i = 0
        while i < abs(temp_shift):
            temp_token = keyframe[temp_dir]
            if temp_token == '':
                return None
            keyframe = self.nusc.get('sample_data', temp_token)
            i += 1

        return keyframe

    def get_color(self, frame_index, camera_sensor, temp_shift):
        filename = self.get_cam_sample_data(frame_index, camera_sensor, temp_shift)['filename']
        return Image.open(os.path.join(self.nusc.dataroot, filename)), filename

    def get_weather(self, frame_index):
        return self.scenes_weather_info[self.nusc.sample[frame_index]['scene_token']]

    def check_depth(self, index, sensor):
        keyframe = self.nusc.get('sample_data', self.nusc.sample[index]['data'][sensor])
        if 'sweeps' in keyframe['filename']:
            return False
        return self.load_depth_gt

    def get_depth(self, index, sensor):
        lidar_data = self.nusc.get('sample_data', self.nusc.sample[index]['data']['LIDAR_TOP'])
        cam_data = self.nusc.get('sample_data', self.nusc.sample[index]['data'][sensor])
        points, depth, _ = self.nusc_explorer.map_pointcloud_to_image(pointsensor_token=lidar_data['token'],
                                                                      camera_token=cam_data['token'])
        lidar_proj = self.generate_image_from_points(points[:2], depth, (self.original_height, self.original_width))

        return np.expand_dims(lidar_proj, axis=0)

    def get_cam_intrinsics(self, camera_sensor):
        sensor_token = self.nusc.field2token("sensor", "channel", camera_sensor)[0]
        calibrated_sensor_token = self.nusc.field2token("calibrated_sensor", "sensor_token", sensor_token)[0]
        return np.array(self.nusc.get("calibrated_sensor", calibrated_sensor_token)["camera_intrinsic"])

    def get_pose(self, frame_index, camera_sensor, temp_shift):
        cam_data_origin = self.get_cam_sample_data(frame_index, camera_sensor, 0)
        cam_data_shifted = self.get_cam_sample_data(frame_index, camera_sensor, temp_shift)

        ego_pose_record = self.nusc.get('ego_pose', cam_data_shifted['ego_pose_token'])
        ego_pose_origin_record = self.nusc.get('ego_pose', cam_data_origin['ego_pose_token'])

        ego_to_global_transform = transform_matrix(translation=ego_pose_record['translation'],
                                                   rotation=Quaternion(ego_pose_record['rotation']))
        ego_origin_to_global_transform = transform_matrix(translation=np.array(ego_pose_origin_record['translation']),
                                                          rotation=Quaternion(ego_pose_origin_record['rotation']))

        calibrated_sensor_record = self.nusc.get('calibrated_sensor', cam_data_shifted['calibrated_sensor_token'])
        calibrated_sensor_origin_record = self.nusc.get('calibrated_sensor', cam_data_origin['calibrated_sensor_token'])

        ref_to_ego_transform = transform_matrix(translation=calibrated_sensor_record['translation'],
                                                rotation=Quaternion(calibrated_sensor_record["rotation"]))
        ref_to_ego_origin_transform = transform_matrix(translation=calibrated_sensor_origin_record['translation'],
                                                       rotation=Quaternion(calibrated_sensor_origin_record['rotation']))

        return np.linalg.inv(ref_to_ego_transform) @ np.linalg.inv(
            ego_to_global_transform) @ ego_origin_to_global_transform @ ref_to_ego_origin_transform

    def get_train_val_indices(self):
        if 'mini' in self.nusc.version:
            scenes_split_names_map = {'train': [], 'val': []}
            for idx, scene in enumerate(self.nusc.scene):
                if idx in [1, 9]:
                    scenes_split_names_map['val'].append(scene['name'])
                else:
                    scenes_split_names_map['train'].append(scene['name'])
        else:
            scenes_split_names_map = create_splits_scenes(verbose=False)

        scenes_split_tokens_map = {'train': [], 'val': []}
        weather_counter = {
            'train-day-clear': 0, 'train-day-rain': 0, 'train-night-clear': 0, 'train-night-rain': 0,
            'val-day-clear': 0, 'val-day-rain': 0, 'val-night-clear': 0, 'val-night-rain': 0
        }

        self.console_logger.info(
            f"Creating train-val split. The training set includes scenes with visibility/weather conditions: "
            f"{self.weather_conditions_train}")

        for scene in self.nusc.scene:
            scene_token = scene['token']
            weather_curr = self.scenes_weather_info[scene_token]
            if scene['name'] in scenes_split_names_map['train'] and weather_curr in self.weather_conditions_train:
                scenes_split_tokens_map['train'].append(scene_token)
                weather_counter['train-' + weather_curr] += 1
            elif scene['name'] in scenes_split_names_map['val'] and weather_curr in self.weather_conditions_val:
                scenes_split_tokens_map['val'].append(scene_token)
                weather_counter['val-' + weather_curr] += 1
        train_sample_indices, val_samples_indices = self.split_train_val_temp_valid_samples(scenes_split_tokens_map)

        self.console_logger.info(f"Weather distribution in train and val scenes:\n{weather_counter}")

        return np.asarray(train_sample_indices), np.asarray(val_samples_indices)

    def load_modified_scene_description(self, modified_json):
        # Load filtered json file
        self.nusc.scene = self.nusc.__load_table__(os.path.splitext(modified_json)[0])
        # Access to protected member, but this is the easiest way to do this
        table = 'scene'
        self.nusc._token2ind[table] = {}
        for ind, member in enumerate(getattr(self.nusc, table)):
            self.nusc._token2ind[table][member['token']] = ind

    def write_data_to_json(self, fpath, filtered_samples):
        self.console_logger.info(f"Writing filtered samples to Json: {fpath}.")

        with open(fpath, 'w') as f:
            f.write(json.dumps(filtered_samples, indent=0))
        f.close()

        self.console_logger.info(f"Finished writing JSON file.")

    def modify_weather_descriptions_of_nuscenes_scenes(self):
        modified_json = f"scene_with_weather.json"
        new_filepath = os.path.join(self.nusc.dataroot, self.nusc.version, modified_json)

        if not os.path.isfile(new_filepath):
            # Add weather information to sample data based on manually extracted information
            excluded_cam_sensors = [excluded_cam for excluded_cam in list(self.orientation_to_camera.values()) if excluded_cam not in self.camera_sensors]
            modified_scene = []
            for s in self.nusc.scene:
                for camera_sensor in self.camera_sensors:
                    filename = os.path.splitext(os.path.basename(self.nusc.get('sample_data', self.nusc.get('sample', s['first_sample_token'])['data'][camera_sensor])['filename']))[0]
                    if camera_sensor in filename and not any([excluded_cam_sensor in filename for excluded_cam_sensor in excluded_cam_sensors]):
                        scene = s.copy()
                        for k, v in self.nusc_test_weather_to_scene_range.items():
                            if any([scene_range[0] <= filename <= scene_range[1] for scene_range in v]):
                                scene['description'] = k
                                break
                            else:
                                scene['description'] = 'day-clear'
                        modified_scene.append(scene)

            self.write_data_to_json(new_filepath, modified_scene)

        self.console_logger.info(f"Filtered samples file {new_filepath} exists. Loading from JSON...")
        self.load_modified_scene_description(modified_json)

    def get_test_indices(self):
        scenes_split_names_map = create_splits_scenes(verbose=False)
        scenes_test_tokens_map = []
        weather_counter = {
            'test-day-clear': 0, 'test-day-rain': 0, 'test-night-clear': 0, 'test-night-rain': 0
        }

        self.console_logger.info(
            f"Creating test split. The test set includes scenes with visibility/weather conditions: "
            f"{self.weather_conditions_eval}")

        for scene in self.nusc.scene:
            scene_token = scene['token']
            weather_curr = self.scenes_weather_info[scene_token]
            if scene['name'] in scenes_split_names_map['test'] and weather_curr in self.weather_conditions_eval:
                scenes_test_tokens_map.append(scene_token)
                weather_counter['test-' + weather_curr] += 1

        test_sample_indices = []
        for idx, sample in enumerate(self.nusc.sample):
            scene_token = sample['scene_token']
            if scene_token in scenes_test_tokens_map:
                self.add_index_to_index_list(idx, test_sample_indices)

        self.console_logger.info(f"Weather distribution in test scenes:\n{weather_counter}")

        return np.asarray(test_sample_indices)

    def add_index_to_index_list(self, i, sample_indices):
        for j, camera_sensor in enumerate(self.camera_sensors):
            sample_indices.append(i * len(self.camera_sensors) + j)

    def split_train_val_temp_valid_samples(self, scenes_split_lut):
        train_sample_indices = []
        val_sample_indices = []
        for i, sample in enumerate(self.nusc.sample):
            scene_token = sample['scene_token']
            if scene_token in scenes_split_lut['train']:
                self.add_index_to_index_list(i, train_sample_indices)
            elif scene_token in scenes_split_lut['val']:
                self.add_index_to_index_list(i, val_sample_indices)

        return train_sample_indices, val_sample_indices

    def is_velocity_above_thresh(self, sample_data, vel_thresh):
        if not sample_data['prev'] or not sample_data['next']:
            return False

        translation = np.array(self.nusc.get('ego_pose', sample_data['ego_pose_token'])['translation'])
        vel = []
        for dir in ['prev', 'next']:
            sample_data_pn = self.nusc.get('sample_data', sample_data[dir])
            translation_pn = np.array(self.nusc.get('ego_pose', sample_data_pn['ego_pose_token'])['translation'])
            euclidean_dist = np.linalg.norm(translation - translation_pn)
            time = np.abs(sample_data_pn['timestamp'] - sample_data['timestamp']) * 1e-6
            vel.append(float(euclidean_dist / time))
        vel_mean = sum(vel) / len(vel)
        return vel_mean >= vel_thresh

    def get_samples_below_velocity(self, velocity):
        scenes_split_names_map = create_splits_scenes(verbose=False)
        scenes_split_tokens_map = {scene['token']: 'train' if scene['name'] in scenes_split_names_map['train'] else 'val' for scene in self.nusc.scene }
        filtered_samples = []
        for camera_sensor in self.camera_sensors:
            filtered_samples += [s for s in self.nusc.sample if scenes_split_tokens_map[s['scene_token']] == 'val' or self.is_velocity_above_thresh(self.nusc.get('sample_data', s['data'][camera_sensor]), velocity)]
        return filtered_samples

    def drop_static_samples(self, filtered_samples_json=f"train_samples_dynamic.json"):
        filepath = os.path.join(self.nusc.dataroot, self.nusc.version, filtered_samples_json)

        if not os.path.isfile(filepath):
            self.console_logger.info(f"No JSON file found to filter samples! Create file...")
            filtered_samples = self.get_samples_below_velocity(0.5)
            self.write_data_to_json(filepath, filtered_samples)

        self.console_logger.info(f"Filter samples using file {filepath}!")
        self.nusc.sample = self.nusc.__load_table__(os.path.splitext(filtered_samples_json)[0])
        table = 'sample'
        self.nusc._token2ind[table] = {}
        for ind, member in enumerate(getattr(self.nusc, table)):
            self.nusc._token2ind[table][member['token']] = ind

    def get_scenes_weather_info(self):
        scenes_weather_info = {}
        for scene in self.nusc.scene:
            weather_info = ""
            if 'night' in scene['description'].lower():
                weather_info += 'night'
            else:
                weather_info += 'day'
            if 'rain' in scene['description'].lower():
                weather_info += '-rain'
            else:
                weather_info += '-clear'
            scenes_weather_info[scene['token']] = weather_info
        return scenes_weather_info

    @staticmethod
    def generate_image_from_points(points, features, imsize):
        h, w = imsize
        points = points.astype(np.int32)
        projection = np.zeros((h, w), dtype=np.float32)
        projection[points[1], points[0]] = features

        return projection


class NuScenesDataSubset(Dataset):
    def __init__(self, nuscenes_dataset, mode, eval_use_val_set=True):
        self.nuscenes_dataset = nuscenes_dataset

        assert mode in ['train', 'val', 'eval']
        self.mode = mode
        self.eval_use_val_set = eval_use_val_set

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.nuscenes_dataset.getitem(index, self.mode)

    @property
    def indices(self):
        if self.mode == 'train':
            return self.nuscenes_dataset.train_idcs
        elif self.mode == 'val' or (self.mode == 'eval' and self.eval_use_val_set):
            return self.nuscenes_dataset.val_idcs
        elif self.mode == 'eval':
            return self.nuscenes_dataset.test_idcs
        else:
            raise NotImplementedError("Dataset mode is not implemented yet")


class NuScenesDataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super(NuScenesDataModule, self).__init__()

        self.console_logger = logging.getLogger("pytorch_lightning.core")

        self.cfg = cfg

        self.num_workers = cfg.SYSTEM.NUM_WORKERS
        self.deterministic = cfg.SYSTEM.DETERMINISTIC
        self.precision = cfg.SYSTEM.PRECISION

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

        self.day_clear_day_rain_translation_path = cfg.LOAD.DAY_CLEAR_DAY_RAIN_TRANSLATION_PATH
        self.day_clear_day_rain_translation_direction = cfg.DATASET.AUGMENTATION.DAY_CLEAR_DAY_RAIN_TRANSLATION.DIRECTION
        self.day_clear_day_rain_translation_prob = cfg.DATASET.AUGMENTATION.DAY_CLEAR_DAY_RAIN_TRANSLATION.PROBABILITY
        self.day_clear_day_rain_translation_keys = cfg.DATASET.AUGMENTATION.DAY_CLEAR_DAY_RAIN_TRANSLATION.SAMPLE_KEYS
        self.day_clear_day_rain_translation_key_frame_only = cfg.DATASET.AUGMENTATION.DAY_CLEAR_DAY_RAIN_TRANSLATION.KEY_FRAME_ONLY
        self.evaluation_day_clear_day_rain_translation_enabled = cfg.EVALUATION.DAY_CLEAR_DAY_RAIN_TRANSLATION_ENABLED

        self.train_batch_size = cfg.TRAINING.BATCH_SIZE
        self.train_repeat = cfg.TRAINING.REPEAT
        self.val_batch_size = cfg.VALIDATION.BATCH_SIZE
        self.eval_batch_size = cfg.EVALUATION.BATCH_SIZE
        self.eval_use_val_set = cfg.EVALUATION.USE_VALIDATION_SET

        self.train_dataset = None
        self.val_dataset = None
        self.eval_dataset = None

    def setup(self, stage: Optional[str] = None):
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
                                                             temp_context=self.temp_context)
        gaussian_blur = data.transforms.RandomGaussianBlur(kernel_size=self.gaussian_blur_kernel_size,
                                                     sigma=self.gaussian_blur_sigma,
                                                     p=self.gaussian_blur_prob,
                                                     sample_keys=self.gaussian_blur_keys, temp_context=self.temp_context)
        to_tensor = data.transforms.ToTensor(sample_keys=['color', 'color_aug', 'color_aug_pose', 'color_original', 'depth_gt'],
                                             temp_context=self.temp_context)

        train_transform_non_deterministic = [resize, to_tensor]
        if self.gaussian_blur_prob > 0.0:
            train_transform_non_deterministic.insert(1, gaussian_blur)
        if self.gaussian_noise_prob > 0.0:
            train_transform_non_deterministic.insert(1, gaussian_noise)
        if self.color_jitter_prob > 0.0:
            train_transform_non_deterministic.insert(1, color_jitter)
        if self.day_night_translation_prob > 0.0:
            day_night_translation_train = data.transforms.DaytimeTranslation(
                path=self.day_night_translation_path,
                direction=self.day_night_translation_direction,
                p=self.day_night_translation_prob,
                sample_keys=self.day_night_translation_keys,
                temp_context=[0] if self.day_night_translation_key_frame_only else self.temp_context)
            train_transform_non_deterministic.insert(1, day_night_translation_train)
        if self.day_clear_day_rain_translation_prob > 0.0:
            day_clear_day_rain_translation_train = data.transforms.DaytimeTranslation(
                path=self.day_clear_day_rain_translation_path,
                direction=self.day_clear_day_rain_translation_direction,
                p=self.day_clear_day_rain_translation_prob,
                sample_keys=self.day_clear_day_rain_translation_keys,
                temp_context=[0] if self.day_clear_day_rain_translation_key_frame_only else self.temp_context)
            train_transform_non_deterministic.insert(1, day_clear_day_rain_translation_train)

        train_transform = train_transform_non_deterministic if not self.deterministic else [resize, to_tensor]
        val_transform = [resize, to_tensor]
        eval_test_transform = [resize, to_tensor]

        if self.evaluation_day_night_translation_enabled:
            day_night_translation_eval = data.transforms.DaytimeTranslation(
                path=self.day_night_translation_path,
                direction=self.day_night_translation_direction,
                p=1.0,
                sample_keys=self.day_night_translation_keys,
                temp_context=self.temp_context)
            eval_test_transform.insert(1, day_night_translation_eval)
        if self.evaluation_day_clear_day_rain_translation_enabled:
            day_clear_day_rain_translation_eval = data.transforms.DaytimeTranslation(
                path=self.day_clear_day_rain_translation_path,
                direction=self.day_clear_day_rain_translation_direction,
                p=1.0,
                sample_keys=self.day_clear_day_rain_translation_keys,
                temp_context=self.temp_context)
            eval_test_transform.insert(1, day_clear_day_rain_translation_eval)

        transform = {'train': data.transforms.CustomCompose(train_transform),
                     'val': data.transforms.CustomCompose(val_transform),
                     'eval': data.transforms.CustomCompose(eval_test_transform)}

        if self.dataset_version == 'full':
            nuscenes_dataset = NuScenesDataset(self.cfg, transform, 'trainval')
            nuscenes_subset_eval = NuScenesDataSubset(nuscenes_dataset=nuscenes_dataset if self.eval_use_val_set else NuScenesDataset(self.cfg, transform, 'test'), mode='eval', eval_use_val_set=self.eval_use_val_set)
            self.eval_dataset = Subset(nuscenes_subset_eval, nuscenes_subset_eval.indices)

        elif self.cfg.DATASET.VERSION == 'mini':
            nuscenes_dataset = NuScenesDataset(self.cfg, transform, 'mini')
            nuscenes_subset_eval = NuScenesDataSubset(nuscenes_dataset=nuscenes_dataset, mode='eval', eval_use_val_set=True)
            self.eval_dataset = Subset(nuscenes_subset_eval, nuscenes_subset_eval.indices)

        else:
            raise NotImplementedError(f"The dataset version {self.cfg.DATASET.VERSION} is not implemented")

        nuscenes_subset_train = NuScenesDataSubset(nuscenes_dataset=nuscenes_dataset, mode='train')
        if self.train_repeat > 1:
            self.train_dataset = Subset(nuscenes_subset_train, [train_idx for train_idcs in
                                                           [nuscenes_subset_train.indices for _ in
                                                            range(self.train_repeat)]
                                                           for train_idx in train_idcs])
        else:
            self.train_dataset = Subset(nuscenes_subset_train, nuscenes_subset_train.indices)

        nuscenes_subset_val = NuScenesDataSubset(nuscenes_dataset=nuscenes_dataset, mode='val')
        self.val_dataset = Subset(nuscenes_subset_val, nuscenes_subset_val.indices)

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
        test_loader = DataLoader(dataset=self.eval_dataset, batch_size=self.eval_batch_size, shuffle=False,
                                 num_workers=self.num_workers, pin_memory=True, drop_last=False)
        self.console_logger.info(f"Number of samples in test-set: {len(self.eval_dataset)}")
        return test_loader

