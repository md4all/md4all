# Adapted from https://github.com/wayveai/fiery/blob/master/fiery/config.py

import argparse
from fvcore.common.config import CfgNode as _CfgNode


def convert_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary."""
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, _CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                'Key {} with value {} is not a valid type; valid types: {}'.format(
                    '.'.join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


class CfgNode(_CfgNode):
    """Remove once https://github.com/rbgirshick/yacs/issues/19 is merged."""

    def convert_to_dict(self):
        return convert_to_dict(self)


CN = CfgNode

_C = CN()
_C.EXPERIMENT_NAME = 'default'

# General settings
_C.SYSTEM = CN()
_C.SYSTEM.SEED = 42  # Random seed that is used
_C.SYSTEM.ACCELERATOR = 'gpu'  # Accelerator to use
_C.SYSTEM.DEVICES = 1  # Number of devices to use
_C.SYSTEM.PRECISION = 32  # Precision to use for floating point operations
_C.SYSTEM.NUM_WORKERS = 8  # Number of workers for the dataloader
_C.SYSTEM.DETERMINISTIC = False  # For debugging
_C.SYSTEM.DETERMINISTIC_ALGORITHMS = False  # Use deterministic algorithms as far as possible
_C.SYSTEM.BENCHMARK = True  # Use cuda benchmark

_C.TRAINING = CN()
_C.TRAINING.EPOCHS = 20  # Number of epochs to train
_C.TRAINING.BATCH_SIZE = 16  # Batch size for training
_C.TRAINING.GRAD_NORM_CLIP = None  # For gradient clipping
_C.TRAINING.REPEAT = 2  # Repeat training indices
_C.TRAINING.ACCUMULATE_GRAD_BATCHES = None  # Accumulates gradients over this number of batches before stepping the optimizer (Todo: Must be changed to 1 for newer pl versions)

_C.VALIDATION = CN()
_C.VALIDATION.BATCH_SIZE = 16  # Batch size for validation

_C.SAVE = CN()
_C.SAVE.CHECKPOINT_PATH = 'log/checkpoints'  # Folder within md4all to store checkpoints
_C.SAVE.LOG_DIR = 'log/training'  # Directory to store tensorboard logs
_C.SAVE.VIS_INTERVAL = 1000  # Visualization interval for tensorboard
_C.SAVE.LOGGING_INTERVAL = 1  # Logging interval for tensorboard
_C.SAVE.MONITOR = 'metrics_night/everything/abs_rel_pp'  # Metric to monitor
_C.SAVE.TOP_K = 3  # Number of top models (w.r.t. monitoring metric) to save
_C.SAVE.LAST = True  # For saving last checkpoint

_C.LOAD = CN()
_C.LOAD.CHECKPOINT_PATH = None  # Checkpoint that should be loaded
_C.LOAD.DAY_NIGHT_TRANSLATION_PATH = None  # Path where the day-night-translated images are stored
_C.LOAD.DAY_CLEAR_DAY_RAIN_TRANSLATION_PATH = None  # Path where the day-clear-day-rain-translated images are stored
_C.LOAD.DAYTIME_TRANSLATION_TEACHER_PATH = None  # Path where the teacher model (e.g. a baseline ckpt) is stored
_C.LOAD.SPLIT_FILES_PATH = None  # Split files for the dataset (only used for RobotCar)
_C.LOAD.FILTER_FILES_PATH = None  # To filter overexposed and static samples (only used for RobotCar)

_C.DATASET = CN()
_C.DATASET.DATAROOT = '/mnt/data/nuscenes'  # Path to main data folder containing nuScenes or RobotCar
_C.DATASET.VERSION = 'full'  # Mainly used for nuScenes to be able to load also mini or full i.e. trainval + test (optional)
_C.DATASET.NAME = 'nuscenes'  # Dataset name
_C.DATASET.ORIGINAL_SIZE = CN()
_C.DATASET.ORIGINAL_SIZE.HEIGHT = 900  # Original height of dataset images
_C.DATASET.ORIGINAL_SIZE.WIDTH = 1600  # Original width of dataset images
_C.DATASET.ORIENTATIONS = ['front']  # Not used (originally to also load other orientations)
_C.DATASET.WEATHER = CN()
_C.DATASET.WEATHER.TRAIN = ['day-clear']  # Weather conditions allowed in training set
_C.DATASET.WEATHER.VAL = ['day-clear', 'day-rain', 'night-clear', 'night-rain']  # Weather conditions allowed in validation set
_C.DATASET.WEATHER.TEST = ['day-clear', 'day-rain', 'night-clear', 'night-rain']  # Weather conditions allowed in test set
_C.DATASET.DROP_STATIC = True  # Drop nearly static samples
_C.DATASET.SCALES = [0, 1, 2, 3]  # Scales to use (was never changed)
_C.DATASET.TEMP_CONTEXT = [0, -1, 1]  # Temporal context frame to use (was never changed)

_C.DATASET.LOAD = CN()
_C.DATASET.LOAD.GT = CN()
_C.DATASET.LOAD.GT.DEPTH = True  # Load ground truth depth in dataloader
_C.DATASET.LOAD.GT.POSE = True  # Load ground truth pose in dataloader
_C.DATASET.LOAD.COLOR_FULL_SIZE = False  # Load color samples also in full size in dataloader

_C.DATASET.AUGMENTATION = CN()
_C.DATASET.AUGMENTATION.CROP = CN()
_C.DATASET.AUGMENTATION.CROP.TOP = 0  # Top pixel to start image cropping
_C.DATASET.AUGMENTATION.CROP.LEFT = 0  # Left pixel to start image cropping
_C.DATASET.AUGMENTATION.CROP.HEIGHT = 900  # Height for image cropping
_C.DATASET.AUGMENTATION.CROP.WIDTH = 1600  # Width for image cropping

_C.DATASET.AUGMENTATION.RESIZE = CN()
_C.DATASET.AUGMENTATION.RESIZE.HEIGHT = 320  # Image resizing height
_C.DATASET.AUGMENTATION.RESIZE.WIDTH = 576  # Image resizing width
_C.DATASET.AUGMENTATION.HORIZONTAL_FLIP_PROBABILITY = 0.5  # Probability of flipping image horizontally during training

_C.DATASET.AUGMENTATION.COLOR_JITTER = CN()
_C.DATASET.AUGMENTATION.COLOR_JITTER.PROBABILITY = 1.0  # Probability of applying color jittering
_C.DATASET.AUGMENTATION.COLOR_JITTER.BRIGHTNESS = [0.8, 1.2]  # Color jitter brightness interval
_C.DATASET.AUGMENTATION.COLOR_JITTER.CONTRAST = [0.8, 1.2]  # Color jitter contrast interval
_C.DATASET.AUGMENTATION.COLOR_JITTER.SATURATION = [0.8, 1.2]  # Color jitter saturation interval
_C.DATASET.AUGMENTATION.COLOR_JITTER.HUE = [-0.05, 0.05]  # Color jitter hue interval
_C.DATASET.AUGMENTATION.COLOR_JITTER.SAMPLE_KEYS = ['color_aug', 'color_aug_pose']  # Apply color jitter to

_C.DATASET.AUGMENTATION.GAUSSIAN_NOISE = CN()
_C.DATASET.AUGMENTATION.GAUSSIAN_NOISE.PROBABILITY = 0.0  # Probability of applying gaussian noise
_C.DATASET.AUGMENTATION.GAUSSIAN_NOISE.RANDOM_MIN = 0.005  # Minimum gaussian noise to apply
_C.DATASET.AUGMENTATION.GAUSSIAN_NOISE.RANDOM_MAX = 0.05  # Maximum gaussian noise to apply
_C.DATASET.AUGMENTATION.GAUSSIAN_NOISE.SAMPLE_KEYS = ['color_aug', 'color_aug_pose']  # Apply gaussian noise to

_C.DATASET.AUGMENTATION.GAUSSIAN_BLUR = CN()
_C.DATASET.AUGMENTATION.GAUSSIAN_BLUR.PROBABILITY = 0.0  # Probability of applying gaussian blurring
_C.DATASET.AUGMENTATION.GAUSSIAN_BLUR.KERNEL_SIZE = 9  # Kernel size for blurring
_C.DATASET.AUGMENTATION.GAUSSIAN_BLUR.SIGMA = 2  # Sigma for blurring
_C.DATASET.AUGMENTATION.GAUSSIAN_BLUR.SAMPLE_KEYS = ['color_aug', 'color_aug_pose']  # Apply gaussian blurring to

_C.DATASET.AUGMENTATION.NORMALIZE = CN()
_C.DATASET.AUGMENTATION.NORMALIZE.MODE = 'Dataset'  # Normalization type (Dataset, Daytime, Image) => Use Image mode if daytime is not known a priori instead of Dataset
_C.DATASET.AUGMENTATION.NORMALIZE.DATASET = CN()
_C.DATASET.AUGMENTATION.NORMALIZE.DATASET.MEAN = 0.45  # Dataset normalization mean
_C.DATASET.AUGMENTATION.NORMALIZE.DATASET.STD = 0.225  # Dataset normalization std
_C.DATASET.AUGMENTATION.NORMALIZE.DAYTIME = CN()
_C.DATASET.AUGMENTATION.NORMALIZE.DAYTIME.DAY = CN()
_C.DATASET.AUGMENTATION.NORMALIZE.DAYTIME.DAY.MEAN = 0.4130  # Daytime normalization mean for day
_C.DATASET.AUGMENTATION.NORMALIZE.DAYTIME.DAY.STD = 0.2093  # Day normalization std for day
_C.DATASET.AUGMENTATION.NORMALIZE.DAYTIME.NIGHT = CN()
_C.DATASET.AUGMENTATION.NORMALIZE.DAYTIME.NIGHT.MEAN = 0.1652  # Daytime normalization mean for night
_C.DATASET.AUGMENTATION.NORMALIZE.DAYTIME.NIGHT.STD = 0.1728  # Daytime normalization std for night

_C.DATASET.AUGMENTATION.DAY_NIGHT_TRANSLATION = CN()
_C.DATASET.AUGMENTATION.DAY_NIGHT_TRANSLATION.PROBABILITY = 0.0  # Probability of applying day-night translation
_C.DATASET.AUGMENTATION.DAY_NIGHT_TRANSLATION.DIRECTION = 'day->night'  # Direction of translation
_C.DATASET.AUGMENTATION.DAY_NIGHT_TRANSLATION.SAMPLE_KEYS = ['color_aug']  # Apply day-night translation to
_C.DATASET.AUGMENTATION.DAY_NIGHT_TRANSLATION.KEY_FRAME_ONLY = True  # Apply translation only to key-frame

_C.DATASET.AUGMENTATION.DAY_CLEAR_DAY_RAIN_TRANSLATION = CN()
_C.DATASET.AUGMENTATION.DAY_CLEAR_DAY_RAIN_TRANSLATION.PROBABILITY = 0.0  # Probability of applying day-clear-day-rain translation
_C.DATASET.AUGMENTATION.DAY_CLEAR_DAY_RAIN_TRANSLATION.DIRECTION = 'day-clear->day-rain'  # Direction of translation
_C.DATASET.AUGMENTATION.DAY_CLEAR_DAY_RAIN_TRANSLATION.SAMPLE_KEYS = ['color_aug']  # Apply day-clear-day-rain translation to
_C.DATASET.AUGMENTATION.DAY_CLEAR_DAY_RAIN_TRANSLATION.KEY_FRAME_ONLY = True  # Apply translation only to key-frame

_C.MODEL = CN()
_C.MODEL.DEPTH = CN()
_C.MODEL.DEPTH.ENCODER = CN()
_C.MODEL.DEPTH.ENCODER.VERSION = 'resnet-18'  # Depth encoder version
_C.MODEL.DEPTH.ENCODER.PRETRAINED = True  # Load depth encoder weights
_C.MODEL.DEPTH.MIN_DEPTH = 0.1  # Min depth
_C.MODEL.DEPTH.MAX_DEPTH = 80.0  # Max depth

_C.MODEL.POSE = CN()
_C.MODEL.POSE.ROTATION_MODE = 'euler'  # Rotation modeling type
_C.MODEL.POSE.TEMPORAL_ORDERING = False  # Temporal ordering for pose network
_C.MODEL.POSE.ENCODER = CN()
_C.MODEL.POSE.ENCODER.VERSION = 'resnet-18'  # Pose encoder version
_C.MODEL.POSE.ENCODER.PRETRAINED = True  # Load pose encoder weights
_C.MODEL.POSE.ENCODER.NUM_INPUT_IMAGES = 2  # Number of input images for pose encoder
_C.MODEL.POSE.DECODER = CN()
_C.MODEL.POSE.DECODER.NUM_INPUT_FEATURES = 1  # Number of input features for pose decoder
_C.MODEL.POSE.DECODER.NUM_FRAMES_TO_PREDICT_FOR = 2  # Number of frames to predict pose for

_C.LOSS = CN()
_C.LOSS.DEPTH_UPSAMPLE = True  # Upsample depth for loss computations
_C.LOSS.PHOTOMETRIC = CN()
_C.LOSS.PHOTOMETRIC.WEIGHT = 1.0  # Photometric loss weight
_C.LOSS.PHOTOMETRIC.REDUCE_OP = 'min'  # Photometric reduce operation
_C.LOSS.PHOTOMETRIC.SSIM = CN()
_C.LOSS.PHOTOMETRIC.SSIM.WEIGHT = 0.85  # SSIM weight
_C.LOSS.PHOTOMETRIC.SSIM.C1 = 1e-4  # SSIM C1
_C.LOSS.PHOTOMETRIC.SSIM.C2 = 9e-4  # SSIM C2
_C.LOSS.PHOTOMETRIC.PADDING_MODE = 'zeros'  # Padding mode for photometric loss
_C.LOSS.PHOTOMETRIC.CLIP = 0.0  # Clipping value for photometric loss (only applied for positive values)
_C.LOSS.PHOTOMETRIC.AUTOMASK = True  # Use automask for photometric loss
_C.LOSS.SMOOTHNESS_WEIGHT = 1e-3  # Smoothness loss weight
_C.LOSS.VELOCITY_WEIGHT = 0.02  # Velocity loss weight
_C.LOSS.SUPERVISED = CN()
_C.LOSS.SUPERVISED.WEIGHT = 0.0  # Supervised loss weight
_C.LOSS.SUPERVISED.METHOD = 'abs_rel'  # Supervised loss type

_C.OPTIMIZER = CN()
_C.OPTIMIZER.DEPTH_LR = 2e-4  # Learning rate for depth network
_C.OPTIMIZER.POSE_LR = 2e-4  # Learning rate for pose network
_C.OPTIMIZER.WEIGHT_DECAY = 0.0  # Weight decay for optimizer
_C.OPTIMIZER.STEP_SIZE = 5  # Step size for optimizer
_C.OPTIMIZER.GAMMA = 0.5  # Gamma for optimizer

_C.EVALUATION = CN()
_C.EVALUATION.BATCH_SIZE = 16  # Evaluation batch size
_C.EVALUATION.CONDITION_WISE = True  # Do evaluation condition-wise
_C.EVALUATION.DAY_NIGHT_TRANSLATION_ENABLED = False  # Day-night translation during evaluation
_C.EVALUATION.DAY_CLEAR_DAY_RAIN_TRANSLATION_ENABLED = False  # Day-clear-day-rain translation during evaluation
_C.EVALUATION.USE_VALIDATION_SET = True  # Use validation set for evaluation

_C.EVALUATION.DEPTH = CN()
_C.EVALUATION.DEPTH.MIN_DEPTH = 0.1  # Minimum depth for evaluation
_C.EVALUATION.DEPTH.MAX_DEPTH = 80.0  # Maximum depth for evaluation

_C.EVALUATION.SAVE = CN()
_C.EVALUATION.SAVE.QUANTITATIVE_RES_PATH = None  # Path to save quantitative results
_C.EVALUATION.SAVE.QUALITATIVE_RES_PATH = None  # Path to save qualitative results
_C.EVALUATION.SAVE.VISUALIZATION_SET = []  # Set of images to visualize (empty list means all of them)
_C.EVALUATION.SAVE.RGB = True  # Save RGB images
_C.EVALUATION.SAVE.DEPTH = CN()
_C.EVALUATION.SAVE.DEPTH.PRED = True  # Save depth predictions
_C.EVALUATION.SAVE.DEPTH.GT = True  # Save depth gt


def get_parser():
    parser = argparse.ArgumentParser(description='md4all training')
    parser.add_argument('--config', default='', metavar='FILE', help='Path to config file')
    parser.add_argument(
        'opts', help='Modify config options using the command-line', default=None, nargs=argparse.REMAINDER,
    )
    return parser


def get_cfg(args=None, cfg_dict=None):
    """ First get default config. Then merge cfg_dict. Then merge according to args. """

    cfg = _C.clone()

    if cfg_dict is not None:
        cfg.merge_from_other_cfg(CfgNode(cfg_dict))

    if args is not None:
        if args.config:
            cfg.merge_from_file(args.config)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
    return cfg
