import logging
import os.path

import pytorch_lightning as pl

from functools import partial
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class Crop:
    def __init__(self, top, left, height, width):
        self.crop = partial(transforms.functional.crop, top=top, left=left, height=height, width=width)

    def __call__(self, img):
        return self.crop(img)


class CustomDataset(Dataset):
    def __init__(self, image_paths, daytimes=None, transform=None):
        self.image_paths = image_paths
        self.daytimes = daytimes
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)

        # Transform image if transformation is defined
        if self.transform:
            img = self.transform(img)

        sample = {('color', 0): img,
                  ('filename', 0): os.path.basename(img_path).split('.')[0]}

        # Use daytime information if available (not necessary)
        if self.daytimes:
            sample['weather'] = self.daytimes[idx]

        return sample


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, cfg, image_paths, daytimes=None):
        super(CustomDataModule, self).__init__()

        assert cfg.DATASET.AUGMENTATION.NORMALIZE.MODE != "Daytime" or daytimes

        self.console_logger = logging.getLogger("pytorch_lightning.core")

        self.image_paths = image_paths
        self.daytimes = daytimes

        self.num_workers = cfg.SYSTEM.NUM_WORKERS
        self.crop_top = cfg.DATASET.AUGMENTATION.CROP.TOP
        self.crop_left = cfg.DATASET.AUGMENTATION.CROP.LEFT
        self.crop_height = cfg.DATASET.AUGMENTATION.CROP.HEIGHT
        self.crop_width = cfg.DATASET.AUGMENTATION.CROP.WIDTH
        self.resized_height = cfg.DATASET.AUGMENTATION.RESIZE.HEIGHT
        self.resized_width = cfg.DATASET.AUGMENTATION.RESIZE.WIDTH

        self.transform = None

    def setup(self, stage: str) -> None:
        self.transform = transforms.Compose([
            Crop(self.crop_top, self.crop_left, self.crop_height, self.crop_width),
            transforms.Resize(size=(self.resized_height, self.resized_width),
                              interpolation=transforms.InterpolationMode.LANCZOS, antialias=True),
            transforms.ToTensor()
        ])

    def predict_dataloader(self):
        predict_dataset = CustomDataset(self.image_paths, self.daytimes, self.transform)
        predict_loader = DataLoader(dataset=predict_dataset, batch_size=1, shuffle=False,
                                    num_workers=self.num_workers, pin_memory=True, drop_last=False)
        self.console_logger.info(f"Number of samples in predict-set: {len(predict_dataset)}")
        return predict_loader
