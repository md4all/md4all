import os
from functools import partial

import torch
import torchvision.transforms.functional as ttf

from torchvision import transforms
from PIL import Image


def apply(fct, sample, sample_keys, frame_idcs):
    for sample_key in sample_keys:
        for frame_idx in frame_idcs:
            if (sample_key, frame_idx) in sample:
                sample[(sample_key, frame_idx)] = fct(sample[(sample_key, frame_idx)])

    return sample


class CustomCompose(transforms.Compose):
    def __init__(self, *args, **kwargs):
        super(CustomCompose, self).__init__(*args, **kwargs)

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize:
    def __init__(self, size, interpolation, antialias, sample_keys, temp_context, original_size):
        self.resize = transforms.Resize(size=size, interpolation=interpolation, antialias=antialias)
        self.sample_keys = sample_keys
        self.temp_context = temp_context
        self.sample_keys_contains_K = 'K' in self.sample_keys
        if self.sample_keys_contains_K:
            self.sample_keys.remove('K')
            self.size = size
            self.original_size = original_size

    def __call__(self, sample):
        if self.sample_keys_contains_K:
            sample['K'][0, :] *= (self.size[1] / self.original_size[1])
            sample['K'][1, :] *= (self.size[0] / self.original_size[0])
        return apply(self.resize, sample, self.sample_keys, self.temp_context)


class Crop:
    def __init__(self, top, left, height, width, sample_keys, temp_context):
        self.u, self.v = left, top
        self.crop = partial(transforms.functional.crop, top=top, left=left, height=height, width=width)
        self.sample_keys = sample_keys
        self.temp_context = temp_context
        self.sample_keys_contains_K = 'K' in self.sample_keys
        self.sample_keys_contains_depth_gt = 'depth_gt' in self.sample_keys
        self.sample_keys_contains_depth_gt_raw = 'depth_gt_raw' in self.sample_keys
        if self.sample_keys_contains_K:
            self.sample_keys.remove('K')
        if self.sample_keys_contains_depth_gt:
            self.sample_keys.remove('depth_gt')
        if self.sample_keys_contains_depth_gt_raw:
            self.sample_keys.remove('depth_gt_raw')

    def __call__(self, sample):
        if self.sample_keys_contains_K and 'K' in sample:
            # According to https://arxiv.org/pdf/1901.01445.pdf and https://robotcar-dataset.robots.ox.ac.uk/images/robotcar_ijrr.pdf
            # the standard camera coordinate system being in the top left corner was used
            sample['K'][0, 2] -= self.u
            sample['K'][1, 2] -= self.v
        if self.sample_keys_contains_depth_gt and 'depth_gt' in sample:
            sample['depth_gt'] = self.crop(sample['depth_gt'])
        if self.sample_keys_contains_depth_gt_raw and 'depth_gt_raw' in sample:
            sample['depth_gt_raw'] = self.crop(sample['depth_gt_raw'])
        return apply(self.crop, sample, self.sample_keys, self.temp_context)


class RandomColorJitter:

    def __init__(self, brightness, contrast, saturation, hue, p, sample_keys, temp_context):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
        self.sample_keys = sample_keys
        self.temp_context = temp_context
        self.brightness_factor = None
        self.contrast_factor = None
        self.saturation_factor = None
        self.hue_factor = None

    def __call__(self, sample):

        def color_jitter(sample_i):
            if self.brightness is not None:
                sample_i = ttf.adjust_brightness(sample_i, self.brightness_factor)
            if self.contrast is not None:
                sample_i = ttf.adjust_contrast(sample_i, self.contrast_factor)
            if self.saturation is not None:
                sample_i = ttf.adjust_saturation(sample_i, self.saturation_factor)
            if self.hue is not None:
                sample_i = ttf.adjust_hue(sample_i, self.hue_factor)
            return sample_i

        _, self.brightness_factor, self.contrast_factor, self.saturation_factor, self.hue_factor = \
            transforms.ColorJitter.get_params(brightness=self.brightness,
                                              contrast=self.contrast,
                                              saturation=self.saturation,
                                              hue=self.hue)
        if torch.rand(1) < self.p:
            sample = apply(color_jitter, sample, self.sample_keys, self.temp_context)

        return sample


class RandomGaussianNoise:
    def __init__(self, min_rand, max_rand, p, sample_size, sample_keys, temp_context):
        self.min_rand = min_rand
        self.max_rand = max_rand
        self.p = p
        self.sample_size = sample_size
        self.sample_keys = sample_keys
        self.temp_context = temp_context

    def __call__(self, sample):
        def gaussian_noise(sample_i, noise):
            t_img = sample_i
            if not isinstance(sample_i, torch.Tensor):
                if not isinstance(sample_i, Image.Image):
                    raise TypeError(f"img should be PIL Image or Tensor. Got {type(sample_i)}")

                t_img = ttf.to_tensor(sample_i, )

            output = torch.clamp(t_img + (self.min_rand + (self.max_rand - self.min_rand) * noise), 0, 1)

            if not isinstance(sample_i, torch.Tensor):
                output = ttf.to_pil_image(output, mode=sample_i.mode)
            return output

        if torch.rand(1) < self.p:
            for frame_idx in self.temp_context:
                noise = torch.randn(self.sample_size)
                for sample_key in self.sample_keys:
                    if (sample_key, frame_idx) in sample:
                        sample[(sample_key, frame_idx)] = gaussian_noise(sample[(sample_key, frame_idx)], noise)

        return sample


class RandomGaussianBlur:
    def __init__(self, kernel_size, sigma, p, sample_keys, temp_context):
        self.p = p
        self.sample_keys = sample_keys
        self.temp_context = temp_context
        self.gaussian_blur = transforms.GaussianBlur(kernel_size, sigma)

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            sample = apply(self.gaussian_blur, sample, self.sample_keys, self.temp_context)

        return sample


class RandomErasing:
    def __init__(self, p, scale, ratio, value, sample_keys, temp_context):
        self.random_erasing = transforms.RandomErasing(p, scale, ratio, value)
        self.sample_keys = sample_keys
        self.temp_context = temp_context

    def __call__(self, sample):
        return apply(self.random_erasing, sample, self.sample_keys, self.temp_context)


class ToTensor:
    def __init__(self, sample_keys, temp_context):
        self.to_tensor = transforms.ToTensor()
        self.sample_keys = sample_keys
        self.temp_context = temp_context

    def __call__(self, sample):
        return apply(self.to_tensor, sample, self.sample_keys, self.temp_context)


class NormalizeDynamic:
    def __init__(self, cfg):
        self.normalize_mode = cfg.DATASET.AUGMENTATION.NORMALIZE.MODE
        assert self.normalize_mode in ['Dataset', 'Daytime', 'Image']
        if self.normalize_mode == 'Dataset':
            self.normalize = transforms.Normalize(mean=cfg.DATASET.AUGMENTATION.NORMALIZE.DATASET.MEAN,
                                                  std=cfg.DATASET.AUGMENTATION.NORMALIZE.DATASET.STD)
        elif self.normalize_mode == 'Daytime':
            self.normalize_day = transforms.Normalize(mean=cfg.DATASET.AUGMENTATION.NORMALIZE.DAYTIME.DAY.MEAN,
                                                      std=cfg.DATASET.AUGMENTATION.NORMALIZE.DAYTIME.DAY.STD)
            self.normalize_night = transforms.Normalize(mean=cfg.DATASET.AUGMENTATION.NORMALIZE.DAYTIME.NIGHT.MEAN,
                                                        std=cfg.DATASET.AUGMENTATION.NORMALIZE.DAYTIME.NIGHT.STD)

    def normalize_daytime(self, img, daytime):
        if 'day' in daytime:
            img = self.normalize_day(img)
        elif 'night' in daytime:
            img = self.normalize_night(img)
        return img

    def __call__(self, img, daytime):
        if self.normalize_mode == 'Dataset':
            img_normalized = self.normalize(img)
        elif self.normalize_mode == 'Daytime':
            assert daytime
            if len(img.shape) == 3:
                img_normalized = self.normalize_daytime(img, daytime)
            else:
                img_normalized = torch.zeros_like(img, device=img.device)
                for b in range(len(img)):
                    img_normalized[b, :, :, :] = self.normalize_daytime(img[b, :, :, :], daytime[b])
        elif self.normalize_mode == 'Image':
            img_normalized = (img - img.mean()) / img.std()
        else:
            img_normalized = img
        return img_normalized


class DaytimeTranslation:
    def __init__(self, path, direction, p, sample_keys, temp_context):
        assert path is not None and direction in ['day->night', 'night->day', 'day-clear->day-rain', 'day-rain->day-clear']

        self.p = p
        self.sample_keys = sample_keys
        self.temp_context = temp_context
        self.direction = direction
        self.path = path

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            if self.direction == 'night->day' and 'night' in sample['weather'] or \
                    self.direction == 'day->night' and 'day' in sample['weather'] or \
                    self.direction == 'day-clear->day-rain' and 'day-clear' in sample['weather'] or \
                    self.direction == 'day-rain->day-clear' and 'day-rain' in sample['weather']:
                for sample_key in self.sample_keys:
                    for temp in self.temp_context:
                        sample[(sample_key, temp)] = Image.open(os.path.join(self.path, sample[('filename', temp)]))
            if 'color_aug' in self.sample_keys:
                sample['weather_depth'] = f"{self.direction.split('->')[1]}-translated"
            if 'color_aug_pose' in self.sample_keys:
                sample['weather_pose'] = f"{self.direction.split('->')[1]}-translated"
        return sample


class ToPILImage:
    def __init__(self, sample_keys, temp_context):
        self.to_tensor = transforms.ToPILImage()
        self.sample_keys = sample_keys
        self.temp_context = temp_context

    def __call__(self, sample):
        return apply(self.to_tensor, sample, self.sample_keys, self.temp_context)


class CondtionalDaytimeTranslationErasing:
    def __init__(self, daytime_transltion, erasing):
        self.daytime_translation = daytime_transltion
        self.erasing = erasing

        self.to_tensor = ToTensor(erasing.sample_keys, erasing.temp_context)
        self.to_pil = ToPILImage(erasing.sample_keys, erasing.temp_context)

    def __call__(self, sample):
        sample = self.daytime_translation(sample)
        sample = self.to_tensor(sample)
        sample = self.erasing(sample)
        sample = self.to_pil(sample)
        return sample
