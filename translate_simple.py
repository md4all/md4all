# Please be aware that the usage of the ForkGAN model is regulated by the LICENSE file in models/ForkGAN
import argparse
import os

from PIL import Image
from torchvision.transforms import transforms

from data.custom_dataset import Crop
from models.ForkGAN.models import create_model
from models.ForkGAN.options.test_options import TestOptions
from torchvision.utils import save_image

# Define necessary transforms
normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
to_tensor = transforms.ToTensor()


def pre_pocess(input, additional_prep):
    return normalize(to_tensor(additional_prep(input)).unsqueeze(0).cuda())


def post_process(output):
    return (output + 1.0) / 2.0


def translate(forkgan, img, additional_prep):
    img_tcn = pre_pocess(img, additional_prep)
    forkgan_output = forkgan.backward_test(img_tcn)
    img_translated = post_process(forkgan_output)
    return img_translated


if __name__ == '__main__':
    # Parser configurations
    parser = argparse.ArgumentParser(description="Translate specified images to a specific adverse condition.")
    parser.add_argument("--image_path", type=str, help="Path to input image", required=True)
    parser.add_argument("--checkpoint_dir", type=str, help="Checkpoint path", required=True)
    parser.add_argument("--model_name", type=str, help="Model name", choices=["forkgan_nuscenes_day_night",
                                                                              "forkgan_nuscenes_day_rain",
                                                                              "forkgan_robotcar_day_night"],
                        required=True)
    parser.add_argument("--crop_top", type=int, help="Height the images should be cropped to", default=0)
    parser.add_argument("--crop_left", type=int, help="Width the images should be cropped to", default=0)
    parser.add_argument("--crop_height", type=int, help="Height the images should be cropped to")
    parser.add_argument("--crop_width", type=int, help="Width the images should be cropped to")
    parser.add_argument("--resize_height", type=int, help="Height the images should be resized to")
    parser.add_argument("--resize_width", type=int, help="Width the images should be resized to")
    parser.add_argument("--output_dir", type=str, help="Path where the translated image should be stored",
                        required=True)

    # Parse and prepare necessary information
    args = parser.parse_args()
    image_path = args.image_path
    checkpoint_dir = args.checkpoint_dir
    model_name = args.model_name
    crop_top = args.crop_top
    crop_left = args.crop_left
    crop_height = args.crop_height
    crop_width = args.crop_width
    resize_height = args.resize_height
    resize_width = args.resize_width
    output_dir = args.output_dir

    assert os.path.isdir(checkpoint_dir)
    assert os.path.isfile(os.path.join(checkpoint_dir, f"{model_name}_net_G_A.pth"))
    assert os.path.isfile(os.path.join(checkpoint_dir, f"{model_name}_net_G_B.pth"))
    assert os.path.isfile(os.path.join(checkpoint_dir, f"{model_name}_scaler.pth"))

    # Create ForkGAN model and load weights
    opt = TestOptions().parse()
    opt.checkpoints_dir = checkpoint_dir
    opt.epoch = model_name
    forkgan = create_model(opt)
    forkgan.setup(opt)

    # Create output directory if not existent
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Create resize and cropping
    tfs_ = []
    if crop_height and crop_width:
        crop = Crop(crop_top, crop_left, crop_height, crop_width)
        tfs_.append(crop)
    if resize_height and resize_width:
        resize = transforms.Resize(size=(resize_height, resize_width),
                                   interpolation=transforms.InterpolationMode.LANCZOS)
        tfs_.append(resize)
    tfs = transforms.Compose(tfs_) if tfs_ else [lambda x: x]

    # Translate input images and store them
    img = Image.open(image_path)
    img_name, img_extension = os.path.basename(image_path).split(".")
    img_translated = translate(forkgan, img, tfs)
    save_image(img_translated, os.path.join(output_dir, f"{img_name}_{model_name}.{img_extension}"))
