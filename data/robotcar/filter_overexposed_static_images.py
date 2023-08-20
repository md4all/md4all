import argparse
import json
import os.path

import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter overexposed and completely photometric inconsistent rgb images')
    parser.add_argument('--dataroot', type=str, help='Dataroot of robotcar dataset', required=True)
    parser.add_argument('--scenes', nargs='+', help='Scenes for filtering', required=True)
    parser.add_argument('--modes', nargs='+', help='Modes for filtering (only train or val)', required=True)
    parser.add_argument('--camera_sensor', type=str, help='Camera sensor that should be used for filter', required=True)
    parser.add_argument('--out_dir', type=str, help='Output directory for filtered images', required=True)

    args = parser.parse_args()
    dataroot = args.dataroot
    scenes = args.scenes
    modes = args.modes
    camera_sensor = args.camera_sensor
    out_dir = args.out_dir

    to_tensor = torchvision.transforms.ToTensor()
    filtered_timestamps = []
    for scene in scenes:
        with open(os.path.join(dataroot, scene, "poses_synchronized.json")) as scene_pose_file:
            poses = json.load(scene_pose_file)
            timestamp_to_pose = {data['timestamp']: data for data in poses}
            for mode in modes:
                images_dir = os.path.join(dataroot, scene, camera_sensor)
                with open(os.path.join(dataroot, f"splits/{scene}_{mode}_stride_1.txt")) as timestamps_file:
                    for timestamp_triplet in tqdm(timestamps_file.readlines()):
                        timestamp_triplet = timestamp_triplet.rstrip().split(" ")
                        color = to_tensor(Image.open(os.path.join(images_dir, f"{timestamp_triplet[1]}.png")))
                        color_prev = to_tensor(Image.open(os.path.join(images_dir, f"{timestamp_triplet[0]}.png")))
                        color_next = to_tensor(Image.open(os.path.join(images_dir, f"{timestamp_triplet[2]}.png")))

                        pose = timestamp_to_pose[int(timestamp_triplet[1])]
                        t_prev = np.array(pose['pose_to_prev'])[:3, 3]
                        t_next = np.array(pose['pose_to_next'])[:3, 3]
                        if np.linalg.norm(t_prev, 2) < 1e-3 or np.linalg.norm(t_next) < 1e-3:
                            filtered_timestamps.append(timestamp_triplet[1])
                            save_image(color, os.path.join(out_dir, f"static/{timestamp_triplet[1]}.png"))
                            save_image(color_prev, os.path.join(out_dir, f"static/{timestamp_triplet[0]}.png"))
                            save_image(color_next, os.path.join(out_dir, f"static/{timestamp_triplet[2]}.png"))
                        elif torch.mean(color[:768, :, :]).item() >= 0.9 or \
                                abs(torch.mean(color_prev[:768, :, :]) - torch.mean(color[:768, :, :])) > 0.05 or \
                                abs(torch.mean(color_next[:768, :, :]) - torch.mean(color[:768, :, :])) > 0.05:
                            filtered_timestamps.append(timestamp_triplet[1])
                            save_image(color, os.path.join(out_dir, f"overexposed/{timestamp_triplet[1]}.png"))
                            save_image(color_prev, os.path.join(out_dir, f"overexposed/{timestamp_triplet[0]}.png"))
                            save_image(color_next, os.path.join(out_dir, f"overexposed/{timestamp_triplet[2]}.png"))
        print(filtered_timestamps)
        print(len(filtered_timestamps))

        with open(os.path.join(out_dir, "filter_train_samples.txt"), mode='w') as timestamp_filter_file:
            for timestamp in filtered_timestamps:
                timestamp_filter_file.write(f"{timestamp}\n")
