import argparse
import os
import time
from functools import partial

from PIL import Image
from robotcar_dataset_sdk.camera_model import CameraModel
from robotcar_dataset_sdk.image import load_image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def load_demosaic_undistort_images(timestamp, images_dir, camera_model, out_dir):
    image = Image.fromarray(load_image(os.path.join(images_dir, f"{timestamp}.png"), camera_model))
    outpath = os.path.join(out_dir, f"{timestamp}.png")
    image.save(outpath)

    time.sleep(0.001)  # to visualize the progress


def precompute_rgb_and_store(dataroot, scene, camera_sensor, out_dir):
    images_dir = os.path.join(dataroot, scene, camera_sensor)
    models_dir = os.path.join(dataroot, "models")
    camera_model = CameraModel(models_dir, images_dir)
    out_dir = os.path.join(out_dir, scene, camera_sensor)
    timestamps_path = os.path.join(dataroot, scene, "stereo.timestamps")
    load_demosaic_undistort_images_partial = partial(load_demosaic_undistort_images, images_dir=images_dir,
                                                     camera_model=camera_model, out_dir=out_dir)
    with open(timestamps_path) as timestamps_file:
        timestamps = [ts.split(" ")[0] for ts in timestamps_file.readlines()]
        with ProcessPoolExecutor(max_workers=20) as executor:
            results = list(tqdm(executor.map(load_demosaic_undistort_images_partial, timestamps), total=len(timestamps)))
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demosaicing and undistorting rgb images')
    parser.add_argument('--dataroot', type=str, help='Dataroot of robotcar dataset', required=True)
    parser.add_argument('--scenes', nargs='+', help='Scenes for demosaicing and undistorting', required=True)
    parser.add_argument('--camera_sensor', type=str, help='Camera sensor that should be used for precomputation', required=True)
    parser.add_argument('--out_dir', type=str, help='Output directory that is used as new dataroot', required=True)

    args = parser.parse_args()
    dataroot = args.dataroot
    scenes = args.scenes
    camera_sensor = args.camera_sensor
    out_dir = args.out_dir
    print(f"Starting rgb image precomputing with scenes {scenes}")

    for scene in scenes:
        print(f"Demosaicing and undistorting rgb images of scene {scene} ...")
        precompute_rgb_and_store(dataroot, scene, camera_sensor, out_dir)
        print(f"Finished demosaicing and undistorting rgb images of scene {scene} ...")
