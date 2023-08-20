import argparse
import os
import re
import time
from functools import partial

import numpy as np

from robotcar_dataset_sdk.build_pointcloud import build_pointcloud
from robotcar_dataset_sdk.camera_model import CameraModel
from robotcar_dataset_sdk.transform import build_se3_transform
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal


def save_pointcloud(timestamp, pointcloud, out_dir_pcd):
    filepath = f"{os.path.join(out_dir_pcd, str(timestamp))}.pcd.bin"
    pointcloud.tofile(filepath)
    pcd_cmp = np.reshape(np.fromfile(filepath, dtype=np.float32), (3, -1))
    if not np.allclose(pointcloud, pcd_cmp):
        raise AssertionError("Loaded pointcloud should correspond to stored point cloud.")

    time.sleep(0.001)  # to visualize the progress


def build_and_store_pointcloud(timestamp, laser_dir, poses_file, extrinsics_dir, G_camera_posesource, time_span, out_dir_pcd):
    pointcloud, reflectance = build_pointcloud(laser_dir, poses_file, extrinsics_dir,
                                               timestamp - time_span, timestamp + time_span, timestamp)
    pointcloud = np.dot(G_camera_posesource, pointcloud)
    pcd = np.asarray(pointcloud[:3, :]).astype(np.float32)

    save_pointcloud(timestamp, pcd, out_dir_pcd)


def precompute_and_store_pointcloud(timestamps_path, laser_dir, poses_file, extrinsics_dir, camera_model, time_span, out_dir_pcd):
    extrinsics_path = os.path.join(extrinsics_dir, camera_model.camera + '.txt')
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

    G_camera_vehicle = build_se3_transform(extrinsics)
    G_camera_posesource = None

    poses_type = re.search('(vo|ins|rtk)\.csv', poses_file).group(1)
    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
    else:
        # VO frame and vehicle frame are the same
        G_camera_posesource = G_camera_vehicle

    build_project_store_pointcloud_partial = partial(build_and_store_pointcloud,
                                                     laser_dir=laser_dir, poses_file=poses_file,
                                                     extrinsics_dir=extrinsics_dir,
                                                     G_camera_posesource=G_camera_posesource, time_span=time_span,
                                                     out_dir_pcd=out_dir_pcd)

    with open(timestamps_path) as timestamp_file:
        timestamps = [int(t.split(" ")[0] if "test" in timestamps_path else t.split(" ")[1]) for t in timestamp_file.readlines()]

        with ProcessPoolExecutor(max_workers=20) as executor:
            results = list(tqdm(executor.map(build_project_store_pointcloud_partial, timestamps), total=len(timestamps)))
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build, project, and store the pointclouds')
    parser.add_argument('--dataroot', type=str, help='Dataroot of robotcar dataset', required=True)
    parser.add_argument('--scenes', nargs='+', help='Scenes for pose interpolation', required=True)
    parser.add_argument('--modes', nargs='+', help='Modes for pose interpolation', required=True)
    parser.add_argument('--camera_sensor', type=str, default="stereo/left", help='Modes for pose interpolation')
    parser.add_argument('--laser_sensor', type=str, default="lms_front", help='Modes for pose interpolation')
    parser.add_argument('--poses_subdir', type=str, default="vo/vo.csv", help='Subdirectory of poses')
    parser.add_argument('--time_span', type=int, default=4e6, help='Time span for building and reprojecting pointcloud to key frame')

    args = parser.parse_args()
    dataroot = args.dataroot
    scenes = args.scenes
    modes = args.modes
    camera_sensor = args.camera_sensor
    laser_sensor = args.laser_sensor
    poses_subdir = args.poses_subdir
    poses_to_use = os.path.basename(poses_subdir).split(".")[0]
    time_span = args.time_span
    time_span_scientific = '%.1e' % Decimal(str(time_span))
    print(f"Starting to build point clouds with scenes {scenes} and modes {modes} with time span {time_span_scientific}")

    for scene in scenes:
        for mode in modes:
            print(f"Building point clouds for scene {scene} mode {mode}...")
            image_dir = os.path.join(dataroot, scene, camera_sensor)
            laser_dir = os.path.join(dataroot, scene, laser_sensor)
            poses_file = os.path.join(dataroot, scene, poses_subdir)
            extrinsics_dir = f"{dataroot}/extrinsics"
            camera_model = CameraModel(f"{dataroot}/models", os.path.join(dataroot, scene, camera_sensor))

            out_dir_pcd = os.path.join(dataroot, scene, f"{laser_sensor}_synchronized/{poses_to_use}/time_margin=+-{time_span_scientific}")
            out_dir_vis = os.path.join(dataroot, scene, f"{laser_sensor}_visualized/{poses_to_use}/time_margin=+-{time_span_scientific}")
            timestamps_path = os.path.join(dataroot, f"splits/{scene}_{mode}")
            timestamps_path += ".txt" if mode == 'test' else "_stride_1.txt"

            os.makedirs(out_dir_pcd, exist_ok=True)

            precompute_and_store_pointcloud(timestamps_path, laser_dir, poses_file, extrinsics_dir, camera_model, time_span, out_dir_pcd)
            print(f"Finished finished building point clouds for scene {scene} mode {mode}.")
