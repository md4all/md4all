import json
import os
import time
import argparse

from robotcar_dataset_sdk.interpolate_poses import interpolate_ins_poses, interpolate_vo_poses
from functools import partial
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def write_pose_to_json(fpath, filtered_samples):
    print(f"Writing pose information to Json: {fpath}.")

    with open(fpath, 'w') as f:
        f.write(json.dumps(filtered_samples, indent=0))
    f.close()

    print(f"Finished writing JSON file.")


def interpolate_and_store_poses(timestamp, dataroot, poses_subdir):
    pose_file = os.path.join(dataroot, scene, poses_subdir)
    if "ins" in poses_subdir:
        pose0m1 = interpolate_ins_poses(pose_file, [timestamp[-1]], timestamp[0], use_rtk=False)[0]
        pose0p1 = interpolate_ins_poses(pose_file, [timestamp[1]], timestamp[0], use_rtk=False)[0]
    elif "rtk" in poses_subdir:
        pose0m1 = interpolate_ins_poses(pose_file, [timestamp[-1]], timestamp[0], use_rtk=True)[0]
        pose0p1 = interpolate_ins_poses(pose_file, [timestamp[1]], timestamp[0], use_rtk=True)[0]
    elif "vo" in poses_subdir:
        pose0m1 = interpolate_vo_poses(pose_file, [timestamp[-1]], timestamp[0])[0]
        pose0p1 = interpolate_vo_poses(pose_file, [timestamp[1]], timestamp[0])[0]
    else:
        raise NotImplementedError("This pose type is not supported.")

    time.sleep(0.001)  # to visualize the progress

    return {"timestamp": timestamp[0], "pose_to_prev": pose0m1.tolist(), "pose_to_next": pose0p1.tolist()}


def precompute_and_store_poses(timestamps_path, dataroot, poses_subdir):
    assert 'test' not in timestamps_path, "Pose interpolation for test is not supported as the split file does not contain reference samples."
    interpolate_and_store_poses_partial = partial(interpolate_and_store_poses, dataroot=dataroot, poses_subdir=poses_subdir)
    with open(timestamps_path) as timestamp_file:
        timestamps = []
        for ts in timestamp_file.readlines():
            ts_split = ts.split(" ")
            timestamps.append({-1: int(ts_split[0]), 0: int(ts_split[1]), 1: int(ts_split[2])})

        with ProcessPoolExecutor(max_workers=20) as executor:
            results = list(tqdm(executor.map(interpolate_and_store_poses_partial, timestamps), total=len(timestamps)))
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose interpolation')
    parser.add_argument('--dataroot', type=str, help='Dataroot of robotcar dataset', required=True)
    parser.add_argument('--scenes', nargs='+', help='Scenes for pose interpolation', required=True)
    parser.add_argument('--modes', nargs='+', help='Modes for pose interpolation', required=True)
    parser.add_argument('--poses_subdir', type=str, help='Subdirectory of poses', required=True)

    args = parser.parse_args()
    dataroot = args.dataroot
    scenes = args.scenes
    modes = args.modes
    poses_subdir = args.poses_subdir
    print(f"Starting pose interpolation with scenes {scenes} and modes {modes}")

    precomputed_poses = []

    for scene in scenes:
        for mode in modes:
            print(f"Computing poses for scene {scene} mode {mode}...")
            timestamps_path = os.path.join(dataroot, f"splits/{scene}_{mode}_stride_1.txt")
            precomputed_poses += precompute_and_store_poses(timestamps_path, dataroot, poses_subdir)
            print(f"Finished computing poses for scene {scene} mode {mode}...")

            out_file = os.path.join(dataroot, scene, f"poses_synchronized_{mode}_{os.path.splitext(os.path.basename(poses_subdir))[0]}.json")
            write_pose_to_json(out_file, precomputed_poses)
            precomputed_poses.clear()
