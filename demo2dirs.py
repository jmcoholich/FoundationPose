#!/usr/bin/env python3
import argparse
import os
import pickle
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Take a demo.pkl file and extract depth, rgb, mesh, masks, and camera intrinsics for Foundation Pose.")
    parser.add_argument(
        "--input_dir",
        help="Path to the input dir containing .pkl demo file")
    parser.add_argument(
        "--output_dir",
        help="Directory where output subfolders (depth, rgb, masks, mesh) will be created")
    parser.add_argument(
        "--cam_num",
        type=int,
        help="Camera index to extract (default: 2)")
    parser.add_argument(
        "--cam_K_matrix",
        default="912.0 0.0 360.0\n0.0 912.0 360.0\n0.0 0.0 1\n",
        help="Path to the camera intrinsics txt (will be copied as cam_K.txt)")
    parser.add_argument(
        "--mesh_file",
        help="Path to the mesh file to copy into the mesh folder")
    args = parser.parse_args()

    # pkl demo file should be the only pkl file in the input dir
    pkl_files = [f for f in os.listdir(args.input_dir) if f.endswith('.pkl')]
    if len(pkl_files) != 1:
        raise ValueError(f"Expected exactly one .pkl file in {args.input_dir}, found {len(pkl_files)}")
    pkl_demo_file = os.path.join(args.input_dir, pkl_files[0])
    print(f"Loading demo from {pkl_demo_file}")

    # load demo
    with open(pkl_demo_file, 'rb') as f:
        demo = pickle.load(f)
    n_frames = len(demo['depth_frames'])

    # create output dirs
    depth_dir = os.path.join(args.output_dir, 'depth')
    rgb_dir   = os.path.join(args.output_dir, 'rgb')
    mesh_dir  = os.path.join(args.output_dir, 'mesh')
    for d in (depth_dir, rgb_dir, mesh_dir):
        os.makedirs(d, exist_ok=True)

    # extract frames
    for i in tqdm(range(n_frames), desc="Extracting frames"):
        # depth: crop to 720×720 region
        depth_frame = demo['depth_frames'][i, args.cam_num, :, 280:1000]
        Image.fromarray(depth_frame).save(
            os.path.join(depth_dir, f"{i:04d}.png"))

        # rgb: crop + BGR→RGB
        rgb_frame = demo['rgb_frames'][i, args.cam_num, :, 280:1000, ::-1]
        Image.fromarray(rgb_frame).save(
            os.path.join(rgb_dir, f"{i:04d}.png"))

    # copy mesh and camera intrinsics
    shutil.copy(args.mesh_file, os.path.join(mesh_dir, os.path.basename(args.mesh_file)))
    with open(os.path.join(args.output_dir, 'cam_K.txt'), 'w') as f:
        f.write(args.cam_K_matrix)
    print(f"Finished extracting to {args.output_dir}")

if __name__ == "__main__":
    main()
