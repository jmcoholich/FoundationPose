#!/usr/bin/env python3
import argparse
import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageFile
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Take a demo.pkl file and extract depth, rgb, mesh, masks, and camera intrinsics for Foundation Pose."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the input dir containing .pkl demo file"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where output subfolders (depth, rgb, mesh) will be created"
    )
    parser.add_argument(
        "--cam_num",
        type=int,
        default=2,
        help="Camera index to extract (default: 2)"
    )
    parser.add_argument(
        "--cam_K_matrix",
        default="912.0 0.0 360.0\n0.0 912.0 360.0\n0.0 0.0 1\n",
        help="Camera intrinsics matrix contents (will be written to cam_K.txt)"
    )
    parser.add_argument(
        "--mesh_file",
        required=True,
        help="Path to the mesh file to copy into the mesh folder"
    )
    args = parser.parse_args()

    # Locate the single .pkl file
    pkl_files = [f for f in os.listdir(args.input_dir) if f.endswith('.pkl')]
    if len(pkl_files) != 1:
        raise ValueError(f"Expected exactly one .pkl file in {args.input_dir}, found {len(pkl_files)}")
    pkl_demo_file = os.path.join(args.input_dir, pkl_files[0])
    print(f"Loading demo from {pkl_demo_file}")

    # Load the demo
    with open(pkl_demo_file, 'rb') as f:
        demo = pickle.load(f)
    n_frames = len(demo['depth_frames'])

    # Prepare output directories
    depth_dir = os.path.join(args.output_dir, 'depth')
    rgb_dir   = os.path.join(args.output_dir, 'rgb')
    mesh_dir  = os.path.join(args.output_dir, 'mesh')
    for d in (depth_dir, rgb_dir, mesh_dir):
        os.makedirs(d, exist_ok=True)

    # Allow Pillow to handle large PNG blocks
    ImageFile.MAXBLOCK = 2**20

    # Worker function to save one frame (depth + rgb)
    def save_frame(i):
        # Depth: crop to selected camera
        depth = demo['depth_frames'][i, args.cam_num]
        Image.fromarray(depth).save(
            os.path.join(depth_dir, f"{i:04d}.png"),
            compress_level=1
        )
        # RGB: convert BGR->RGB
        rgb = demo['rgb_frames'][i, args.cam_num, ..., ::-1]
        Image.fromarray(rgb).save(
            os.path.join(rgb_dir, f"{i:04d}.png"),
            compress_level=1
        )

    # Parallelize all the saves with a ThreadPool
    max_workers = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(save_frame, i) for i in range(n_frames)]
        for _ in tqdm(as_completed(futures), total=n_frames, desc="Extracting frames"):
            pass

    # Copy mesh and write camera intrinsics
    shutil.copy(args.mesh_file, os.path.join(mesh_dir, os.path.basename(args.mesh_file)))
    with open(os.path.join(args.output_dir, 'cam_K.txt'), 'w') as f:
        f.write(args.cam_K_matrix)

    print(f"Finished extracting to {args.output_dir}")


if __name__ == "__main__":
    main()
