"""
Recommended to use a copy of the h5 files instead of modifying the originals
"""
import os
import h5py
import numpy as np
import argparse
import cv2
import pickle
import concurrent.futures


def load_mask(mask_path):
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read mask image: {mask_path}")
    return img


def load_pose(path):
    with open(path, 'r') as f:
        return np.array([float(x) for x in f.read().strip().split()]).reshape(4, 4)


def get_args():
    parser = argparse.ArgumentParser(description="Update h5 files with robot and camera poses.")
    parser.add_argument('--h5_dir', type=str, required=True, help='Directory containing h5 files.')
    parser.add_argument('--foundationPose_dir', type=str, required=True, help='Directory containing FoundationPose data.')
    parser.add_argument('--obj', type=str, required=True, help='Object identifier (e.g., "block", "cube").')
    return parser.parse_args()

def main():
    pose_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    mask_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    args = get_args()
    obj = args.obj
    print(f"Processing object: {obj}")
    # h5_dir = f"/data3/stack_three_{obj}_real_with_poses/"
    h5_dir = args.h5_dir
    foundationPose_dir = args.foundationPose_dir
    print(f"H5 Directory: {h5_dir}")
    print(f"FoundationPose Directory: {foundationPose_dir}")
    print()

    # List all h5 files in the directory
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    for h5_filename in h5_files:
        h5_path = os.path.join(h5_dir, h5_filename)
        print(f"Processing h5 file: {h5_filename}")
        # Open the h5 file
        with h5py.File(h5_path, 'r+') as h5_file:

            # get the demo number
            demo_num = h5_filename.split('_')[1].split('.')[0]
            fpose_subdir = os.path.join(foundationPose_dir, f"demonstration_stack_three_{obj}_{demo_num}")
            # get all the output subdirs (beginning with output)
            output_subdirs = [d for d in os.listdir(fpose_subdir) if d.startswith('output')] + ["outputs_Franka_robot_arm"]
            obj_prompts = [d[len('output_'):] for d in output_subdirs] + ["_Franka_robot_arm"]
            if not output_subdirs:
                raise FileNotFoundError(f"No output subdirectories found in {fpose_subdir}")
            # in each output subdir, load all 0000_robot2block.txt and 0000.txt
            print(f"Found {output_subdirs} output subdirectories.")
            cam_idxs = ["side_cam", "front_cam", "overhead_cam"]
            masks = {
                "front_cam": {prompt: [] for prompt in obj_prompts},
                "overhead_cam": {prompt: [] for prompt in obj_prompts},
                "side_cam": {prompt: [] for prompt in obj_prompts},
            }
            obj_poses = {
                "robot_frame": {prompt: [] for prompt in obj_prompts},
                "cam_frame": {prompt: [] for prompt in obj_prompts},
                }
            obj_3D_bboxes = {
                "robot_frame": {prompt: [] for prompt in obj_prompts},
                "cam_frame": {prompt: [] for prompt in obj_prompts},
                }

            n_steps = len(os.listdir(os.path.join(fpose_subdir, output_subdirs[0]))) // 4
            for output_subdir, obj_prompt in zip(output_subdirs, obj_prompts):
                print(f"    Processing output subdirectory: {output_subdir}...")
                # count the number of files
                output_dir = os.path.join(fpose_subdir, output_subdir)
                # obj_prompt = output_subdir[len('output_'):]

                for i in range(n_steps):
                    if "Franka" not in output_subdir:
                        robot_pose_path = os.path.join(output_dir, f"{i:04d}_robot2block.txt")
                        cam_pose_path = os.path.join(output_dir, f"{i:04d}.txt")

                        # 2 rows, 4 columns (2x homogeneous points)
                        bbox_3D_cam_frame = os.path.join(output_dir, f"{i:04d}_3d_bbox_cam_frame.txt")
                        bbox_3D_robot_frame = os.path.join(output_dir, f"{i:04d}_3d_bbox_robot_frame.txt")

                        robot_pose, cam_pose, bbox_3D_cam_frame_data, bbox_3D_robot_frame_data = pose_executor.map(
                            load_pose,
                            [robot_pose_path, cam_pose_path, bbox_3D_cam_frame, bbox_3D_robot_frame]
                        )

                        obj_poses["robot_frame"][obj_prompt].append(robot_pose)
                        obj_poses["cam_frame"][obj_prompt].append(cam_pose)
                        obj_3D_bboxes["cam_frame"][obj_prompt].append(bbox_3D_cam_frame_data)
                        obj_3D_bboxes["robot_frame"][obj_prompt].append(bbox_3D_robot_frame_data)

                    mask_paths = [
                        os.path.join(fpose_subdir, f"rgb_cam_{cam}_masks{obj_prompt}", f"{i:04d}.png")
                        for cam in range(3)
                    ]

                    mask_imgs = list(mask_executor.map(load_mask, mask_paths))

                    for cam, mask_img in enumerate(mask_imgs):
                        masks[cam_idxs[cam]][obj_prompt].append(mask_img)


            if "2D_bboxes" in h5_file: del h5_file["2D_bboxes"]
            if "masks" in h5_file: del h5_file["masks"]
            if "obj_poses" in h5_file: del h5_file["obj_poses"]
            if "obj_3D_bboxes" in h5_file: del h5_file["obj_3D_bboxes"]
            # load boxes.pkl
            pkl_file = os.path.join(fpose_subdir, "boxes.pkl")
            if not os.path.exists(pkl_file):
                raise FileNotFoundError(f"Boxes file not found: {pkl_file}")
            with open(pkl_file, 'rb') as f:
                boxes = pickle.load(f)

            # save groups 2D_bboxes, cam, then array
            # create a group for 2D_bboxes
            h5_file.create_group("2D_bboxes")
            for cam_key in boxes:
                cam_num = int(cam_key.split("_")[-1])
                # create a group for each camera
                cam_group = h5_file["2D_bboxes"].create_group(cam_idxs[cam_num])
                for prompt, bboxes in boxes[cam_key].items():
                    # create a dataset for each prompt
                    cam_group.create_dataset("_" + prompt.replace(' ', '_'), data=np.stack(bboxes), compression='lzf')

            # Save the data to the h5 file
            h5_file.create_group("obj_poses")
            for pose_type in obj_poses:
                # create nested group for each pose type
                h5_file["obj_poses"].create_group(pose_type)
                for prompt, poses in obj_poses[pose_type].items():
                    if "Franka" in prompt:
                        continue
                    h5_file["obj_poses"][pose_type].create_dataset(prompt, data=np.stack(poses), compression='lzf')

            h5_file.create_group("obj_3D_bboxes")
            for bbox_type in obj_3D_bboxes:
                h5_file["obj_3D_bboxes"].create_group(bbox_type)
                for prompt, bboxes in obj_3D_bboxes[bbox_type].items():
                    if "Franka" in prompt:
                        continue
                    h5_file["obj_3D_bboxes"][bbox_type].create_dataset(prompt, data=np.stack(bboxes), compression='lzf')

            h5_file.create_group("masks")
            for cam in masks:
                # create a nested group for each camera
                cam_group = h5_file["masks"].create_group(cam)
                for prompt, masks_list in masks[cam].items():
                    cam_group.create_dataset(prompt, data=np.stack(masks_list), compression='lzf')

            h5_file.flush()


        print(f"Finished processing h5 file: {h5_filename}")
    pose_executor.shutdown()
    mask_executor.shutdown()

if __name__ == "__main__":
    main()





