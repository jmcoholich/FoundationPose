"""
Recommended to use a copy of the h5 files instead of modifying the originals
"""
import os
import h5py
import numpy as np
import argparse
import cv2


def get_args():
    parser = argparse.ArgumentParser(description="Update h5 files with robot and camera poses.")
    parser.add_argument('--h5_dir', type=str, required=True, help='Directory containing h5 files.')
    parser.add_argument('--foundationPose_dir', type=str, required=True, help='Directory containing FoundationPose data.')
    parser.add_argument('--obj', type=str, required=True, help='Object identifier (e.g., "block", "cube").')
    return parser.parse_args()


def main():
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
            output_subdirs = [d for d in os.listdir(fpose_subdir) if d.startswith('output')]
            obj_prompts = [d[len('output_'):] for d in output_subdirs]
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

            for output_subdir in output_subdirs:
                print(f"    Processing output subdirectory: {output_subdir}...")
                # count the number of files
                output_dir = os.path.join(fpose_subdir, output_subdir)
                obj_prompt = output_subdir[len('output_'):]
                x = os.listdir(output_dir)
                n_steps = len(x) // 4

                for i in range(n_steps):
                    robot_pose_path = os.path.join(output_dir, f"{i:04d}_robot2block.txt")
                    cam_pose_path = os.path.join(output_dir, f"{i:04d}.txt")

                    # 2 rows, 4 columns (2x homogeneous points)
                    bbox_3D_cam_frame = os.path.join(output_dir, f"{i:04d}_3d_bbox_cam_frame.txt")
                    bbox_3D_robot_frame = os.path.join(output_dir, f"{i:04d}_3d_bbox_robot_frame.txt")

                    # Load robot pose
                    with open(robot_pose_path, 'r') as f:
                        robot_pose = np.array([float(x) for x in f.read().strip().split()]).reshape(4, 4)
                        # robot_poses.append(robot_pose)
                        obj_poses["robot_frame"][obj_prompt].append(robot_pose)

                    # Load camera pose
                    with open(cam_pose_path, 'r') as f:
                        cam_pose = np.array([float(x) for x in f.read().strip().split()]).reshape(4, 4)
                        # cam_poses.append(cam_pose)
                        obj_poses["cam_frame"][obj_prompt].append(cam_pose)

                    with open(bbox_3D_cam_frame, 'r') as f:
                        bbox_3D_cam_frame_data = np.array([float(x) for x in f.read().strip().split()]).reshape(2, 4)
                        # bboxes_3D_cam_frame.append(bbox_3D_cam_frame_data)
                        obj_3D_bboxes["cam_frame"][obj_prompt].append(bbox_3D_cam_frame_data)

                    with open(bbox_3D_robot_frame, 'r') as f:
                        bbox_3D_robot_frame_data = np.array([float(x) for x in f.read().strip().split()]).reshape(2, 4)
                        # bboxes_3D_robot_frame.append(bbox_3D_robot_frame_data)
                        obj_3D_bboxes["robot_frame"][obj_prompt].append(bbox_3D_robot_frame_data)

                    # masks_at_t = []
                    for cam in range(3):
                        cam_dir = f"rgb_cam_{cam}_masks{obj_prompt}"
                        mask_path = os.path.join(fpose_subdir, cam_dir, f"{i:04d}.png")
                        if not os.path.exists(mask_path):
                            raise FileNotFoundError(f"Mask file not found: {mask_path}")
                        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask_img is None:
                            raise ValueError(f"Failed to read mask image: {mask_path}")
                        # Ensure mask is binary (0 or 255)
                        assert np.array_equal(np.unique(mask_img), [0, 255]), f"Mask image is not binary: {mask_path}"
                        masks[cam_idxs[cam]][obj_prompt].append(mask_img)
                        # masks_at_t.append(mask_img)
                    # masks.append(masks_at_t)

            if "masks" in h5_file: del h5_file["masks"]
            if "obj_poses" in h5_file: del h5_file["obj_poses"]
            if "obj_3D_bboxes" in h5_file: del h5_file["obj_3D_bboxes"]
            # Save the data to the h5 file
            h5_file.create_group("obj_poses")
            for pose_type in obj_poses:
                # create nested group for each pose type
                h5_file["obj_poses"].create_group(pose_type)
                for prompt, poses in obj_poses[pose_type].items():
                    h5_file["obj_poses"][pose_type].create_dataset(prompt, data=np.array(poses), compression='gzip', compression_opts=9)

            h5_file.create_group("obj_3D_bboxes")
            for bbox_type in obj_3D_bboxes:
                h5_file["obj_3D_bboxes"].create_group(bbox_type)
                for prompt, bboxes in obj_3D_bboxes[bbox_type].items():
                    h5_file["obj_3D_bboxes"][bbox_type].create_dataset(prompt, data=np.array(bboxes), compression='gzip', compression_opts=9)

            h5_file.create_group("masks")
            for cam in masks:
                # create a nested group for each camera
                cam_group = h5_file["masks"].create_group(cam)
                for prompt, masks_list in masks[cam].items():
                    cam_group.create_dataset(prompt, data=np.array(masks_list), compression='gzip', compression_opts=9)

            h5_file.flush()


        print(f"Finished processing h5 file: {h5_filename}")

if __name__ == "__main__":
    main()





