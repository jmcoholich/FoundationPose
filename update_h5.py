"""
Recommended to use a copy of the h5 files instead of modifying the originals
"""
import os
import h5py
import numpy as np
import argparse

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
            if not output_subdirs:
                raise FileNotFoundError(f"No output subdirectories found in {fpose_subdir}")
            # in each output subdir, load all 0000_robot2block.txt and 0000.txt
            print(f"Found {output_subdirs} output subdirectories.")

            for output_subdir in output_subdirs:
                print(f"    Processing output subdirectory: {output_subdir}...")
                # count the number of files
                output_dir = os.path.join(fpose_subdir, output_subdir)
                obj_prompt = output_subdir[len('output_'):]
                x = os.listdir(output_dir)
                n_steps = len(x) // 4
                robot_poses = []
                cam_poses = []
                for i in range(n_steps):
                    robot_pose_path = os.path.join(output_dir, f"{i:04d}_robot2block.txt")
                    cam_pose_path = os.path.join(output_dir, f"{i:04d}.txt")

                    # 2 rows, 4 columns (2x homogeneous points)
                    bbox_3D_cam_frame = os.path.join(output_dir, f"{i:04d}_3d_bbox_cam_frame.txt")
                    bbox_3D_robot_frame = os.path.join(output_dir, f"{i:04d}_3d_bbox_robot_frame.txt")

                    # Load robot pose
                    with open(robot_pose_path, 'r') as f:
                        robot_pose = np.array([float(x) for x in f.read().strip().split()]).reshape(4, 4)
                        robot_poses.append(robot_pose)

                    # Load camera pose
                    with open(cam_pose_path, 'r') as f:
                        cam_pose = np.array([float(x) for x in f.read().strip().split()]).reshape(4, 4)
                        cam_poses.append(cam_pose)

                    with open(bbox_3D_cam_frame, 'r') as f:
                        bbox_3D_cam_frame_data = np.array([float(x) for x in f.read().strip().split()]).reshape(2, 4)
                        # print(f"bbox_3D_cam_frame_data: {bbox_3D_cam_frame_data}")

                    with open(bbox_3D_robot_frame, 'r') as f:
                        bbox_3D_robot_frame_data = np.array([float(x) for x in f.read().strip().split()]).reshape(2, 4)
                        # print(f"bbox_3D_robot_frame_data: {bbox_3D_robot_frame_data}")


                rkey = f"pose_robot_frame_{obj_prompt}"
                ckey = f"pose_cam_frame_{obj_prompt}"

                # rkey_old = f"robot_poses_{obj_prompt}"
                # ckey_old = f"cam_poses_{obj_prompt}"
                bbox_3D_cam_frame_key = f"bbox_3D_cam_frame_{obj_prompt}"
                bbox_3D_robot_frame_key = f"bbox_3D_robot_frame_{obj_prompt}"
                if rkey in h5_file: del h5_file[rkey]
                if ckey in h5_file: del h5_file[ckey]
                if bbox_3D_cam_frame_key in h5_file: del h5_file[bbox_3D_cam_frame_key]
                if bbox_3D_robot_frame_key in h5_file: del h5_file[bbox_3D_robot_frame_key]
                # if rkey_old in h5_file: del h5_file[rkey_old]
                # if ckey_old in h5_file: del h5_file[ckey_old]

                # h5_file[rkey] = np.array(robot_poses)
                # h5_file[ckey] = np.array(cam_poses)
                h5_file.create_dataset(rkey, data=np.array(robot_poses), compression='gzip', compression_opts=9)
                h5_file.create_dataset(ckey, data=np.array(cam_poses), compression='gzip', compression_opts=9)
                h5_file.create_dataset(bbox_3D_cam_frame_key, data=bbox_3D_cam_frame_data, compression='gzip', compression_opts=9)
                h5_file.create_dataset(bbox_3D_robot_frame_key, data=bbox_3D_robot_frame_data, compression='gzip', compression_opts=9)
                h5_file.flush()
        print(f"Finished processing h5 file: {h5_filename}")

if __name__ == "__main__":
    main()





