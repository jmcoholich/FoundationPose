# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
import ast
from pupil_apriltags import Detector, Detection
from scipy.spatial.transform import Rotation as R
import os
import numpy as np
from Utils import to_homo

CAM_TAG_DEFAULT_DETECTION = Detection()
CAM_TAG_DEFAULT_DETECTION.tag_family = b'tagStandard41h12'
CAM_TAG_DEFAULT_DETECTION.tag_id = 0
CAM_TAG_DEFAULT_DETECTION.hamming = 0
CAM_TAG_DEFAULT_DETECTION.decision_margin = 0.0
CAM_TAG_DEFAULT_DETECTION.homography = np.array([[ 3.52548972e+01,  5.86007098e+00,  9.19357271e+02],
                                                  [-4.51931082e+00,  4.51964660e+01,  2.12557441e+02],
                                                  [-1.11048126e-02,  7.12507814e-03,  1.00000000e+00]])
CAM_TAG_DEFAULT_DETECTION.center = np.array([919.35727091, 212.55744096])
CAM_TAG_DEFAULT_DETECTION.corners = np.array([[874.0289917,  257.5776062],
                                                  [964.30993652, 254.24642944],
                                                  [966.36889648, 165.8653717],
                                                  [874.76098633, 171.19895935]])
CAM_TAG_DEFAULT_DETECTION.pose_R = np.array([[ 0.99489053,  0.04440697,  0.09066891],
                                                  [-0.03501397,  0.99409808, -0.10267929],
                                                  [-0.09469346,  0.09897998,  0.99057363]])
CAM_TAG_DEFAULT_DETECTION.pose_t = np.array([[ 0.64103056],
                                                  [-0.16901774],
                                                  [ 1.0445981]])
CAM_TAG_DEFAULT_DETECTION.pose_err = 0.0



def main():
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--cam_number', type=int)
    # parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    # parser.add_argument('--mask_dir', type=str, default=f'{code_dir}/demo_data/mustard0/masks')
    parser.add_argument('--prompts', type=str, nargs='+')
    parser.add_argument('--init_rot_guess', type=ast.literal_eval)
    parser.add_argument('--map_to_table_frame', action='store_true', help='whether to map the object pose to the table frame')
    parser.add_argument('--use_all_masks', action='store_true', help='condition on masks at every timestep')
    parser.add_argument('--headless', action='store_true', help='do not show the visualization, good for running on a server')

    args = parser.parse_args()

    # load annotations pkl file (pkl file with "annotations" in the name in test_scene_dir)
    annotations_file = None
    for f in os.listdir(args.test_scene_dir):
        if "annotations" in f and f.endswith('.pkl'):
            annotations_file = os.path.join(args.test_scene_dir, f)
            break
    cam_idxs = ["side_cam", "front_cam", "overhead_cam"]
    cam_name = cam_idxs[args.cam_number]
    if annotations_file is not None:
        # process annotations file, we only care about cam_number
        with open(annotations_file, "rb") as f:
            annotations = pickle.load(f)

    for i in range(len(args.prompts)):
        args.prompts[i] = args.prompts[i].replace(' ', '_')
    # breakpoint()
    # args.mask_dir = args.mask_dir.replace(' ', '_')
    # args.debug_dir = args.debug_dir.replace(' ', '_')
    output_dirs = [os.path.join(args.test_scene_dir, "outputs_" + prompt) for prompt in args.prompts]
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)
    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)

    debug = args.debug
    # debug_dir = args.debug_dir
    # os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    # to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    # I want to preserve the mesh axes for downstream transforms
    centroid = mesh.centroid
    to_origin = np.eye(4)
    to_origin[:3, 3] = -centroid
    extents = mesh.extents
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    bbox_homo = to_homo(np.stack([extents/2, extents/2, extents/2, extents/2], axis=0).reshape(4,3))
    for i in range(3):
        bbox_homo[i, i] *= -1

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    ests = [FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner,  debug=debug, glctx=glctx) for _ in range(len(args.prompts))]
    logging.info("estimator initialization done")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, cam_num=args.cam_number, shorter_side=None, zfar=np.inf)
    stopped_tracking = [False, False, False]
    out_of_frame = [False, False, False]
    for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if i==0:
            poses = []
            for j in range(len(ests)):
                mask = reader.get_mask(0, dirname="_masks_" + args.prompts[j]).astype(bool)
                poses.append(ests[j].register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter, init_rot_guess=args.init_rot_guess))
            if args.map_to_table_frame:
                detections = get_april_tag(color, reader)
                cam2tag = np.eye(4)
                cam2tag[:3, :3] = detections["cam_tag"]["detection"].pose_R
                cam2tag[:3, 3] = detections["cam_tag"]["detection"].pose_t.reshape(3)

            if debug>=3:
                raise NotImplementedError("debug>=3 not implemented with multiple objects")
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth>=0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
            poses = []
            for j in range(len(ests)):
                img_key = f"{i}_{cam_name}"
                obj_key = args.prompts[j].replace("_", " ")
                if annotations_file is not None and img_key in annotations and obj_key in annotations[img_key]:
                    print(annotations[img_key])
                    val = annotations[img_key][obj_key]
                    if val == "stop_tracking":
                        print(f"Stopping tracking for {args.prompts[j]} at frame {i}")
                        stopped_tracking[j] = True
                    elif val == "out_of_frame":
                        print(f"Object {args.prompts[j]} is out of frame at frame {i}")
                        out_of_frame[j] = True
                    else:
                        # its a bounding box, reinit tracking w/ mask
                        mask = reader.get_mask(i, dirname="_masks_" + args.prompts[j]).astype(bool)
                        print("="*20 + "\nusing mask\n" + "="*20)
                        poses.append(ests[j].register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter, init_rot_guess=args.init_rot_guess))
                        out_of_frame[j] = False
                        continue
                if stopped_tracking[j]:
                    poses.append(last_poses[j])
                    continue
                if out_of_frame[j]:
                    poses.append(np.zeros_like(last_poses[j]))
                    continue
                if args.use_all_masks:
                    mask = reader.get_mask(i, dirname="_masks_" + args.prompts[j]).astype(bool)
                    print("="*20 + "\nusing mask\n" + "="*20)
                    poses.append(ests[j].register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter, init_rot_guess=args.init_rot_guess))
                else:
                    print("@"*20 + "\nno mask\n" + "@"*20)
                    poses.append(ests[j].track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter))
        last_poses = poses.copy()
        transforms = {}
        for j in range(len(ests)):
            os.makedirs(f'{output_dirs[j]}', exist_ok=True)
            cam2block = poses[j].reshape(4,4)
            np.savetxt(f'{output_dirs[j]}/{reader.id_strs[i]}.txt', cam2block)

            # also save 3D bounding box of the detected objects
            center_pose = cam2block @ np.linalg.inv(to_origin)
            bbox_cam_frame = (center_pose @ bbox_homo.T).T
            np.savetxt(f'{output_dirs[j]}/{reader.id_strs[i]}_3d_bbox_cam_frame.txt', bbox_cam_frame)

            if args.map_to_table_frame:
                robot2block = ROBO2TAG @ np.linalg.inv(cam2tag) @ cam2block
                np.savetxt(f'{output_dirs[j]}/{reader.id_strs[i]}_robot2block.txt', robot2block.reshape(4,4))

                bbox_robot_frame = (ROBO2TAG @ np.linalg.inv(cam2tag) @ bbox_cam_frame.T).T
                np.savetxt(f'{output_dirs[j]}/{reader.id_strs[i]}_3d_bbox_robot_frame.txt', bbox_cam_frame)

                transforms[args.prompts[j]] = robot2block
                # print(f"translation: {translation}")

        if debug>=1:
            vis = color.copy()
            if args.map_to_table_frame:
                add_translation_text(vis, transforms, [(0, 50), (0, 100), (0, 150)], i)
            for j in range(len(ests)):
                if out_of_frame[j]:
                    continue
                center_pose = poses[j]@np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(reader.K, img=vis, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.05, K=reader.K, thickness=2, transparency=0, is_input_rgb=True)
                if args.map_to_table_frame:
                    vis = vis_tag(vis, [detections["cam_tag"]["detection"]])
            if not args.headless:
                cv2.imshow('1', vis[...,::-1])
                cv2.waitKey(1)


        if debug>=2:
            os.makedirs(f'{args.test_scene_dir}/track_vis', exist_ok=True)
            path = f'{args.test_scene_dir}/track_vis/{reader.id_strs[i]}.png'
            # imageio.imwrite(path, vis)
            cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))  # if vis is RGB


def add_translation_text(vis, translations, locations, frame_index):
    # write the key, value of the translations on the image at the location
    for i, (key, value) in enumerate(translations.items()):
        # translations
        value_str = f"{key + ' translation'}: {value[0, 3]:.2f}, {value[1, 3]:.2f}, {value[2, 3]:.2f}"
        cv2.putText(vis, value_str, locations[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # rotations, convert to zyx euler angles
        euler = R.from_matrix(value[:3, :3]).as_euler('zyx', degrees=True)
        value_str = f"{key + ' ZYX Euler'}: {euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}"
        locations[i] = (locations[i][0], locations[i][1] + 20)
        cv2.putText(vis, value_str, locations[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # add the frame index to the bottom right corner (4 digits filled in with zeros)
    frame_index_str = f"Frame: {frame_index:04d}"
    cv2.putText(vis, frame_index_str, (vis.shape[1] - 150, vis.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def get_april_tag(img, reader):
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cx = reader.K[0, 2]
    cy = reader.K[1, 2]
    fx = reader.K[0, 0]
    fy = reader.K[1, 1]
    w = reader.H
    h = reader.W
    at_detector = Detector(
        families="tagStandard41h12",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    tags = {
        "cam_tag": {"id": 0, "size": 0.099},
        # "coke_tag": {"id": 3, "size": 0.0134},
    }
    for name, props in tags.items():
        x = at_detector.detect(
            img,
            estimate_tag_pose=True,
            camera_params=[fx, fy, cx, cy],  # fx fy cx cy, all units are in pixels
            tag_size=props["size"],
            )
        if x:
            for r in x:
                if r.tag_id == props["id"]:
                    props["detection"] = r
                    break
            else:
                if name == "cam_tag":
                    # raise ValueError(f"No April Tags detected for tag {name}")
                    print(f"No April Tags detected for tag {name}, using default detection")
                    props["detection"] = CAM_TAG_DEFAULT_DETECTION
                else:
                    props["detection"] = None
        else:
            # raise ValueError(f"No April Tags detected for tag {name}")
            print(f"No April Tags detected for tag {name}, using default detection")
            props["detection"] = CAM_TAG_DEFAULT_DETECTION
    return tags
    # all_tags = [tags["cam_tag"]["detection"], tags["coke_tag"]["detection"]]
    # all_tags = [tags["cam_tag"]["detection"]]
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # img = vis_tag(img, all_tags)
    # cv2.line(img, (0, int(cy)), (w, int(cy)), (0, 255, 0), 2)
    # cv2.line(img, (int(cx), 0), (int(cx), h), (0, 255, 0), 2)
    # cv2.imwrite(os.path.join(sample_dir, 'april_tags.png'), img)

    # cam_tag_pose = np.eye(4)
    # cam_tag_pose[:3, :3] = tags["cam_tag"]["detection"].pose_R
    # cam_tag_pose[:3, 3] = tags["cam_tag"]["detection"].pose_t.reshape(3)

    # if tags["coke_tag"]["detection"] is None:
    #     coke_tag_pose = None
    # else:
    #     coke_tag_pose = np.eye(4)
    #     coke_tag_pose[:3, :3] = tags["coke_tag"]["detection"].pose_R
    #     coke_tag_pose[:3, 3] = tags["coke_tag"]["detection"].pose_t.reshape(3)

    # return cam_tag_pose, coke_tag_pose

def vis_tag(img: np.ndarray, results):
    img = img.astype(np.uint8)
    for r in results:
        if r is None:
            continue
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        # draw the bounding box of the AprilTag detection
        cv2.line(img, ptA, ptB, (0, 255, 0), 2)
        cv2.line(img, ptB, ptC, (0, 255, 0), 2)
        cv2.line(img, ptC, ptD, (0, 255, 0), 2)
        cv2.line(img, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(
            img,
            tagFamily,
            (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        print(r.tag_id)
        print(r.pose_t)
        # now convert R to XYZ euler angles
        r = R.from_matrix(r.pose_R)
        print("XYZ Euler angles (in degrees) are:")
        print(r.as_euler('xyz', degrees=True))

        print()
    return img

def get_robot2tag_mat():
        # The following transformations are to get to the tag frame from the robot frame
    # this is the robot mat without the 14 degree tilt or whatever
    robo_mat = np.array([[0, 0, 1, -0.30895138],
                            [0, -1, 0, 0],
                            [1, 0, 0, 0.82001764],
                            [0, 0, 0, 1]])

    # rotate 180 about world z-axis
    robo_mat[:3, :3] = np.array([[-1, 0, 0],
                                [0, -1, 0],
                                [0, 0, 1]]) @ robo_mat[:3, :3]

    tag_size = 0.099
    # Move to surface of robot base. aligned with front tip. 3 cm off the ground. 2.3 cm from the side
    robo_mat[:3, 3] += np.array([
        0.225 / 2.0 + 0.005,  # add width of clipboard and screws offset
        .508 / 2.0 -0.006 - tag_size / 5 + tag_size / 2.0,
        -0.14 / 2.0 + (tag_size * 9/5 / 2) + 0.043 - tag_size / 5,
        ])
    robo_mat[:3, :3] = robo_mat[:3, :3] @ np.array([[0, -1, 0],
                                                    [1, 0, 0],
                                                    [0, 0, 1]])

    # # rotate 180 about world z-axis
    # robo_mat[:3, :3] = np.array([[-1, 0, 0],
    #                             [0, -1, 0],
    #                             [0, 0, 1]]) @ robo_mat[:3, :3]
    return robo_mat

ROBO2TAG = get_robot2tag_mat()

if __name__ == "__main__":
    main()
