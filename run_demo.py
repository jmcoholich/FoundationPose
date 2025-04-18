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
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R


def main():
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--mask_dir', type=str, default=f'{code_dir}/demo_data/mustard0/masks')
    parser.add_argument('--init_rot_guess', type=ast.literal_eval, default='[[1, 0, 0], [0, 1, 0], [0, 0, 1]]')
    parser.add_argument('--map_to_table_frame', action='store_true', help='whether to map the object pose to the table frame')

    args = parser.parse_args()

    args.mask_dir = args.mask_dir.replace(' ', '_')
    args.debug_dir = args.debug_dir.replace(' ', '_')
    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if i==0:
            mask = reader.get_mask(0, dirname=args.mask_dir).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter, init_rot_guess=args.init_rot_guess)
            print("ASDF")
            if args.map_to_table_frame:
                detections = get_april_tag(color, reader)

            if debug>=3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth>=0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))


        if debug>=1:
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            if args.map_to_table_frame:
                vis = vis_tag(vis, [detections["cam_tag"]["detection"]])
            cv2.imshow('1', vis[...,::-1])
            cv2.waitKey(1)


        if debug>=2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)


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
                    raise ValueError(f"No April Tags detected for tag {name}")
                else:
                    props["detection"] = None
        else:
            raise ValueError(f"No April Tags detected for tag {name}")
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


if __name__ == "__main__":
    main()
