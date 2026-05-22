# yapf: disable
import argparse
import cv2
import datetime
import glob
import mmcv
import numpy as np
import os
from mmhuman3d.core.visualization.visualize_smpl import (
    visualize_smpl_calibration,
)
from mmhuman3d.utils.demo_utils import get_different_colors
from typing import List
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.utils.log_utils import setup_logger

from xrmocap.core.estimation.builder import build_estimator
# from xrmocap.core.visualization import visualize_project_keypoints3d
from xrmocap.visualization import visualize_keypoints3d_projected

# yapf: enable


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.enable_log_file:
        time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_path = os.path.join(args.output_dir, f'{time_str}.txt')
        logger = setup_logger(logger_name=__name__, logger_path=log_path)
    else:
        logger = None

    # build estimator
    estimator_config = dict(mmcv.Config.fromfile(args.estimator_config))
    estimator_config['logger'] = logger
    smpl_estimator = build_estimator(estimator_config)
    # load camera parameter and images
    image_dir = []
    fisheye_param_paths = []
    with open(args.image_and_camera_param, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if i % 2 == 0:
                image_dir.append(line)
            else:
                fisheye_param_paths.append(line)
    fisheye_params = load_camera_parameters(fisheye_param_paths)
    mview_img_list = []
    for idx in range(len(fisheye_params)):
        sview_img_list = sorted(
            glob.glob(os.path.join(image_dir[idx], '*.png')) +
            glob.glob(os.path.join(image_dir[idx], '*.jpg')) +
            glob.glob(os.path.join(image_dir[idx], '*.jpeg')))
        mview_img_list.append(sview_img_list)

    pred_keypoints3d, smpl_data_list = smpl_estimator.run(
        cam_param=fisheye_params,
        img_paths=mview_img_list,
        save_perception_2d=args.save_perception_2d,
        perception_2d_dir=args.output_dir,
        save_association=args.save_association,
        association_dir=args.output_dir)
    npz_path = os.path.join(args.output_dir, 'pred_keypoints3d.npz')
    pred_keypoints3d.dump(npz_path)
    # for i, smpl_data in enumerate(smpl_data_list):
    #     smpl_data.dump(os.path.join(args.output_dir, f'smpl_{i}.npz'))

    # Visualization (SMPL skipped when smpl_data_list is empty)
    if not args.disable_visualization and len(smpl_data_list) > 0:
        n_frame = len(mview_img_list[0])
        n_person = len(smpl_data_list)
        colors = get_different_colors(n_person)
        tmp = colors[:, 0].copy()
        colors[:, 0] = colors[:, 2]
        colors[:, 2] = tmp
        full_pose_list = []
        transl_list = []
        betas_list = []
        for smpl_data in smpl_data_list:
            # Apply frame interval sampling to SMPL data to match n_frame
            fullpose_data = smpl_data['fullpose']
            transl_data = smpl_data['transl'] 
            betas_data = smpl_data['betas']
            
            # if args.frame_interval > 1:
            #     fullpose_data = fullpose_data[::args.frame_interval]
            #     transl_data = transl_data[::args.frame_interval]
            #     betas_data = betas_data[::args.frame_interval]
                
            full_pose_list.append(fullpose_data[:, np.newaxis])
            transl_list.append(transl_data[:, np.newaxis])
            betas_list.append(betas_data[:, np.newaxis])
        fullpose = np.concatenate(full_pose_list, axis=1)
        transl = np.concatenate(transl_list, axis=1)
        betas = np.concatenate(betas_list, axis=1)

        body_model_cfg = dict(
            type='SMPL',
            gender='neutral',
            num_betas=10,
            keypoint_src='smpl_45',
            keypoint_dst='smpl',
            model_path='xrmocap_data/body_models',
            batch_size=1)
        # prepare camera
        for idx, fisheye_param in enumerate(fisheye_params):
            k_np = np.array(fisheye_param.get_intrinsic(3))
            r_np = np.array(fisheye_param.get_extrinsic_r())
            t_np = np.array(fisheye_param.get_extrinsic_t())
            cam_name = fisheye_param.name
            view_name = cam_name.replace('fisheye_param_', '')

            image_list = []
            for frame_path in mview_img_list[idx]:
                image_np = cv2.imread(frame_path)
                image_list.append(image_np)
            image_array = np.array(image_list)

            visualize_keypoints3d_projected(
                keypoints=pred_keypoints3d,
                camera=fisheye_param,
                output_path=os.path.join(args.output_dir, 'kps3d',
                                         f'project_view_{view_name}.mp4'),
                background_img_list=mview_img_list[idx],
                overwrite=True)

            # Handle fullpose reshape with proper dimension checking
            total_elements = fullpose.size
            print(f"Debug: fullpose.size={total_elements}, n_frame={n_frame}, n_person={n_person}")
            
            # Calculate expected dimensions based on SMPL pose format (72 params per person)
            pose_dim = 72  # Standard SMPL pose dimension
            
            # First, try to determine if this is the correct number of frames
            if total_elements % (n_frame * pose_dim) == 0:
                # Can be divided evenly with expected frames and standard pose dims
                actual_n_person = total_elements // (n_frame * pose_dim)
                print(f"Using n_frame={n_frame}, actual_n_person={actual_n_person}, pose_dim={pose_dim}")
                poses_reshaped = fullpose.reshape(n_frame, actual_n_person, pose_dim)
            elif total_elements % (pose_dim * n_person) == 0:
                # Can be divided evenly with expected persons and standard pose dims
                actual_n_frame = total_elements // (pose_dim * n_person)
                print(f"Using actual_n_frame={actual_n_frame}, n_person={n_person}, pose_dim={pose_dim}")
                poses_reshaped = fullpose.reshape(actual_n_frame, n_person, pose_dim)
            else:
                # Try different pose dimensions
                for test_pose_dim in [69, 72, 75, 156]:  # Common SMPL variants
                    if total_elements % (n_frame * test_pose_dim) == 0:
                        actual_n_person = total_elements // (n_frame * test_pose_dim)
                        print(f"Using n_frame={n_frame}, actual_n_person={actual_n_person}, pose_dim={test_pose_dim}")
                        poses_reshaped = fullpose.reshape(n_frame, actual_n_person, test_pose_dim)
                        break
                    elif total_elements % (n_person * test_pose_dim) == 0:
                        actual_n_frame = total_elements // (n_person * test_pose_dim)
                        print(f"Using actual_n_frame={actual_n_frame}, n_person={n_person}, pose_dim={test_pose_dim}")
                        poses_reshaped = fullpose.reshape(actual_n_frame, n_person, test_pose_dim)
                        break
                else:
                    # Last resort: calculate most likely dimensions
                    print(f"Warning: Cannot find standard SMPL dimensions. Total elements: {total_elements}")
                    if total_elements % n_frame == 0:
                        remaining = total_elements // n_frame
                        poses_reshaped = fullpose.reshape(n_frame, -1)
                        print(f"Reshaped to ({n_frame}, {remaining})")
                    else:
                        raise ValueError(f"Cannot reshape fullpose with size {total_elements} "
                                       f"for {n_frame} frames and {n_person} persons")

            visualize_smpl_calibration(
                poses=poses_reshaped,
                betas=betas,
                transl=transl,
                palette=colors,
                output_path=os.path.join(args.output_dir, 'smpl',
                                         f'{view_name}_smpl.mp4'),
                body_model_config=body_model_cfg,
                K=k_np,
                R=r_np,
                T=t_np,
                image_array=image_array,
                resolution=(image_array.shape[1], image_array.shape[2]),
                overwrite=True)


def load_camera_parameters(fisheye_param_paths: List[str]):
    """Load multi-scene fisheye parameters."""
    mview_list = []
    for path in fisheye_param_paths:
        fisheye_param = FisheyeCameraParameter.fromfile(path)
        if fisheye_param.world2cam:
            fisheye_param.inverse_extrinsic()
        mview_list.append(fisheye_param)

    return mview_list


def setup_parser():
    parser = argparse.ArgumentParser(
        description='MultiViewMultiPersonTopDownEstimator')
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory saving all possible output files.',
        default='./output/estimation')
    parser.add_argument(
        '--estimator_config',
        help='Config file for MultiViewMultiPersonTopDownEstimator.',
        type=str,
        default='configs/modules/core/estimation/'
        'mview_mperson_topdown_estimator.py')
    parser.add_argument(
        '--image_and_camera_param',
        help='A text file contains the image path and the corresponding'
        'camera parameters',
        default='./xrmocap_data/Shelf/image_and_camera_param.txt')
    # parser.add_argument(
    #     '--frame_interval', 
    #     type=int, 
    #     default=1,
    #     help='Frame sampling interval (default: 1, means every frame)')
    parser.add_argument(
        '--enable_log_file',
        action='store_true',
        help='If checked, log will be written as file.',
        default=False)
    parser.add_argument(
        '--disable_visualization',
        action='store_true',
        help='If checked, visualize result.',
        default=False)
    parser.add_argument(
        '--save_perception_2d',
        action='store_true',
        help='If checked, save 2D perception results (bbox and keypoints) to perception_2d.npz.',
        default=False)
    parser.add_argument(
        '--save_association',
        action='store_true',
        help='If checked, save cross-view ID matching/association results to cross_view_association.txt.',
        default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
