#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project 3D GT keypoints onto 2D images and visualize.

This script loads 3D GT keypoints from the SportCenter dataset, projects them
onto 2D images from each camera view, and saves the visualization results.

Data format:
- keypoints3d_GT.npz: contains pose (20, 10, 13, 4), mask (20, 10, 13), convention
  - pose: [n_frames, n_persons, n_keypoints, 4], last dim is [x, y, z, confidence]
  - mask: keypoint visibility mask
  - convention: keypoint convention 
"""

import os
import numpy as np
import cv2
import json
from typing import List, Dict, Tuple
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.data_structure import Keypoints
from xrmocap.ops.projection.aniposelib_projector import AniposelibProjector
from xrmocap.visualization.visualize_keypoints2d import visualize_keypoints2d


def load_camera_parameters(camera_dir: str, n_views: int = 8) -> List[FisheyeCameraParameter]:
    """Load camera parameters.

    Args:
        camera_dir: Path to the camera parameter directory.
        n_views: Number of camera views.

    Returns:
        List of camera parameters.
    """
    camera_params = []
    for view_idx in range(n_views):
        cam_path = os.path.join(camera_dir, f'fisheye_param_{view_idx:02d}.json')
        if not os.path.exists(cam_path):
            raise FileNotFoundError(f"Camera parameter file not found: {cam_path}")
        cam_param = FisheyeCameraParameter.fromfile(cam_path)
        if cam_param.world2cam:
            cam_param.inverse_extrinsic()
        camera_params.append(cam_param)
    return camera_params


def load_gt_keypoints(gt_path: str):
    """Load GT keypoint data.

    Args:
        gt_path: Path to the GT data file.

    Returns:
        tuple: (keypoints3d, mask, convention)
            - keypoints3d: [n_frames, n_persons, n_keypoints, 4]
            - mask: [n_frames, n_persons, n_keypoints]
            - convention: keypoint convention name
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT data file not found: {gt_path}")

    gt_data = np.load(gt_path)
    print(f"GT data file keys: {list(gt_data.keys())}")

    if 'pose' not in gt_data.keys():
        raise ValueError("'pose' key not found in GT data file")

    keypoints3d = gt_data['pose']  # [n_frames, n_persons, n_keypoints, 4]
    mask = gt_data.get('mask', None)  # [n_frames, n_persons, n_keypoints]
    convention = str(gt_data.get('convention', 'unknown'))

    print(f"Loaded pose data, shape: {keypoints3d.shape}")
    print(f"Loaded mask data, shape: {mask.shape if mask is not None else None}")
    print(f"Keypoint convention: {convention}")

    return keypoints3d, mask, convention


def load_pred_gt_matching(json_path: str) -> Dict[int, Dict[int, int]]:
    """Load prediction-GT matching information.

    Args:
        json_path: Path to the evaluation_results JSON file.

    Returns:
        Dict[frame_idx, Dict[person_idx, gt_person_idx]]: per-frame pred-GT matching dict.
            e.g. {0: {1: 7, 2: 1, ...}, ...} means in frame 0,
            predicted person_idx=1 is matched to GT person_idx=7.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Matching info file not found: {json_path}")

    with open(json_path, 'r') as f:
        eval_results = json.load(f)

    matching_dict = {}

    for frame_summary in eval_results.get('frame_summaries', []):
        frame_idx = frame_summary['frame_idx']
        matching_dict[frame_idx] = {}

        for person in frame_summary.get('persons', []):
            person_idx = person['person_idx']
            gt_person_idx = person['gt_person_idx']
            matching_dict[frame_idx][person_idx] = gt_person_idx

    print(f"Loaded matching info for {len(matching_dict)} frames")
    return matching_dict


def generate_color_palette(n_colors: int) -> List[Tuple[int, int, int]]:
    """Generate a color palette.

    Args:
        n_colors: Number of colors needed.

    Returns:
        List of colors, each as a (B, G, R) tuple.
    """
    import colorsys
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        # Convert to BGR for OpenCV
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors


def load_image_lists(image_root: str, n_views: int = None) -> List[List[str]]:
    """Load image path lists by scanning all subdirectories under image_root.

    Args:
        image_root: Image root directory; each subdirectory corresponds to one view.
        n_views: Number of views to load; None means load all subdirectories.

    Returns:
        List of image path lists per view (sorted by subdirectory name).
    """
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
    subdirs = sorted([
        d for d in os.listdir(image_root)
        if os.path.isdir(os.path.join(image_root, d))
    ])
    if n_views is not None:
        subdirs = subdirs[:n_views]

    image_lists = []
    for subdir in subdirs:
        dir_path = os.path.join(image_root, subdir)
        imgs = sorted([
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ])
        image_lists.append(imgs)

    return image_lists


def project_keypoints3d_to_2d(
    keypoints3d: np.ndarray,
    mask: np.ndarray,
    projector: AniposelibProjector
) -> np.ndarray:
    """Project 3D keypoints to 2D.

    Args:
        keypoints3d: [n_frames, n_persons, n_keypoints, 4]
        mask: [n_frames, n_persons, n_keypoints]
        projector: Projector instance.

    Returns:
        keypoints2d: [n_views, n_frames, n_persons, n_keypoints, 3]
            Last dim is [x, y, confidence].
    """
    n_frames, n_persons, n_keypoints, _ = keypoints3d.shape
    n_views = len(projector.camera_parameters)

    # Initialize 2D keypoint array
    keypoints2d = np.zeros((n_views, n_frames, n_persons, n_keypoints, 3))

    # Project per frame per person
    for frame_idx in range(n_frames):
        for person_idx in range(n_persons):
            # Get 3D keypoints for current frame and person
            kps3d = keypoints3d[frame_idx, person_idx, :, :3]  # [n_keypoints, 3]
            if keypoints3d.shape[3] >= 4:
                confidence = keypoints3d[frame_idx, person_idx, :, 3]  # [n_keypoints]
            else:
                confidence = np.ones(n_keypoints, dtype=np.float32)
            if mask is None:
                kps_mask = np.ones(n_keypoints)
            elif mask.ndim == 3:  # (n_frames, n_persons, n_keypoints)
                kps_mask = mask[frame_idx, person_idx, :]
            else:  # (n_frames, n_persons) per-person mask
                kps_mask = np.ones(n_keypoints) * mask[frame_idx, person_idx]

            # Project to each view
            kps2d = projector.project(kps3d, kps_mask)  # [n_views, n_keypoints, 2]

            # Save results and keep confidence
            keypoints2d[:, frame_idx, person_idx, :, :2] = kps2d
            keypoints2d[:, frame_idx, person_idx, :, 2] = confidence * kps_mask

    return keypoints2d


def visualize_projected_keypoints_with_matching(
    image_lists: List[List[str]],
    output_dir: str,
    draw_mode: str = 'both',
    pred_keypoints2d: np.ndarray = None,
    gt_keypoints2d: np.ndarray = None,
    matching_dict: Dict[int, Dict[int, int]] = None,
    n_views: int = 8,
    start_frame: int = 0,
    end_frame: int = None
):
    """Visualize projected keypoints.

    Args:
        image_lists: Image path lists per view.
        output_dir: Output directory.
        draw_mode: Drawing mode, one of:
            'both'      - Draw both prediction (filled circle) and GT (hollow circle).
            'gt_only'   - Draw GT keypoints only (hollow circle).
            'pred_only' - Draw prediction keypoints only (filled circle).
        pred_keypoints2d: [n_views, n_frames, n_persons, n_keypoints, 3] predicted keypoints.
        gt_keypoints2d: [n_views, n_frames, n_persons, n_keypoints, 3] GT keypoints.
        matching_dict: Pred-GT matching dict {frame_idx: {pred_person_idx: gt_person_idx}}.
        n_views: Number of views.
        start_frame: Start frame index.
        end_frame: End frame index.
    """
    assert draw_mode in ('both', 'gt_only', 'pred_only'), \
        f"draw_mode must be 'both', 'gt_only', or 'pred_only', got: {draw_mode}"

    if draw_mode in ('both', 'pred_only'):
        assert pred_keypoints2d is not None, \
            "draw_mode='{}' requires pred_keypoints2d".format(draw_mode)
    if draw_mode in ('both', 'gt_only'):
        assert gt_keypoints2d is not None, \
            "draw_mode='{}' requires gt_keypoints2d".format(draw_mode)

    os.makedirs(output_dir, exist_ok=True)

    # Get dimensions from available data
    if pred_keypoints2d is not None:
        n_views_actual, n_frames, n_pred_persons, n_keypoints, _ = pred_keypoints2d.shape
    else:
        n_views_actual, n_frames, n_pred_persons, n_keypoints, _ = gt_keypoints2d.shape
        n_pred_persons = 0

    n_gt_persons = gt_keypoints2d.shape[2] if gt_keypoints2d is not None else 0

    if end_frame is None:
        end_frame = n_frames
    else:
        end_frame = min(end_frame, n_frames)

    print(f"\nStarting visualization, mode: {draw_mode}, frame range: {start_frame} - {end_frame}")
    if draw_mode in ('both', 'pred_only'):
        print(f"Number of predicted persons: {n_pred_persons}")
    if draw_mode in ('both', 'gt_only'):
        print(f"Number of GT persons: {n_gt_persons}")

    max_persons = max(n_pred_persons, n_gt_persons)
    colors = generate_color_palette(max(max_persons, 1))

    # Visualize per view
    for view_idx in range(min(n_views, n_views_actual)):
        print(f"\nProcessing view {view_idx}...")

        view_img_list = image_lists[view_idx][start_frame:end_frame]

        output_images_dir = os.path.join(output_dir, f'projection_{draw_mode}_view_{view_idx:02d}')
        os.makedirs(output_images_dir, exist_ok=True)

        for frame_offset in range(end_frame - start_frame):
            frame_idx = start_frame + frame_offset
            img_path = view_img_list[frame_offset]

            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            h, w = img.shape[:2]

            def draw_pt(im, x, y, conf, clr, filled):
                """Draw a single keypoint, filtering out invalid coordinates."""
                if conf <= 0 or not np.isfinite(x) or not np.isfinite(y):
                    return
                px, py = int(round(float(x))), int(round(float(y)))
                if px < -w or px > 2 * w or py < -h or py > 2 * h:
                    return
                thickness = -1 if filled else 2
                cv2.circle(im, (px, py), 5, clr, thickness)

            if draw_mode == 'both':
                # Color by matched pair if matching info available, else by person index
                frame_matching = matching_dict.get(frame_idx, {}) if matching_dict else {}
                matched_gt_indices = set(frame_matching.values())

                # Draw matched pred-GT pairs (same color)
                for pred_person_idx, gt_person_idx in frame_matching.items():
                    if pred_person_idx >= n_pred_persons or gt_person_idx >= n_gt_persons:
                        continue
                    color = colors[pred_person_idx % len(colors)]

                    pred_kps = pred_keypoints2d[view_idx, frame_idx, pred_person_idx]
                    for kp_idx in range(n_keypoints):
                        x, y, conf = pred_kps[kp_idx]
                        draw_pt(img, x, y, conf, color, filled=True)

                    gt_kps = gt_keypoints2d[view_idx, frame_idx, gt_person_idx]
                    for kp_idx in range(n_keypoints):
                        x, y, conf = gt_kps[kp_idx]
                        draw_pt(img, x, y, conf, color, filled=False)

                # Unmatched GT persons (gray hollow circles)
                unmatched_color = (180, 180, 180)
                for gt_person_idx in range(n_gt_persons):
                    if gt_person_idx in matched_gt_indices:
                        continue
                    gt_kps = gt_keypoints2d[view_idx, frame_idx, gt_person_idx]
                    for kp_idx in range(n_keypoints):
                        x, y, conf = gt_kps[kp_idx]
                        draw_pt(img, x, y, conf, unmatched_color, filled=False)

            elif draw_mode == 'gt_only':
                for gt_person_idx in range(n_gt_persons):
                    color = colors[gt_person_idx % len(colors)]
                    gt_kps = gt_keypoints2d[view_idx, frame_idx, gt_person_idx]
                    for kp_idx in range(n_keypoints):
                        x, y, conf = gt_kps[kp_idx]
                        draw_pt(img, x, y, conf, color, filled=False)

            elif draw_mode == 'pred_only':
                for pred_person_idx in range(n_pred_persons):
                    color = colors[pred_person_idx % len(colors)]
                    pred_kps = pred_keypoints2d[view_idx, frame_idx, pred_person_idx]
                    for kp_idx in range(n_keypoints):
                        x, y, conf = pred_kps[kp_idx]
                        draw_pt(img, x, y, conf, color, filled=True)

            output_img_path = os.path.join(output_images_dir, os.path.basename(img_path))
            cv2.imwrite(output_img_path, img)

        print(f"View {view_idx} visualization complete: {output_images_dir}")


def main():
    """Main function."""

    # ============= Drawing mode =============
    # 'both'      - Draw both prediction (filled circle) and GT (hollow circle);
    #               requires pred_path, gt_path, matching_json_path
    # 'gt_only'   - Draw GT keypoints only (hollow circle); requires gt_path
    # 'pred_only' - Draw prediction keypoints only (filled circle); requires pred_path
    draw_mode = 'gt_only'

    # GT data file (required for draw_mode='both' or 'gt_only')
    gt_path = '/home/y_li/workspace7/humanm3/test/basketball2/keypoints3d_GT.npz'

    # Prediction data file (required for draw_mode='both' or 'pred_only')
    pred_path = '/home/y_li/workspace7/sam-3d-body/output/humanm3/basketball2/predicted_3d_poses_xrmocap.npz'

    # Matching info file (required for draw_mode='both')
    matching_json_path = '/home/y_li/workspace7/sam-3d-body/output/humanm3/plaza/evaluation_results_xrmocap.json'

    # Camera parameter directory
    camera_dir = '/home/y_li/workspace7/humanm3/test/basketball2/camera_calibration'

    # Image root directory (each subdirectory corresponds to one view)
    image_root = '/home/y_li/workspace7/humanm3/test/basketball2/images'

    # Output directory
    output_dir = os.path.join('/home/y_li/workspace7/xrmocap', 'output_bas2')

    # Number of views
    n_views = 3

    # Frame range
    start_frame = 0
    end_frame = None  # None means process all frames

    print("=" * 60)
    print(f"Projection visualization (mode: {draw_mode})")
    print("  both:      prediction=filled circle, GT=hollow circle, matched pairs share color")
    print("  gt_only:   GT only (hollow circle)")
    print("  pred_only: prediction only (filled circle)")
    print("=" * 60)

    # ============= 1. Load camera parameters =============
    print("\n[1] Loading camera parameters...")
    camera_params = load_camera_parameters(camera_dir, n_views)
    print(f"Loaded {len(camera_params)} camera parameters")

    projector = AniposelibProjector(camera_parameters=camera_params)

    # ============= 2. Load prediction keypoints if needed =============
    pred_keypoints2d = None
    if draw_mode in ('both', 'pred_only'):
        print("\n[2] Loading predicted keypoint data...")
        pred_keypoints3d, pred_mask, pred_convention = load_gt_keypoints(pred_path)
        print("  Projecting predicted keypoints...")
        pred_keypoints2d = project_keypoints3d_to_2d(pred_keypoints3d, pred_mask, projector)
        print(f"  Predicted 2D keypoints shape: {pred_keypoints2d.shape}")

    # ============= 3. Load GT keypoints if needed =============
    gt_keypoints2d = None
    if draw_mode in ('both', 'gt_only'):
        print("\n[3] Loading GT keypoint data...")
        gt_keypoints3d, gt_mask, gt_convention = load_gt_keypoints(gt_path)
        print("  Projecting GT keypoints...")
        gt_keypoints2d = project_keypoints3d_to_2d(gt_keypoints3d, gt_mask, projector)
        print(f"  GT 2D keypoints shape: {gt_keypoints2d.shape}")

    # ============= 4. Load matching info if needed =============
    matching_dict = None
    if draw_mode == 'both':
        print("\n[4] Loading pred-GT matching info...")
        matching_dict = load_pred_gt_matching(matching_json_path)

    # ============= 5. Load image lists =============
    print("\n[5] Loading image path lists...")
    image_lists = load_image_lists(image_root, n_views)
    print(f"Loaded image lists for {len(image_lists)} views")
    for view_idx, img_list in enumerate(image_lists):
        print(f"  View {view_idx}: {len(img_list)} images")

    # ============= 6. Visualize =============
    print("\n[6] Visualizing projection results...")
    visualize_projected_keypoints_with_matching(
        image_lists=image_lists,
        output_dir=output_dir,
        draw_mode=draw_mode,
        pred_keypoints2d=pred_keypoints2d,
        gt_keypoints2d=gt_keypoints2d,
        matching_dict=matching_dict,
        n_views=n_views,
        start_frame=start_frame,
        end_frame=end_frame
    )

    print("\n" + "=" * 60)
    print(f"Done! Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
