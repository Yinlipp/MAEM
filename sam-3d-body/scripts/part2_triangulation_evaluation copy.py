#!/usr/bin/env python3
"""
Part 2: Triangulation and Evaluation using xrmocap

This script:
1. Loads matched clusters from Part 1
2. Performs triangulation using xrmocap's AniposelibTriangulator
3. Evaluates against ground truth
4. Generates evaluation reports

Input:
- matched_clusters.pkl from Part 1

Output:
- evaluation_results.json
- 3D pose predictions
"""

import argparse
import json
import logging
import os
import pickle
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

# xrmocap imports
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrmocap.ops.triangulation.aniposelib_triangulator import AniposelibTriangulator

# Import evaluation functions from original script
from tri_pointmatch_evaluate_multiview_3dpose import (
    setup_logger,
    compute_mpjpe,
    compute_pa_mpjpe,
    align_keypoints_to_gt,
    find_best_matching_gt_person,
)


# ============================================================================
# RANSAC TRIANGULATION HELPERS
# ============================================================================

def load_projection_matrices(camera_params_dict: Dict, camera_param_dir: str) -> Dict[str, np.ndarray]:
    """Load camera projection matrices P = K @ [R | t] from JSON files.

    Returns:
        Dict mapping view_name -> projection matrix (3, 4)
    """
    proj_matrices = {}
    for view_name in sorted(camera_params_dict.keys()):
        view_idx = int(view_name.split('_')[1])
        param_file = os.path.join(camera_param_dir, f'fisheye_param_{view_idx:02d}.json')
        with open(param_file) as f:
            d = json.load(f)
        R = np.array(d['extrinsic_r'], dtype=np.float64)    # (3, 3)
        t = np.array(d['extrinsic_t'], dtype=np.float64)    # (3,)
        K = np.array(d['intrinsic'], dtype=np.float64)[:3, :3]  # (3, 3)
        P = K @ np.hstack([R, t.reshape(3, 1)])             # (3, 4)
        proj_matrices[view_name] = P
    return proj_matrices


def _dlt_triangulate_point(pts2d: np.ndarray, P_list: List[np.ndarray]) -> Optional[np.ndarray]:
    """DLT triangulation for one 3D point from N views (N >= 2).

    Args:
        pts2d:  (N, 2) pixel coordinates
        P_list: list of N projection matrices (3, 4)

    Returns:
        3D point (3,) in world coordinates, or None if degenerate.
    """
    A = []
    for (x, y), P in zip(pts2d, P_list):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-10:
        return None
    return X[:3] / X[3]


def _reproj_error(pt3d: np.ndarray, pt2d: np.ndarray, P: np.ndarray) -> float:
    """Reprojection error in pixels."""
    X = np.append(pt3d, 1.0)
    proj = P @ X
    if abs(proj[2]) < 1e-10:
        return float('inf')
    return float(np.linalg.norm(proj[:2] / proj[2] - pt2d))


def _triangulate_point_ransac(
    pts2d_all: np.ndarray,
    P_all: List[np.ndarray],
    valid_mask: np.ndarray,
    reproj_threshold: float = 20.0,
) -> Optional[np.ndarray]:
    """RANSAC triangulation for a single 3D point.

    Enumerates all view-pair hypotheses, selects the one with the most
    inliers (ties broken by mean reprojection error), then re-triangulates
    using all inlier views via least-squares DLT.

    Args:
        pts2d_all:        (N, 2) pixel coords for all N views (including invalid)
        P_all:            list of N projection matrices (3, 4)
        valid_mask:       (N,) bool array, True = view has a valid observation
        reproj_threshold: inlier threshold in pixels (default 20 px)

    Returns:
        3D point (3,) or None if not enough valid views.
    """
    valid_idx = np.where(valid_mask)[0]
    n_valid = len(valid_idx)

    if n_valid < 2:
        return None

    # With only 2 views there is a single hypothesis — skip RANSAC loop
    if n_valid == 2:
        pts = pts2d_all[valid_idx]
        Ps  = [P_all[i] for i in valid_idx]
        return _dlt_triangulate_point(pts, Ps)

    best_pt3d        = None
    best_n_inliers   = -1
    best_mean_err    = float('inf')
    best_inlier_idx  = None

    for pair in combinations(valid_idx, 2):
        pts = pts2d_all[list(pair)]
        Ps  = [P_all[i] for i in pair]
        pt3d = _dlt_triangulate_point(pts, Ps)
        if pt3d is None or not np.all(np.isfinite(pt3d)):
            continue

        # Evaluate against every valid view
        errors = [_reproj_error(pt3d, pts2d_all[i], P_all[i]) for i in valid_idx]
        inlier_flags = [e < reproj_threshold for e in errors]
        n_inliers    = sum(inlier_flags)
        mean_err     = float(np.mean([e for e, ok in zip(errors, inlier_flags) if ok])) \
                       if n_inliers > 0 else float('inf')

        if (n_inliers > best_n_inliers or
                (n_inliers == best_n_inliers and mean_err < best_mean_err)):
            best_n_inliers  = n_inliers
            best_mean_err   = mean_err
            best_pt3d       = pt3d
            best_inlier_idx = [valid_idx[k] for k, ok in enumerate(inlier_flags) if ok]

    # Re-triangulate using all inlier views (least-squares DLT)
    if best_inlier_idx is not None and len(best_inlier_idx) >= 2:
        pts = pts2d_all[best_inlier_idx]
        Ps  = [P_all[i] for i in best_inlier_idx]
        refined = _dlt_triangulate_point(pts, Ps)
        if refined is not None and np.all(np.isfinite(refined)):
            return refined

    # Fallback: all valid views
    if best_pt3d is not None:
        return best_pt3d

    pts = pts2d_all[valid_idx]
    Ps  = [P_all[i] for i in valid_idx]
    return _dlt_triangulate_point(pts, Ps)


def triangulate_cluster_ransac(
    matched_cluster: Dict,
    view_names_sorted: List[str],
    proj_matrices: Dict[str, np.ndarray],
    reproj_threshold: float = 20.0,
    logger=None,
) -> Optional[np.ndarray]:
    """Triangulate all keypoints for one person cluster using per-keypoint RANSAC.

    Args:
        matched_cluster:  cluster dict from Part 1
        view_names_sorted: ordered list of all view names
        proj_matrices:    dict view_name -> P (3, 4)
        reproj_threshold: RANSAC inlier threshold in pixels

    Returns:
        (n_keypoints, 3) array or None
    """
    poses_dict  = matched_cluster['poses']
    n_keypoints = 13
    n_views     = len(view_names_sorted)

    # Build (n_views, n_keypoints, 2) and (n_views, n_keypoints) valid mask
    pts2d_all  = np.zeros((n_views, n_keypoints, 2), dtype=np.float64)
    valid_mask = np.zeros((n_views, n_keypoints), dtype=bool)
    P_all      = []

    for vi, view_name in enumerate(view_names_sorted):
        P_all.append(proj_matrices[view_name])
        if view_name not in poses_dict:
            continue
        kpts = poses_dict[view_name].get('pred_keypoints_2d', None)
        if kpts is None or kpts.shape[1] < 2:
            continue
        for ki in range(n_keypoints):
            xy = kpts[ki, :2]
            if np.all(xy == 0) or not np.all(np.isfinite(xy)):
                continue
            pts2d_all[vi, ki] = xy
            valid_mask[vi, ki] = True

    valid_view_names = [view_names_sorted[vi] for vi in range(n_views)
                        if view_names_sorted[vi] in poses_dict]
    if logger:
        logger.info(f"      Triangulating from {len(valid_view_names)} views (RANSAC, thresh={reproj_threshold}px)")
        logger.info(f"        Valid views: {', '.join(valid_view_names)}")

    points3d = np.zeros((n_keypoints, 3), dtype=np.float64)
    for ki in range(n_keypoints):
        pt = _triangulate_point_ransac(
            pts2d_all[:, ki, :],   # (n_views, 2)
            P_all,
            valid_mask[:, ki],
            reproj_threshold=reproj_threshold,
        )
        if pt is not None:
            points3d[ki] = pt

    if logger:
        logger.info(f"        Output 3D pose shape: {points3d.shape}")

    return points3d


# ============================================================================
# TRIANGULATION USING XRMOCAP
# ============================================================================

def load_camera_parameters_xrmocap(camera_params_dict: Dict, camera_param_dir: str) -> List[FisheyeCameraParameter]:
    """
    Load camera parameters as FisheyeCameraParameter list

    Args:
        camera_params_dict: Dict mapping view_name to camera parameters (not used, for compatibility)
        camera_param_dir: Directory containing camera parameter JSON files

    Returns:
        List of FisheyeCameraParameter objects
    """
    camera_params = []
    view_names_sorted = sorted(camera_params_dict.keys())  

    # Extract view index from view name and load from file
    for view_name in view_names_sorted:
        
        view_idx = int(view_name.split('_')[1])
        param_file = os.path.join(camera_param_dir, f'fisheye_param_{view_idx:02d}.json')

        # Load using fromfile method (this is the correct way)
        cam_param = FisheyeCameraParameter.fromfile(param_file)
        camera_params.append(cam_param)

    return camera_params


def triangulate_cluster_xrmocap(
    matched_cluster: Dict,
    triangulator: AniposelibTriangulator,
    view_names_sorted: List[str],
    logger=None
) -> Optional[np.ndarray]:
    """
    Triangulate 3D pose from matched cluster using xrmocap

    Args:
        matched_cluster: Dict containing matched poses from different views
        triangulator: AniposelibTriangulator instance
        view_names_sorted: Sorted list of view names (ace_0, ace_1, ...)
        logger: Logger instance

    Returns:
        3D pose array of shape (n_keypoints, 3), or None if failed
    """
    poses_dict = matched_cluster['poses']
    num_views = matched_cluster['num_views']

    if logger:
        logger.info(f"      Triangulating from {num_views} views")

    # Collect 2D keypoints from all views
    all_views_points = []
    all_views_mask = []
    valid_view_names = []

    for view_name in view_names_sorted:
        if view_name not in poses_dict:
            # View not present, fill with zeros and mask as invalid
            n_keypoints = 13  # SportCenter dataset has 13 keypoints
            all_views_points.append(np.zeros((n_keypoints, 2)))
            all_views_mask.append(np.zeros((n_keypoints, 1)))
        else:
            pose_data = poses_dict[view_name]
            keypoints_2d = pose_data.get('pred_keypoints_2d', None)

            if keypoints_2d is None:
                if logger:
                    logger.warning(f"        {view_name}: No 2D keypoints found")
                n_keypoints = 13
                all_views_points.append(np.zeros((n_keypoints, 2)))
                all_views_mask.append(np.zeros((n_keypoints, 1)))
                continue

            # keypoints_2d shape: (n_keypoints, 2) or (n_keypoints, 3)
            if keypoints_2d.shape[1] >= 2:
                points_2d = keypoints_2d[:, :2]  # Extract x, y
            else:
                if logger:
                    logger.warning(f"        {view_name}: Invalid keypoints shape {keypoints_2d.shape}")
                n_keypoints = 13
                all_views_points.append(np.zeros((n_keypoints, 2)))
                all_views_mask.append(np.zeros((n_keypoints, 1)))
                continue

            # Create mask: 1 for valid points, 0 for invalid
            n_keypoints = points_2d.shape[0]
            mask = np.ones((n_keypoints, 1))

            # Mark [0, 0] points as invalid
            for kpt_idx in range(n_keypoints):
                if np.all(points_2d[kpt_idx] == 0) or np.any(~np.isfinite(points_2d[kpt_idx])):
                    mask[kpt_idx] = 0

            all_views_points.append(points_2d)
            all_views_mask.append(mask)
            valid_view_names.append(view_name)

    # Convert to numpy arrays
    # points2d: [n_view, n_keypoints, 2]
    # mask: [n_view, n_keypoints, 1]
    try:
        points2d = np.array(all_views_points, dtype=np.float64)
        mask = np.array(all_views_mask, dtype=np.int32)

        if logger:
            logger.info(f"        Input shape: points2d={points2d.shape}, mask={mask.shape}")
            logger.info(f"        Valid views: {', '.join(valid_view_names)}")
            logger.info(f"        Total valid points: {np.sum(mask)}/{mask.size}")

        # Triangulate using xrmocap
        points3d = triangulator.triangulate(points2d, points_mask=mask)

        if logger:
            logger.info(f"        Output 3D pose shape: {points3d.shape}")

        return points3d

    except Exception as e:
        if logger:
            logger.error(f"        Triangulation failed: {e}")
        return None


def fuse_poses_xrmocap(
    matched_clusters: List[Dict],
    triangulator: AniposelibTriangulator,
    view_names_sorted: List[str],
    proj_matrices: Optional[Dict[str, np.ndarray]] = None,
    reproj_threshold: float = 20.0,
    logger=None
) -> List[np.ndarray]:
    """
    Triangulate 3D poses for all matched clusters.

    When proj_matrices is provided, uses per-keypoint RANSAC triangulation.
    Otherwise falls back to AniposelibTriangulator (standard DLT).

    Args:
        matched_clusters:  List of matched person clusters
        triangulator:      AniposelibTriangulator instance (fallback)
        view_names_sorted: Sorted list of view names
        proj_matrices:     Optional dict view_name -> P (3,4); enables RANSAC
        reproj_threshold:  RANSAC inlier threshold in pixels
        logger:            Logger instance

    Returns:
        List of triangulated 3D poses
    """
    fused_poses = []

    for person_idx, cluster in enumerate(matched_clusters):
        if logger:
            logger.info(f"    Processing person {person_idx}")

        if proj_matrices is not None:
            pose_3d = triangulate_cluster_ransac(
                matched_cluster=cluster,
                view_names_sorted=view_names_sorted,
                proj_matrices=proj_matrices,
                reproj_threshold=reproj_threshold,
                logger=logger,
            )
        else:
            pose_3d = triangulate_cluster_xrmocap(
                matched_cluster=cluster,
                triangulator=triangulator,
                view_names_sorted=view_names_sorted,
                logger=logger,
            )

        if pose_3d is not None:
            fused_poses.append(pose_3d)
        else:
            if logger:
                logger.warning(f"      Failed to triangulate person {person_idx}, skipping")

    return fused_poses


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Part 2: Triangulation and evaluation using xrmocap')

    # I/O paths
    parser.add_argument('--intermediate_input', type=str,
                    #    default='/home/y_li/workspace3/sam-3d-body/output/sportscenter_21380/intermediate_matched_clusters.pkl',
                       default='/home/y_li/workspace3/sam-3d-body/output/humanm3/basketball1/split1/intermediate_matched_clusters.pkl',
                       help='Input path for intermediate matched clusters from Part 1')
    parser.add_argument('--output_dir', type=str,
                    #    default='/home/y_li/workspace3/sam-3d-body/output/sportscenter_21380',
                       default='/home/y_li/workspace3/sam-3d-body//output/humanm3/basketball1/split1/',
                       help='Output directory for evaluation results')
    parser.add_argument('--gt_file', type=str,
                    #    default='/home/y_li/workspace3/xrmocap/xrmocap_data/xrmocap_sportscenter/xrmocap_meta_sportscenter/scene_0/keypoints3d_GT.npz',
                       default='/home/y_li/workspace3/humanm3/test/basketball1/split1/keypoints3d_GT.npz',
                       help='Ground truth npz file')
    parser.add_argument('--save_3d_poses', action='store_true', default=True,
                       help='Save triangulated 3D poses to npz file')
    parser.add_argument('--visualize_3d', action='store_true', default=True,
                       help='Visualize 3D poses (requires xrmocap visualization)')

    args = parser.parse_args()

    # Setup logger
    logger, log_file = setup_logger(args.output_dir, 'part2_triangulation')
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    logger.info("Part 2: Triangulation and Evaluation using xrmocap")
    logger.info("="*80)
    logger.info(f"Intermediate input: {args.intermediate_input}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"GT file: {args.gt_file}")
    logger.info("")

    # Load intermediate data from Part 1
    logger.info("Loading intermediate data from Part 1...")

    # Handle numpy version compatibility issues
    import sys
    import numpy.core._multiarray_umath as _multiarray_umath
    sys.modules['numpy._core._multiarray_umath'] = _multiarray_umath
    sys.modules['numpy._core'] = np.core

    with open(args.intermediate_input, 'rb') as f:
        intermediate_data = pickle.load(f)

    all_frame_data = intermediate_data['all_frame_data']
    camera_params_dict = intermediate_data['camera_params_dict']

    logger.info(f"  Loaded {len(all_frame_data)} frames")
    logger.info(f"  Total matched clusters: {sum(fd['num_persons'] for fd in all_frame_data)}")
    logger.info("")

    # Load camera parameters for xrmocap
    logger.info("Loading camera parameters for xrmocap...")

    # Get camera parameter directory from args (from Part 1)
    part1_args = intermediate_data.get('args', {})
    camera_param_dir = part1_args.get('camera_param_dir')
    print(f'camera_param_dir {camera_param_dir}')

    camera_params_xrmocap = load_camera_parameters_xrmocap(camera_params_dict, camera_param_dir)
    view_names_sorted = sorted(camera_params_dict.keys())
    logger.info(f"  Loaded {len(camera_params_xrmocap)} camera parameters")
    logger.info(f"  Views: {', '.join(view_names_sorted)}")
    logger.info("")

    # Load projection matrices for RANSAC triangulation
    proj_matrices = load_projection_matrices(camera_params_dict, camera_param_dir)
    logger.info(f"  Loaded projection matrices for {len(proj_matrices)} views (RANSAC)")
    logger.info("")

    # Create triangulator (fallback)
    logger.info("Creating AniposelibTriangulator...")
    triangulator = AniposelibTriangulator(camera_parameters=camera_params_xrmocap)
    logger.info("  Triangulator created successfully")
    logger.info("")

    # Load ground truth
    logger.info(f"Loading ground truth from {args.gt_file}...")
    gt_data = np.load(args.gt_file)
    gt_poses = gt_data['pose']
    gt_mask = gt_data['mask'] if 'mask' in gt_data else None
    logger.info(f"  GT shape: {gt_poses.shape}")
    logger.info("")

    # Initialize result containers
    all_mpjpe = []
    all_pa_mpjpe = []
    all_frame_summaries = []

    # Storage for 3D poses
    # We'll store poses per frame, then convert to [n_frames, n_persons, n_keypoints, 3]
    all_frames_poses = []  # List of [n_persons, n_keypoints, 3] for each frame
    frame_to_idx_mapping = {}  # Map frame_num to index in all_frames_poses

    # Process each frame
    logger.info("Processing frames...")
    for frame_data in all_frame_data:
        frame_idx = frame_data['frame_idx']
        frame_num = frame_data['frame_num']
        matched_clusters = frame_data['matched_clusters']
        num_persons = frame_data['num_persons']

        logger.info(f"\n--- Frame {frame_num} ({frame_idx}/{len(all_frame_data)}) ---")
        logger.info(f"  Processing {num_persons} matched person(s)")

        # Triangulate 3D poses using RANSAC
        logger.info(f"  Triangulating 3D poses...")
        frame_avg_poses = fuse_poses_xrmocap(
            matched_clusters=matched_clusters,
            triangulator=triangulator,
            view_names_sorted=view_names_sorted,
            proj_matrices=proj_matrices,
            reproj_threshold=20.0,
            logger=logger,
        )

        if not frame_avg_poses:
            logger.info(f"  No valid triangulated poses for frame {frame_num}")
            continue

        logger.info(f"  Successfully triangulated {len(frame_avg_poses)} pose(s)")

        # Store triangulated 3D poses for this frame
        if args.save_3d_poses and frame_avg_poses:
            # Store poses: [n_persons, n_keypoints, 3]
            frame_poses_array = np.array(frame_avg_poses)
            all_frames_poses.append(frame_poses_array)
            frame_to_idx_mapping[frame_num] = len(all_frames_poses) - 1
            logger.info(f"  Stored 3D poses for frame {frame_num}, shape: {frame_poses_array.shape}")

        # Evaluate against ground truth
        logger.info(f"  Matching to ground truth...")

        if frame_idx >= len(gt_poses):
            logger.info(f"  Warning: frame_idx {frame_idx} out of GT range, skipping")
            continue

        gt_pose_frame = gt_poses[frame_idx]
        gt_mask_frame = gt_mask[frame_idx] if gt_mask is not None else None

        used_gt_indices = set()
        frame_person_results = []

        for person_idx, fused_pose in enumerate(frame_avg_poses):
            logger.info(f"    Evaluating person {person_idx}...")

            best_gt_idx, best_gt_pose, best_gt_mask, _ = find_best_matching_gt_person(
                fused_pose, gt_pose_frame, gt_mask_frame, used_gt_indices, logger
            )

            if best_gt_idx is None:
                logger.info(f"      No valid GT match found")
                continue

            used_gt_indices.add(best_gt_idx)

            # Align if needed
            if fused_pose.shape[0] != best_gt_pose.shape[0]:
                try:
                    fused_pose_aligned, _ = align_keypoints_to_gt(fused_pose, best_gt_pose, use_hungarian=True)
                except Exception as e:
                    logger.info(f"      Failed to align: {e}")
                    continue
            else:
                fused_pose_aligned = fused_pose

            # Compute metrics
            mpjpe = compute_mpjpe(fused_pose_aligned, best_gt_pose, best_gt_mask)
            pa_mpjpe = compute_pa_mpjpe(fused_pose_aligned, best_gt_pose, best_gt_mask)

            logger.info(f"      Matched to GT person {best_gt_idx}")
            logger.info(f"      MPJPE: {mpjpe:.2f} mm")
            logger.info(f"      PA-MPJPE: {pa_mpjpe:.2f} mm")

            all_mpjpe.append(mpjpe)
            all_pa_mpjpe.append(pa_mpjpe)

            frame_person_results.append({
                'person_idx': person_idx,
                'gt_person_idx': best_gt_idx,
                'mpjpe': float(mpjpe),
                'pa_mpjpe': float(pa_mpjpe)
            })

        # Store frame summary
        if frame_person_results:
            all_frame_summaries.append({
                'frame_idx': frame_idx,
                'frame_num': frame_num,
                'num_matched': len(frame_person_results),
                'persons': frame_person_results
            })

    # Compute overall statistics
    logger.info(f"\n{'='*80}")
    logger.info("Evaluation Results")
    logger.info("="*80)

    if all_mpjpe:
        mean_mpjpe = np.mean(all_mpjpe)
        median_mpjpe = np.median(all_mpjpe)
        mean_pa_mpjpe = np.mean(all_pa_mpjpe)
        median_pa_mpjpe = np.median(all_pa_mpjpe)

        mpjpe_arr = np.array(all_mpjpe)
        ap_thresholds = [75, 100, 125, 150]
        ap_values = {t: float((mpjpe_arr < t).mean()) for t in ap_thresholds}

        logger.info(f"Total evaluated poses: {len(all_mpjpe)}")
        logger.info(f"MPJPE: {mean_mpjpe:.2f} mm (mean), {median_mpjpe:.2f} mm (median)")
        logger.info(f"PA-MPJPE: {mean_pa_mpjpe:.2f} mm (mean), {median_pa_mpjpe:.2f} mm (median)")
        for t in ap_thresholds:
            logger.info(f"AP@{t}mm: {ap_values[t]*100:.2f}%")

        # Save results
        results = {
            'summary': {
                'total_frames': len(all_frame_data),
                'evaluated_frames': len(all_frame_summaries),
                'total_poses': len(all_mpjpe),
                'mean_mpjpe': float(mean_mpjpe),
                'median_mpjpe': float(median_mpjpe),
                'mean_pa_mpjpe': float(mean_pa_mpjpe),
                'median_pa_mpjpe': float(median_pa_mpjpe),
                'ap75': ap_values[75],
                'ap100': ap_values[100],
                'ap125': ap_values[125],
                'ap150': ap_values[150],
            },
            'frame_summaries': all_frame_summaries
        }

        output_json = os.path.join(args.output_dir, 'evaluation_results_xrmocap.json')
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {output_json}")
    else:
        logger.info("No valid evaluations performed")

    # Save 3D poses to npz file
    if args.save_3d_poses and all_frames_poses:
        logger.info(f"\n{'='*80}")
        logger.info("Saving 3D Poses")
        logger.info("="*80)

        # Convert to numpy array with proper padding for frames with different number of persons
        # Find max number of persons across all frames
        max_persons = max(poses.shape[0] for poses in all_frames_poses)
        n_keypoints = all_frames_poses[0].shape[1]

        logger.info(f"  Total frames with poses: {len(all_frames_poses)}")
        logger.info(f"  Max persons per frame: {max_persons}")
        logger.info(f"  Keypoints per person: {n_keypoints}")

        # Create padded array: [n_frames, max_persons, n_keypoints, 3]
        n_frames = len(all_frames_poses)
        poses_3d_padded = np.zeros((n_frames, max_persons, n_keypoints, 3), dtype=np.float32)
        poses_mask = np.zeros((n_frames, max_persons), dtype=np.int32)  # Mask for valid persons

        for frame_idx, frame_poses in enumerate(all_frames_poses):
            n_persons_in_frame = frame_poses.shape[0]
            poses_3d_padded[frame_idx, :n_persons_in_frame, :, :] = frame_poses
            poses_mask[frame_idx, :n_persons_in_frame] = 1

        # Save to npz
        output_npz = os.path.join(args.output_dir, 'predicted_3d_poses_xrmocap.npz')
        np.savez(
            output_npz,
            pose=poses_3d_padded,
            mask=poses_mask,
            frame_mapping=frame_to_idx_mapping,
            convention='13points'
        )

        logger.info(f"  Saved 3D poses to: {output_npz}")
        logger.info(f"  Shape: {poses_3d_padded.shape} (frames, persons, keypoints, xyz)")
        logger.info(f"  Mask shape: {poses_mask.shape}")
        logger.info(f"  Total valid poses: {np.sum(poses_mask)}")

        # Also save in the same format as GT for easy comparison
        # GT format: [n_frames, n_persons, n_keypoints, 3]
        output_npz_gt_format = os.path.join(args.output_dir, 'predicted_3d_poses_gt_format.npz')

        # Get the GT shape to match
        if gt_poses is not None:
            gt_n_frames, gt_n_persons, gt_n_keypoints, _ = gt_poses.shape
            logger.info(f"\n  Matching GT format: {gt_poses.shape}")

            # Create array matching GT dimensions
            poses_3d_gt_format = np.zeros((gt_n_frames, gt_n_persons, gt_n_keypoints, 3), dtype=np.float32)

            # Fill in the predictions we have
            for frame_num, pred_idx in frame_to_idx_mapping.items():
                if frame_num < gt_n_frames:
                    frame_poses = all_frames_poses[pred_idx]
                    n_pred_persons = min(frame_poses.shape[0], gt_n_persons)
                    poses_3d_gt_format[frame_num, :n_pred_persons, :, :] = frame_poses[:n_pred_persons, :, :]

            np.savez(
                output_npz_gt_format,
                pose=poses_3d_gt_format,
                convention='13points'
            )
            logger.info(f"  Saved GT-format poses to: {output_npz_gt_format}")
            logger.info(f"  Shape: {poses_3d_gt_format.shape}")

    # Visualize 3D poses if requested
    if args.visualize_3d and all_frames_poses:
        logger.info(f"\n{'='*80}")
        logger.info("Visualizing 3D Poses")
        logger.info("="*80)

        try:
            from xrmocap.visualization.visualize_keypoints3d import visualize_keypoints3d
            from xrprimer.data_structure import Keypoints

            # Create Keypoints object
            poses_3d_array = poses_3d_padded  # Use the padded array we created

            keypoints3d = Keypoints(
                kps=poses_3d_array.reshape(-1, n_keypoints, 3),  # Flatten frames and persons
                mask=np.ones_like(poses_3d_array[..., 0]),
                convention='13points',
                logger=logger
            )

            # Visualize
            output_video = os.path.join(args.output_dir, 'predicted_3d_poses_visualization.mp4')
            visualize_keypoints3d(
                keypoints=keypoints3d,
                output_path=output_video,
                return_array=False,
                overwrite=True
            )

            logger.info(f"  3D visualization saved to: {output_video}")

        except Exception as e:
            logger.warning(f"  Failed to visualize 3D poses: {e}")
            logger.warning(f"  You can use xrmocap visualization tools separately")

    logger.info(f"\n{'='*80}")
    logger.info("Part 2 Complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
