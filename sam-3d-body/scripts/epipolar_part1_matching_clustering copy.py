#!/usr/bin/env python3
"""
Part 1: Multi-view Pose Matching and Clustering

This script:
1. Loads 3D poses from multiple camera views
2. Transforms poses to world coordinates
3. Matches poses across views using PA-MPJPE
4. Saves matched clusters to intermediate file

Output:
- matched_clusters.pkl: Contains matched pose clusters with 2D keypoints for triangulation
"""

import argparse
import json
import logging
import os
import sys
import pickle
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ============================================================================
# Constants
# ============================================================================

KEYPOINTS_IDX = [0, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]  # 13 keypoints for sportscenter dataset


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(output_dir: str, log_name: str = 'evaluation') -> Tuple[logging.Logger, str]:
    """Setup logger to write to both file and console."""
    logger = logging.getLogger('multiview_eval')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'{log_name}_{timestamp}.log')

    # File handler with detailed format
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)

    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    return logger, log_file


# ============================================================================
# CAMERA TRANSFORMATIONS
# ============================================================================

def load_camera_params(camera_param_path: str) -> Dict:
    """Load camera parameters from JSON file."""
    with open(camera_param_path, 'r') as f:
        return json.load(f)


def get_transform_matrix(camera_params: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract rotation, translation, and intrinsic matrices from camera parameters."""
    if 'extrinsic_r' in camera_params and 'extrinsic_t' in camera_params:
        R = np.array(camera_params['extrinsic_r'])
        t = np.array(camera_params['extrinsic_t'])
    else:
        # extrinsic is a 4x4 homogeneous matrix [R|t; 0 0 0 1]
        extrinsic = np.array(camera_params['extrinsic'])
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
    intrinsic = np.array(camera_params['intrinsic'])[:3, :3]
    return R, t, intrinsic


def get_dist_coeffs(camera_params: Dict) -> np.ndarray:
    """Extract OpenCV distortion coefficients [k1, k2, p1, p2, k3] from camera parameters."""
    return np.array([
        camera_params.get('k1', 0.0),
        camera_params.get('k2', 0.0),
        camera_params.get('p1', 0.0),
        camera_params.get('p2', 0.0),
        camera_params.get('k3', 0.0),
    ])


def undistort_points(pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Undistort 2D pixel points using OpenCV distortion model.

    Args:
        pts:  (N, 2) distorted pixel coordinates
        K:    (3, 3) intrinsic matrix
        dist: (5,)  distortion coefficients [k1, k2, p1, p2, k3]

    Returns:
        (N, 2) undistorted pixel coordinates
    """
    if pts is None or len(pts) == 0:
        return pts
    pts_f = pts.astype(np.float32).reshape(-1, 1, 2)
    undist = cv2.undistortPoints(pts_f, K, dist, P=K)
    return undist.reshape(-1, 2)


def transform_to_world(keypoints3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Transform 3D keypoints from camera to world coordinates.

    Mathematical principle:
        Camera extrinsic defines world-to-camera transform: P_cam = R * P_world + t
        Inverse transform (camera-to-world): P_world = R^T * (P_cam - t)
        Since R is orthogonal: R^(-1) = R^T
    """
    t = np.array(t).ravel()
    return (keypoints3d - t) @ R


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_similarity_transform(S1: np.ndarray, S2: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Computes a similarity transform (sR, t) that takes S1 closest to S2."""
    mu1, mu2 = S1.mean(axis=0), S2.mean(axis=0)
    X1, X2 = S1 - mu1, S2 - mu2

    var1 = np.sum(X1**2)
    K = X1.T @ X2

    U, s, Vh = np.linalg.svd(K)
    V = Vh.T

    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U @ V.T))

    R = V @ Z @ U.T
    scale = np.trace(R @ K) / var1 if var1 >= 1e-10 else 1.0
    t = mu2 - scale * (R @ mu1)

    return scale, R, t


def compute_pa_mpjpe(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute Procrustes Aligned MPJPE in mm."""
    if len(pred) == 0 or len(gt) == 0:
        return float('inf')

    finite_pred = np.all(np.isfinite(pred), axis=1)
    finite_gt = np.all(np.isfinite(gt), axis=1)
    valid = finite_pred & finite_gt
    if mask is not None:
        valid = valid & mask.astype(bool)

    if valid.sum() < 3:
        return float('inf')

    s, R, t = compute_similarity_transform(pred[valid], gt[valid])
    pred_aligned = s * (R @ pred.T).T + t
    distances = np.linalg.norm(pred_aligned - gt, axis=1) * 1000

    return float(np.mean(distances[valid]))


# ============================================================================
# EPIPOLAR GEOMETRY
# ============================================================================

def compute_fundamental_matrix(K1: np.ndarray, R1: np.ndarray, t1: np.ndarray,
                               K2: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """
    Compute fundamental matrix F such that x2^T F x1 = 0.
    Camera model: P_cam = R @ P_world + t  (world-to-camera).
    """
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    tx, ty, tz = t_rel.ravel()
    t_cross = np.array([[0, -tz, ty],
                         [tz,  0, -tx],
                         [-ty, tx,  0]])
    E = t_cross @ R_rel
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F


def compute_epipolar_cost(verts_2d_1: np.ndarray, verts_2d_2: np.ndarray,
                          F: np.ndarray, epi_threshold: float = 20.0, logger=None) -> float:
    """
    Compute normalized mean symmetric epipolar distance [0, 1].

    Args:
        verts_2d_1: (N, 2) points in view1 (pixel coords), N >= 1
        verts_2d_2: (N, 2) points in view2 (pixel coords), N >= 1
        F: (3, 3) fundamental matrix (view1 -> view2)
        epi_threshold: distance threshold in pixels for normalization

    Returns:
        Normalized cost in [0, 1]. Returns 1.0 if inputs are invalid.
    """
    if verts_2d_1 is None or verts_2d_2 is None:
        return 1.0

    # Filter out NaN points (invalid projections)
    N = min(len(verts_2d_1), len(verts_2d_2))
    p1 = verts_2d_1[:N]
    p2 = verts_2d_2[:N]
    valid = (~np.any(np.isnan(p1), axis=1)) & (~np.any(np.isnan(p2), axis=1))
    if valid.sum() < 1:
        return 1.0

    p1, p2 = p1[valid], p2[valid]
    ones = np.ones((len(p1), 1))
    p1_h = np.hstack([p1, ones])  # (N, 3)
    p2_h = np.hstack([p2, ones])  # (N, 3)

    # Epipolar lines: l2 = F @ p1,  l1 = F^T @ p2
    l2 = (F @ p1_h.T).T   # (N, 3)
    l1 = (F.T @ p2_h.T).T  # (N, 3)

    # Symmetric epipolar distance
    d2 = np.abs(np.sum(p2_h * l2, axis=1)) / (np.sqrt(l2[:, 0]**2 + l2[:, 1]**2) + 1e-8)
    d1 = np.abs(np.sum(p1_h * l1, axis=1)) / (np.sqrt(l1[:, 0]**2 + l1[:, 1]**2) + 1e-8)
    mean_dist = (np.mean(d1) + np.mean(d2)) / 2.0
    if logger:
         logger.info(f'mean_dist:{mean_dist}, N_verts={len(p1)}')

    return float(min(mean_dist / epi_threshold, 1.0))


def triangulate_point_dlt(P1: np.ndarray, P2: np.ndarray,
                          pt1: np.ndarray, pt2: np.ndarray) -> Optional[np.ndarray]:
    """
    Triangulate a 3D point from two undistorted 2D observations using DLT.

    Args:
        P1, P2: (3, 4) projection matrices  K @ [R | t]
        pt1, pt2: (2,) undistorted pixel coordinates

    Returns:
        (3,) 3D point in world coordinates, or None if degenerate.
    """
    A = np.array([
        pt1[0] * P1[2] - P1[0],
        pt1[1] * P1[2] - P1[1],
        pt2[0] * P2[2] - P2[0],
        pt2[1] * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    if abs(X_h[3]) < 1e-10:
        return None
    return X_h[:3] / X_h[3]


def compute_reprojection_error(P: np.ndarray, pt_3d: np.ndarray, pt_2d: np.ndarray) -> float:
    """
    Reproject a 3D point and compute pixel distance to the observed 2D point.

    Args:
        P: (3, 4) projection matrix  K @ [R | t]
        pt_3d: (3,) 3D point in world coordinates
        pt_2d: (2,) undistorted observed pixel coordinates

    Returns:
        Reprojection error in pixels.
    """
    X_h = np.append(pt_3d, 1.0)
    x_h = P @ X_h
    if abs(x_h[2]) < 1e-10:
        return float('inf')
    x_proj = x_h[:2] / x_h[2]
    return float(np.linalg.norm(x_proj - pt_2d))


# ============================================================================
# MULTI-VIEW POSE MATCHING - FILTERING
# ============================================================================

def filter_incomplete_bboxes(bboxes, image_width, image_height,
                             area_ratio=0.5, border_margin=5):
    """Filter out incomplete bboxes (small AND touching border)."""
    if not bboxes:
        return []

    bbox_coords = [b['bbox'] if isinstance(b, dict) else b for b in bboxes]
    if not bbox_coords:
        return []

    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bbox_coords]
    median_area = np.median(areas)

    kept_indices = []
    for idx, ((x1, y1, x2, y2), area) in enumerate(zip(bbox_coords, areas)):
        touches_border = (x1 <= border_margin or y1 <= border_margin or
                         x2 >= image_width - border_margin or
                         y2 >= image_height - border_margin)
        is_small = area < area_ratio * median_area

        if not (is_small or  touches_border):
            kept_indices.append(idx)

    return kept_indices


def _filter_poses_by_quality(poses_world_dict, bbox_score_threshold, filter_incomplete_bbox,
                             bbox_area_ratio, bbox_border_margin, logger):
    """Filter poses by bbox score and completeness."""
    filtered_poses = {}

    for view_name, pose_list in poses_world_dict.items():
        valid_poses = []
        for pose_dict in pose_list:
            score = pose_dict.get('scores', 0.0)
            keypoints = pose_dict.get('keypoints', np.array([]))

            if score < bbox_score_threshold:
                if logger:
                    logger.info(f"    Discarded detection in {view_name} (bbox_score={score:.4f} < {bbox_score_threshold})")
                continue

            if len(keypoints) < 10:
                if logger:
                    logger.info(f"    Discarded detection in {view_name} (keypoints count={len(keypoints)} < 10)")
                continue

            valid_poses.append(pose_dict)

        # Filter incomplete bboxes
        if filter_incomplete_bbox and valid_poses:
            view_image_width = valid_poses[0].get('image_w', None)
            view_image_height = valid_poses[0].get('image_h', None)

            if view_image_width is not None and view_image_height is not None:
                kept_indices = filter_incomplete_bboxes(
                    valid_poses, view_image_width, view_image_height,
                    area_ratio=bbox_area_ratio, border_margin=bbox_border_margin
                )

                if len(kept_indices) < len(valid_poses):
                    if logger:
                        logger.info(f"    Filtered {len(valid_poses) - len(kept_indices)} incomplete bboxes in {view_name}")
                    valid_poses = [valid_poses[i] for i in kept_indices]

        if valid_poses:
            filtered_poses[view_name] = valid_poses
        elif logger:
            logger.info(f"    All detections in {view_name} discarded after filtering")

    return filtered_poses


# ============================================================================
# MULTI-VIEW POSE MATCHING - COST COMPUTATION
# ============================================================================

def _compute_hybrid_cost(kp1, kp2, verts_2d_1, verts_2d_2, F,
                         matching_mode, mpjpe_weight=0.5,
                         max_mpjpe=300.0, epi_threshold=20.0, logger=None):
    """
    Compute matching cost.

    Modes:
        pa_mpjpe:      raw PA-MPJPE (mm), for threshold comparison
        epipolar_only: normalized epipolar distance [0, 1]
        hybrid:        mpjpe_weight * mpjpe_norm + (1-mpjpe_weight) * epi_cost
        epi_gate:      handled externally in _compute_pairwise_matches
    """
    pa_mpjpe = compute_pa_mpjpe(kp1, kp2)

    if matching_mode == 'pa_mpjpe':
        return pa_mpjpe

    mpjpe_norm = min(pa_mpjpe / max_mpjpe, 1.0)
    epi_cost = compute_epipolar_cost(verts_2d_1, verts_2d_2, F, epi_threshold, logger)

    if matching_mode == 'epipolar_only':
        return epi_cost
    elif matching_mode == 'hybrid':
        return mpjpe_weight * mpjpe_norm + (1.0 - mpjpe_weight) * epi_cost
    raise ValueError(f"Unknown matching_mode: {matching_mode}")


def _compute_pairwise_matches(poses_world_dict, view_names, matching_mode,
                              max_error_threshold, camera_params_dict,
                              mpjpe_weight=0.5, epi_threshold=20.0, repr_threshold=30.0, logger=None):
    """Compute pairwise pose matches across all view pairs."""
    pairwise_matches = {}
    num_views = len(view_names)

    if logger:
        logger.info(f"    Computing pairwise pose matches for {num_views} views: {view_names}")
        logger.info(f"    Matching mode: {matching_mode}, mpjpe_weight={mpjpe_weight}, "
                    f"epi_threshold={epi_threshold}px, repr_threshold={repr_threshold}px")

    INVALID_COST = 1e6

    for i in range(num_views):
        for j in range(i + 1, num_views):
            view1, view2 = view_names[i], view_names[j]
            poses1 = poses_world_dict[view1]
            poses2 = poses_world_dict[view2]

            if not poses1 or not poses2:
                if logger:
                    logger.info(f"    {view1} <-> {view2}: Skipped (no poses in one or both views)")
                continue

            # Compute camera matrices for this view pair
            F = None
            P1 = P2 = K1 = K2 = dist1 = dist2 = None
            if matching_mode in ('epipolar_only', 'hybrid', 'epi_gate') and camera_params_dict is not None:
                try:
                    R1, t1, K1 = get_transform_matrix(camera_params_dict[view1])
                    R2, t2, K2 = get_transform_matrix(camera_params_dict[view2])
                    dist1 = get_dist_coeffs(camera_params_dict[view1])
                    dist2 = get_dist_coeffs(camera_params_dict[view2])
                    F = compute_fundamental_matrix(K1, R1, t1, K2, R2, t2)
                    if matching_mode == 'epi_gate':
                        P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
                        P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
                except Exception as e:
                    if logger:
                        logger.info(f"    {view1} <-> {view2}: Failed to compute camera matrices: {e}")

            # Compute cost matrix
            num_p1, num_p2 = len(poses1), len(poses2)
            cost_matrix = np.full((num_p1, num_p2), INVALID_COST)

            epi_gate_rejected = repr_gate_rejected = 0
            for p1_idx in range(num_p1):
                kp1 = poses1[p1_idx]['keypoints']
                verts_2d_1_raw = poses1[p1_idx].get('pred_keypoints_2d', None)
                verts_2d_1 = undistort_points(verts_2d_1_raw, K1, dist1) if (K1 is not None and verts_2d_1_raw is not None) else verts_2d_1_raw

                if matching_mode == 'epi_gate':
                    # Precompute per-person-1 data once
                    mesh_1_raw = poses1[p1_idx].get('pred_keypoints_2d_verts', None)
                    mesh_1 = undistort_points(mesh_1_raw, K1, dist1) if (K1 is not None and mesh_1_raw is not None) else mesh_1_raw

                    bbox1 = poses1[p1_idx].get('bbox', None)
                    if bbox1 is not None:
                        cx1 = (bbox1[0] + bbox1[2]) / 2.0
                        cy1 = (bbox1[1] + bbox1[3]) / 2.0
                        bc1 = undistort_points(np.array([[cx1, cy1]]), K1, dist1) if K1 is not None else np.array([[cx1, cy1]])
                    else:
                        bc1 = None

                for p2_idx in range(num_p2):
                    kp2 = poses2[p2_idx]['keypoints']
                    verts_2d_2_raw = poses2[p2_idx].get('pred_keypoints_2d', None)
                    verts_2d_2 = undistort_points(verts_2d_2_raw, K2, dist2) if (K2 is not None and verts_2d_2_raw is not None) else verts_2d_2_raw

                    if matching_mode == 'epi_gate':
                        # Stage 1: reprojection gate using bbox centers
                        bbox2 = poses2[p2_idx].get('bbox', None)
                        if bc1 is not None and bbox2 is not None and P1 is not None:
                            cx2 = (bbox2[0] + bbox2[2]) / 2.0
                            cy2 = (bbox2[1] + bbox2[3]) / 2.0
                            bc2 = undistort_points(np.array([[cx2, cy2]]), K2, dist2) if K2 is not None else np.array([[cx2, cy2]])
                            pt_3d = triangulate_point_dlt(P1, P2, bc1[0], bc2[0])
                            if pt_3d is not None:
                                err1 = compute_reprojection_error(P1, pt_3d, bc1[0])
                                err2 = compute_reprojection_error(P2, pt_3d, bc2[0])
                                if max(err1, err2) >= repr_threshold:
                                    repr_gate_rejected += 1
                                    cost_matrix[p1_idx, p2_idx] = INVALID_COST
                                    continue

                        # Stage 2: epipolar gate using all mesh vertices
                        mesh_2_raw = poses2[p2_idx].get('pred_keypoints_2d_verts', None)
                        mesh_2 = undistort_points(mesh_2_raw, K2, dist2) if (K2 is not None and mesh_2_raw is not None) else mesh_2_raw

                        epi_cost = compute_epipolar_cost(mesh_1, mesh_2, F, epi_threshold, logger)
                        if F is None or epi_cost >= 1.0:
                            epi_gate_rejected += 1
                            cost_matrix[p1_idx, p2_idx] = INVALID_COST
                            continue

                        # Both gates passed: rank by PA-MPJPE
                        cost_matrix[p1_idx, p2_idx] = compute_pa_mpjpe(kp1, kp2)
                    else:
                        cost_matrix[p1_idx, p2_idx] = _compute_hybrid_cost(
                            kp1, kp2, verts_2d_1, verts_2d_2, F,
                            matching_mode, mpjpe_weight,
                            max_mpjpe=max_error_threshold or 300.0,
                            epi_threshold=epi_threshold,
                            logger=logger
                        )

            if matching_mode == 'epi_gate' and logger:
                total = num_p1 * num_p2
                logger.info(f"    {view1} <-> {view2}: reprojection rejected {repr_gate_rejected}/{total}, "
                            f"epipolar rejected {epi_gate_rejected}/{total - repr_gate_rejected}")

            # Log cost matrix statistics
            valid_costs = cost_matrix[cost_matrix < INVALID_COST]
            if logger and len(valid_costs) > 0:
                logger.info(f"    {view1} <-> {view2}: Cost matrix - {len(valid_costs)}/{num_p1*num_p2} valid costs, min={np.min(valid_costs):.4f}, max={np.max(valid_costs):.4f}")
            elif logger:
                logger.info(f"    {view1} <-> {view2}: Cost matrix - ALL costs are INVALID (>= {INVALID_COST})")

            # Hungarian algorithm for optimal matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Filter and save matches
            matches = []

            for p1_idx, p2_idx in zip(row_ind, col_ind):
                if cost_matrix[p1_idx, p2_idx] >= INVALID_COST:
                    continue

                kp1 = poses1[p1_idx]['keypoints']
                kp2 = poses2[p2_idx]['keypoints']
                pa_mpjpe = compute_pa_mpjpe(kp1, kp2)
                matches.append((p1_idx, p2_idx, pa_mpjpe))

            if matches:
                pairwise_matches[(view1, view2)] = matches
                if logger:
                    max_error = max(m[2] for m in matches)
                    logger.info(f"    {view1} <-> {view2}: {len(matches)} matches (max_mpjpe={max_error:.1f}mm)")
            elif logger:
                logger.info(f"    {view1} <-> {view2}: 0 matches")

    return pairwise_matches


# ============================================================================
# MULTI-VIEW POSE MATCHING - CLUSTERING
# ============================================================================

def _build_person_clusters(pairwise_matches):
    """Build person clusters using Union-Find algorithm."""
    person_to_cluster = {}
    cluster_to_persons = defaultdict(list)
    next_cluster_id = [0]

    def get_cluster(view, person_idx):
        key = (view, person_idx)
        if key not in person_to_cluster:
            person_to_cluster[key] = next_cluster_id[0]
            cluster_to_persons[next_cluster_id[0]].append(key)
            next_cluster_id[0] += 1
        return person_to_cluster[key]

    def merge_clusters(cluster1, cluster2):
        if cluster1 == cluster2:
            return cluster1

        views_in_c1 = set(view for view, _ in cluster_to_persons[cluster1])
        views_in_c2 = set(view for view, _ in cluster_to_persons[cluster2])

        if views_in_c1 & views_in_c2:
            return cluster1

        for person_key in cluster_to_persons[cluster2]:
            person_to_cluster[person_key] = cluster1
            cluster_to_persons[cluster1].append(person_key)

        cluster_to_persons[cluster2] = []
        return cluster1

    # Build clusters from pairwise matches
    for (view1, view2), matches in pairwise_matches.items():
        for p1_idx, p2_idx, _ in matches:
            cluster1 = get_cluster(view1, p1_idx)
            cluster2 = get_cluster(view2, p2_idx)
            merge_clusters(cluster1, cluster2)

    return cluster_to_persons, person_to_cluster


def _filter_and_build_valid_clusters(cluster_to_persons, poses_world_dict, pairwise_matches,
                                     min_views, logger):
    """Filter clusters and build final valid clusters."""
    valid_clusters = []

    for cluster_id, person_list in cluster_to_persons.items():
        if not person_list:
            continue

        views_in_cluster = set(view for view, _ in person_list)
        if len(views_in_cluster) < min_views:
            continue

        # Check for duplicate views
        view_to_person = {}
        duplicate = False
        for view, person_idx in person_list:
            if view in view_to_person:
                duplicate = True
                break
            view_to_person[view] = person_idx

        if duplicate:
            if logger:
                logger.info(f"    Warning: cluster {cluster_id} has duplicate views, skipping")
            continue

        # Build cluster result
        cluster_poses = {}
        for view, person_idx in person_list:
            cluster_poses[view] = poses_world_dict[view][person_idx]

        # Compute average error
        total_error = 0.0
        error_count = 0
        for (view1, view2), matches in pairwise_matches.items():
            if view1 not in view_to_person or view2 not in view_to_person:
                continue
            p1_idx = view_to_person[view1]
            p2_idx = view_to_person[view2]

            for m_p1, m_p2, error in matches:
                if m_p1 == p1_idx and m_p2 == p2_idx:
                    total_error += error
                    error_count += 1
                    break

        avg_error = total_error / error_count if error_count > 0 else 0.0

        valid_clusters.append({
            'poses': cluster_poses,
            'num_views': len(views_in_cluster),
            'avg_error': avg_error
        })

    if logger:
        logger.info(f"    Found {len(valid_clusters)} valid person clusters")
        for i, cluster in enumerate(valid_clusters):
            views = ', '.join(cluster['poses'].keys())
            logger.info(f"      Person {i}: {cluster['num_views']} views ({views}), avg_error={cluster['avg_error']:.1f}mm")

    return valid_clusters


def match_poses_across_views(poses_world_dict_all_people: Dict[str, List[Dict]],
                             min_views: int = 3,
                             max_error_threshold: Optional[float] = None,
                             bbox_score_threshold: float = 0.85,
                             filter_incomplete_bbox: bool = True,
                             bbox_area_ratio: float = 0.5,
                             bbox_border_margin: int = 5,
                             matching_mode: str = 'hybrid',
                             camera_params_dict: Optional[Dict] = None,
                             mpjpe_weight: float = 0.5,
                             epi_threshold: float = 20.0,
                             repr_threshold: float = 30.0,
                             logger=None) -> List[Dict]:
    """
    Match person poses across views using Hungarian algorithm.

    matching_mode:
        'pa_mpjpe'      — only PA-MPJPE distance
        'epipolar_only' — only epipolar geometry distance
        'hybrid'        — mpjpe_weight * PA-MPJPE + (1-mpjpe_weight) * epipolar
        'epi_gate'      — two-stage hard gate then PA-MPJPE ranking:
                            1. epipolar gate: mean vertex epipolar dist >= epi_threshold px → reject
                            2. reprojection gate: bbox-center reprojection error >= repr_threshold px → reject
    """
    if logger:
        logger.info(f"    Starting cross-view matching with min_views={min_views}, bbox_score_threshold={bbox_score_threshold}")

    # Filter poses by quality
    poses_world_dict_all_people = _filter_poses_by_quality(
        poses_world_dict_all_people, bbox_score_threshold, filter_incomplete_bbox,
        bbox_area_ratio, bbox_border_margin, logger
    )

    view_names = sorted(list(poses_world_dict_all_people.keys()))
    num_views = len(view_names)

    if num_views < 2:
        if logger:
            logger.info(f"    Only {num_views} view(s) available, skipping matching")
        return []

    # Compute pairwise matches
    pairwise_matches = _compute_pairwise_matches(
        poses_world_dict_all_people, view_names, matching_mode,
        max_error_threshold, camera_params_dict,
        mpjpe_weight=mpjpe_weight, epi_threshold=epi_threshold, repr_threshold=repr_threshold, logger=logger
    )

    if not pairwise_matches:
        if logger:
            logger.info(f"    No pairwise matches found")
        return []

    # Build person clusters
    cluster_to_persons, person_to_cluster = _build_person_clusters(pairwise_matches)

    # Filter and build valid clusters
    return _filter_and_build_valid_clusters(
        cluster_to_persons, poses_world_dict_all_people, pairwise_matches, min_views, logger
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_prediction_npz(npz_path: str, person_idx: int = 0) -> Dict:
    """Load prediction from npz file."""
    data = np.load(npz_path, allow_pickle=True)

    if 'outputs' not in data:
        raise ValueError(f"'outputs' key not found in {npz_path}")

    outputs = data['outputs']
    if len(outputs) == 0:
        raise ValueError(f"No detections found in {npz_path}")
    if person_idx >= len(outputs):
        raise ValueError(f"Person index {person_idx} out of range. Total detections: {len(outputs)}")

    person_data = outputs[person_idx]
    frame_dict = person_data.item() if hasattr(person_data, 'item') and isinstance(person_data.item(), dict) else person_data

    if not isinstance(frame_dict, dict):
        raise ValueError(f"Unexpected format for outputs[{person_idx}]: {type(person_data)}")

    required_fields = ['pred_keypoints_3d', 'bbox_score', 'pred_cam_t']
    for field in required_fields:
        if field not in frame_dict:
            raise ValueError(f"'{field}' not found in outputs[{person_idx}]")

    keypoints3d = np.array(frame_dict['pred_keypoints_3d'])
    if keypoints3d.shape[0] > 21:
        keypoints3d = keypoints3d[KEYPOINTS_IDX]

    # Also select 2D keypoints using the same indices
    pred_keypoints_2d = frame_dict.get('pred_keypoints_2d', None)
    if pred_keypoints_2d is not None:
        pred_keypoints_2d = np.array(pred_keypoints_2d)
        if pred_keypoints_2d.shape[0] > 21:
            pred_keypoints_2d = pred_keypoints_2d[KEYPOINTS_IDX]

    pred_keypoints_2d_verts = frame_dict.get('pred_keypoints_2d_verts', None)
    if pred_keypoints_2d_verts is not None:
        pred_keypoints_2d_verts = np.array(pred_keypoints_2d_verts)

    return {
        'keypoints3d': keypoints3d,
        'bbox_score': frame_dict['bbox_score'],
        'pred_cam_t': np.array(frame_dict['pred_cam_t']),
        'bbox': frame_dict.get('bbox', None),
        'pred_keypoints_2d': pred_keypoints_2d,
        'pred_keypoints_2d_verts': pred_keypoints_2d_verts,
        'pred_vertices': frame_dict.get('pred_vertices', None)
    }


def find_frame_image(folder: str, frame_num: int) -> Optional[str]:
    """Find the image file for a given frame number by matching the filename stem."""
    IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    if not os.path.isdir(folder):
        return None
    for f in os.listdir(folder):
        name, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXTS and name == str(frame_num):
            return os.path.join(folder, f)
    return None


def load_view_data(view_name: str, frame_num: int, output_dir: str, scene_dir: str,
                   camera_params_dict: Dict) -> Tuple[str, Optional[List[Dict]]]:
    """Load all person detections for a single view."""
    # npz_path = os.path.join(output_dir, view_name, 'npz', f'{frame_num:06d}.npz')
    npz_path = os.path.join(output_dir, view_name, 'npz', f'{frame_num}.npz')

    print(npz_path)

    if not os.path.exists(npz_path):
        return view_name, None

    try:
        data = np.load(npz_path, allow_pickle=True)
        if 'outputs' not in data:
            return view_name, None
        num_detections = len(data['outputs'])

        # Get image dimensions
        img_path = find_frame_image(os.path.join(scene_dir, view_name), frame_num)
        image_width, image_height = None, None
        if img_path is not None:
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    image_width, image_height = img.size
            except:
                pass

        # Get camera parameters
        cam_params = camera_params_dict[view_name]
        R, t, _ = get_transform_matrix(cam_params)

        # Process all persons in this view
        view_poses = []
        for person_idx in range(num_detections):
            try:
                pred_data = load_prediction_npz(npz_path, person_idx=person_idx)
                keypoints3d_cam = pred_data['keypoints3d']
                pred_keypoints_2d = pred_data.get('pred_keypoints_2d', None)

                if len(keypoints3d_cam.shape) != 2 or keypoints3d_cam.shape[1] != 3:
                    continue

                keypoints3d_cam_trans = keypoints3d_cam + pred_data['pred_cam_t']
                keypoints3d_world = transform_to_world(keypoints3d_cam_trans, R, t)

                view_poses.append({
                    'keypoints': keypoints3d_world,
                    'scores': pred_data['bbox_score'],
                    'bbox': pred_data.get('bbox', None),
                    'pred_keypoints_2d': pred_keypoints_2d,
                    'pred_keypoints_2d_verts': pred_data.get('pred_keypoints_2d_verts', None),
                    'image_w': image_width,
                    'image_h': image_height,
                })

            except Exception as e:
                print(f"Warning: Failed to process person {person_idx} in {view_name}: {e}")
                continue

        return view_name, view_poses if view_poses else None

    except Exception as e:
        print(f"Error loading view {view_name}: {e}")
        return view_name, None


# ============================================================================
# VISUALIZATION
# ============================================================================

def _draw_bbox_with_label(img: np.ndarray, bbox, label: str, color, font_scale=0.6, thickness=2):
    """Draw bbox and label on image."""
    if bbox is None or len(bbox) < 4:
        return

    import cv2
    x1, y1, x2, y2 = map(int, bbox[:4])

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - baseline - 5), font, font_scale, (255, 255, 255), thickness)


def visualize_matched_clusters(matched_clusters: List[Dict], frame_num: int, scene_dir: str,
                               output_dir: str, n_views: int = 8, logger=None):
    """Visualize matched person clusters by drawing bboxes on images."""
    import cv2

    os.makedirs(output_dir, exist_ok=True)

    if logger:
        logger.info(f"  Visualizing {len(matched_clusters)} matched clusters for frame {frame_num}")

    # Load images for each view
    view_images = {}
    for view_idx in range(n_views):
        # view_name = f'ace_{view_idx}'
        view_name = f'camera_{view_idx}'
        img_path = find_frame_image(os.path.join(scene_dir, view_name), frame_num)

        if img_path is None:
            if logger:
                logger.info(f"    Warning: Image not found for frame {frame_num} in {os.path.join(scene_dir, view_name)}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            if logger:
                logger.info(f"    Warning: Failed to load image: {img_path}")
            continue

        view_images[view_name] = img

    if not view_images:
        if logger:
            logger.info(f"    Warning: No images loaded, skipping visualization")
        return

    # Color palette
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 128, 0), (0, 128, 255), (128, 255, 0)
    ]

    # Create visualizations for each person
    for person_idx, cluster in enumerate(matched_clusters):
        poses = cluster['poses']
        matched_views = sorted([view for view in poses.keys() if view in view_images])

        if not matched_views:
            continue

        person_view_images = []
        color = colors[person_idx % len(colors)]

        for view_name in matched_views:
            img_copy = view_images[view_name].copy()
            bbox = poses[view_name].get('bbox', None)
            score = poses[view_name].get('scores', 0.0)
            label = f'P{person_idx} {view_name} ({score:.2f})'

            _draw_bbox_with_label(img_copy, bbox, label, color)
            person_view_images.append(img_copy)

        if person_view_images:
            # Resize to same height and concatenate
            max_height = max(img.shape[0] for img in person_view_images)
            resized_images = []
            for img in person_view_images:
                if img.shape[0] != max_height:
                    scale = max_height / img.shape[0]
                    new_width = int(img.shape[1] * scale)
                    resized_images.append(cv2.resize(img, (new_width, max_height)))
                else:
                    resized_images.append(img)

            concatenated = cv2.hconcat(resized_images)

            frame_folder = os.path.join(output_dir, f'frame_{frame_num:06d}')
            os.makedirs(frame_folder, exist_ok=True)
            output_path = os.path.join(frame_folder, f'person_{person_idx}_concat.png')
            cv2.imwrite(output_path, concatenated)
            if logger:
                logger.info(f"    Saved visualization for person {person_idx}: {output_path} ({len(matched_views)} views)")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Part 1: Multi-view pose matching and clustering')

    # I/O paths
    parser.add_argument('--output_dir', type=str,
                    #    default='/home/y_li/workspace7/sam-3d-body/output/sportscenter_21380/',
                       default='/work7/y_li/sam-3d-body/output/humanm3/basketball1/split1',
                       help='Directory containing predictions from all views')
    parser.add_argument('--camera_param_dir', type=str,
                    #    default='/home/y_li/workspace7/xrmocap/xrmocap_data/xrmocap_sportscenter/xrmocap_meta_sportscenter/scene_0/camera_parameters',
                       default='/home/y_li/workspace7/humanm3/test/basketball1/split1/camera_calibration',
                       help='Directory containing camera parameters')
    parser.add_argument('--scene_dir', type=str,
                    #    default='/home/y_li/workspace7/xrmocap/xrmocap_data/xrmocap_sportscenter/scene_0',
                       default='/home/y_li/workspace7/humanm3/test/basketball1/split1/images',
                       help='Scene directory for loading images')
    parser.add_argument('--intermediate_output', type=str,
                    #    default='/home/y_li/workspace7/sam-3d-body/output/sportscenter_21380/intermediate_matched_clusters.pkl',
                       default='/work7/y_li/sam-3d-body/output/humanm3/basketball1/split1/intermediate_matched_clusters.pkl',
                       help='Output path for intermediate matched clusters')

    # Matching parameters
    parser.add_argument('--min_views_cluster', type=int, default=2,
                       help='Minimum views required for valid cluster')
    parser.add_argument('--max_pa_mpjpe_threshold', type=float, default=100,
                       help='Maximum PA_MPJPE threshold for matching (mm)')
    parser.add_argument('--bbox_score_threshold', type=float, default=0.7,
                       help='Discard detections with bbox_score below this value')
    parser.add_argument('--matching_mode', type=str, default='pa_mpjpe',
                       choices=['pa_mpjpe', 'epipolar_only', 'hybrid', 'epi_gate'],
                       help='Matching mode: pa_mpjpe / epipolar_only / hybrid / epi_gate')
    parser.add_argument('--mpjpe_weight', type=float, default=0.3,
                       help='Weight of PA-MPJPE in hybrid mode (epipolar weight = 1 - mpjpe_weight)')
    parser.add_argument('--epi_threshold', type=float, default=5.0,
                       help='Epipolar distance threshold in pixels (epi_gate: mean vertex epipolar dist >= this → reject)')
    parser.add_argument('--repr_threshold', type=float, default=30.0,
                       help='Reprojection error threshold in pixels (epi_gate: bbox-center reprojection error >= this → reject)')

    # Visualization
    parser.add_argument('--visualize_matches', action='store_true', default=True,
                       help='Visualize matched clusters')

    args = parser.parse_args()

    # Setup logger
    logger, log_file = setup_logger(args.output_dir, 'part1_matching')
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    logger.info("Part 1: Multi-view Pose Matching and Clustering")
    logger.info("="*80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Intermediate output: {args.intermediate_output}")
    logger.info(f"Min views for cluster: {args.min_views_cluster}")
    logger.info(f"Max PA-MPJPE threshold: {'No limit' if args.max_pa_mpjpe_threshold is None else f'{args.max_pa_mpjpe_threshold}mm'}")
    logger.info(f"Matching mode: {args.matching_mode}")
    logger.info("")

    # Define view names
    # view_names = [f'ace_{i}' for i in range(6)]
    view_names = [f'camera_{i}' for i in range(4)]

    # Load camera parameters
    logger.info("Loading camera parameters...")
    camera_params_dict = {}
    for i, view_name in enumerate(view_names):
        param_file = os.path.join(args.camera_param_dir, f'fisheye_param_{i:02d}.json')
        # param_file = os.path.join(args.camera_param_dir, f'camera_{i}.json')
        camera_params_dict[view_name] = load_camera_params(param_file)
        logger.info(f"  Loaded {view_name}: {param_file}")

    # Get frame list
    # ace_0_npz_dir = os.path.join(args.output_dir, 'ace_0', 'npz')
    ace_0_npz_dir = os.path.join(args.output_dir, 'camera_0', 'npz')
    if not os.path.exists(ace_0_npz_dir):
        raise FileNotFoundError(f"NPZ directory not found: {ace_0_npz_dir}")

    frame_files = sorted([f for f in os.listdir(ace_0_npz_dir) if f.endswith('.npz')])
    frame_numbers = [int(f.split('.')[0]) for f in frame_files]
    logger.info(f"\nProcessing {len(frame_numbers)} frames...")

    # Process each frame and collect matched clusters
    all_frame_data = []

    for frame_idx, frame_num in enumerate(frame_numbers):
        logger.info(f"\n--- Frame {frame_num} ({frame_idx + 1}/{len(frame_numbers)}) ---")

        poses_world_dict_all_people = {}

        # Load all views
        with ThreadPoolExecutor(max_workers=len(view_names)) as executor:
            future_to_view = {
                executor.submit(load_view_data, view_name, frame_num, args.output_dir,
                               args.scene_dir, camera_params_dict): view_name
                for view_name in view_names
            }

            for future in as_completed(future_to_view):
                view_name = future_to_view[future]
                try:
                    view_name_result, view_poses = future.result()
                    if view_poses is not None:
                        poses_world_dict_all_people[view_name_result] = view_poses
                        logger.info(f"  {view_name_result}: {len(view_poses)} person(s)")
                except Exception as e:
                    logger.info(f"  Warning: Failed to process {view_name}: {e}")

        if not poses_world_dict_all_people:
            logger.info(f"  No valid predictions, skipping")
            continue

        # Match poses across views
        logger.info(f"  Performing cross-view matching...")
        matched_clusters = match_poses_across_views(
            poses_world_dict_all_people,
            min_views=args.min_views_cluster,
            max_error_threshold=args.max_pa_mpjpe_threshold,
            bbox_score_threshold=args.bbox_score_threshold,
            matching_mode=args.matching_mode,
            camera_params_dict=camera_params_dict,
            mpjpe_weight=args.mpjpe_weight,
            epi_threshold=args.epi_threshold,
            repr_threshold=args.repr_threshold,
            logger=logger
        )

        if not matched_clusters:
            logger.info(f"  No matched clusters found")
            continue

        logger.info(f"  Found {len(matched_clusters)} matched person(s)")

        # Store frame data
        all_frame_data.append({
            'frame_idx': frame_idx,
            'frame_num': frame_num,
            'matched_clusters': matched_clusters,
            'num_persons': len(matched_clusters)
        })

        # Visualize if requested
        if args.visualize_matches:
            vis_output_dir = os.path.join(args.output_dir, 'matched_visualizations')
            visualize_matched_clusters(
                matched_clusters=matched_clusters,
                frame_num=frame_num,
                scene_dir=args.scene_dir,
                output_dir=vis_output_dir,
                n_views=6,
                logger=logger
            )

    # Save intermediate results
    logger.info(f"\n{'='*80}")
    logger.info(f"Saving intermediate results to {args.intermediate_output}")

    intermediate_data = {
        'all_frame_data': all_frame_data,
        'camera_params_dict': camera_params_dict,
        'args': vars(args)
    }

    os.makedirs(os.path.dirname(args.intermediate_output), exist_ok=True)
    with open(args.intermediate_output, 'wb') as f:
        pickle.dump(intermediate_data, f)

    logger.info(f"Saved {len(all_frame_data)} frames with matched clusters")
    logger.info(f"Total persons across all frames: {sum(fd['num_persons'] for fd in all_frame_data)}")

    # Save matching results as JSON
    matching_json_path = os.path.join(args.output_dir, 'matching_results.json')
    logger.info(f"\nSaving matching results to {matching_json_path}")

    matching_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_frames': len(all_frame_data),
            'total_persons': sum(fd['num_persons'] for fd in all_frame_data),
            'min_views_cluster': args.min_views_cluster,
            'max_pa_mpjpe_threshold': args.max_pa_mpjpe_threshold,
            'matching_mode': args.matching_mode,
            'bbox_score_threshold': args.bbox_score_threshold
        },
        'frames': []
    }

    for frame_data in all_frame_data:
        frame_info = {
            'frame_idx': frame_data['frame_idx'],
            'frame_num': frame_data['frame_num'],
            'num_persons': frame_data['num_persons'],
            'persons': []
        }

        for person_idx, cluster in enumerate(frame_data['matched_clusters']):
            person_info = {
                'person_idx': person_idx,
                'num_views': cluster['num_views'],
                'avg_error': float(cluster.get('avg_error', 0.0)),
                'views': []
            }

            for view_name, view_data in cluster['poses'].items():
                view_info = {
                    'view_name': view_name,
                    'bbox_score': float(view_data.get('scores', 0.0)),
                    'has_keypoints': 'keypoints' in view_data
                }
                person_info['views'].append(view_info)

            frame_info['persons'].append(person_info)

        matching_results['frames'].append(frame_info)

    with open(matching_json_path, 'w') as f:
        json.dump(matching_results, f, indent=2)

    logger.info(f"Saved matching results with {len(all_frame_data)} frames")
    logger.info("="*80)
    logger.info("Part 1 Complete!")


if __name__ == '__main__':
    main()
