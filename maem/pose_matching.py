"""Cross-view pose matching: cost computation and pairwise matching."""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .camera_utils import get_transform_matrix, get_dist_coeffs, undistort_points
from .epipolar_geometry import (compute_fundamental_matrix, compute_epipolar_cost,
                                triangulate_point_dlt, compute_reprojection_error)
from .metrics import compute_pa_mpjpe


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
        if not (is_small or touches_border):
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
                    logger.info(f"    Discarded detection in {view_name} "
                                f"(bbox_score={score:.4f} < {bbox_score_threshold})")
                continue
            if len(keypoints) < 10:
                if logger:
                    logger.info(f"    Discarded detection in {view_name} "
                                f"(keypoints count={len(keypoints)} < 10)")
                continue
            valid_poses.append(pose_dict)

        if filter_incomplete_bbox and valid_poses:
            w = valid_poses[0].get('image_w')
            h = valid_poses[0].get('image_h')
            if w is not None and h is not None:
                kept = filter_incomplete_bboxes(valid_poses, w, h,
                                               area_ratio=bbox_area_ratio,
                                               border_margin=bbox_border_margin)
                if len(kept) < len(valid_poses) and logger:
                    logger.info(f"    Filtered {len(valid_poses) - len(kept)} "
                                f"incomplete bboxes in {view_name}")
                valid_poses = [valid_poses[i] for i in kept]

        if valid_poses:
            filtered_poses[view_name] = valid_poses
        elif logger:
            logger.info(f"    All detections in {view_name} discarded after filtering")

    return filtered_poses


def _compute_hybrid_cost(kp1, kp2, verts_2d_1, verts_2d_2, F,
                         matching_mode, mpjpe_weight=0.5,
                         max_mpjpe=300.0, epi_threshold=10.0, logger=None):
    """Compute matching cost for hybrid / epipolar_only / pa_mpjpe modes.

    Modes:
        pa_mpjpe:      raw PA-MPJPE (mm)
        epipolar_only: normalized epipolar distance [0, 1]
        hybrid:        mpjpe_weight * mpjpe_norm + (1-mpjpe_weight) * epi_cost
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
                              mpjpe_weight=0.5, epi_threshold=10.0,
                              repr_threshold=30.0, logger=None):
    """Compute pairwise pose matches across all view pairs using Hungarian algorithm.

    Returns:
        pairwise_matches: dict {(view1, view2): [(p1_idx, p2_idx, pa_mpjpe), ...]}
        gate_time_sec:    cumulative time spent in Stage 1 + Stage 2 gates
    """
    pairwise_matches = {}
    num_views = len(view_names)
    gate_time_sec = 0.0
    INVALID_COST = 1e6

    if logger:
        logger.info(f"    Matching mode: {matching_mode}, mpjpe_weight={mpjpe_weight}, "
                    f"epi_threshold={epi_threshold}px, repr_threshold={repr_threshold}px")

    for i in range(num_views):
        for j in range(i + 1, num_views):
            view1, view2 = view_names[i], view_names[j]
            poses1 = poses_world_dict.get(view1, [])
            poses2 = poses_world_dict.get(view2, [])

            if not poses1 or not poses2:
                if logger:
                    logger.info(f"    {view1} <-> {view2}: Skipped (no poses)")
                continue

            F = P1 = P2 = K1 = K2 = dist1 = dist2 = None
            if matching_mode in ('epipolar_only', 'hybrid', 'epi_gate',
                                  'repr_gate', 'epi_gate_only') and camera_params_dict:
                try:
                    R1, t1, K1 = get_transform_matrix(camera_params_dict[view1])
                    R2, t2, K2 = get_transform_matrix(camera_params_dict[view2])
                    dist1 = get_dist_coeffs(camera_params_dict[view1])
                    dist2 = get_dist_coeffs(camera_params_dict[view2])
                    F = compute_fundamental_matrix(K1, R1, t1, K2, R2, t2)
                    if matching_mode in ('epi_gate', 'repr_gate'):
                        P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
                        P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
                except Exception as e:
                    if logger:
                        logger.info(f"    {view1} <-> {view2}: Camera matrix error: {e}")

            num_p1, num_p2 = len(poses1), len(poses2)
            cost_matrix = np.full((num_p1, num_p2), INVALID_COST)
            epi_gate_rejected = repr_gate_rejected = 0

            for p1_idx in range(num_p1):
                kp1 = poses1[p1_idx]['keypoints']
                verts_2d_1_raw = poses1[p1_idx].get('pred_keypoints_2d')
                verts_2d_1 = undistort_points(verts_2d_1_raw, K1, dist1) \
                    if (K1 is not None and verts_2d_1_raw is not None) else verts_2d_1_raw

                if matching_mode in ('epi_gate', 'repr_gate', 'epi_gate_only'):
                    mesh_1_raw = poses1[p1_idx].get('pred_keypoints_2d_verts')
                    mesh_1 = undistort_points(mesh_1_raw, K1, dist1) \
                        if (K1 is not None and mesh_1_raw is not None) else mesh_1_raw
                    bbox1 = poses1[p1_idx].get('bbox')
                    if bbox1 is not None:
                        cx1 = (bbox1[0] + bbox1[2]) / 2.0
                        cy1 = (bbox1[1] + bbox1[3]) / 2.0
                        bc1 = undistort_points(np.array([[cx1, cy1]]), K1, dist1) \
                            if K1 is not None else np.array([[cx1, cy1]])
                    else:
                        bc1 = None

                for p2_idx in range(num_p2):
                    kp2 = poses2[p2_idx]['keypoints']
                    verts_2d_2_raw = poses2[p2_idx].get('pred_keypoints_2d')
                    verts_2d_2 = undistort_points(verts_2d_2_raw, K2, dist2) \
                        if (K2 is not None and verts_2d_2_raw is not None) else verts_2d_2_raw

                    if matching_mode in ('epi_gate', 'repr_gate', 'epi_gate_only'):
                        _t0 = time.perf_counter()

                        if matching_mode in ('epi_gate', 'repr_gate'):
                            bbox2 = poses2[p2_idx].get('bbox')
                            if bc1 is None or bbox2 is None or P1 is None:
                                repr_gate_rejected += 1
                                gate_time_sec += time.perf_counter() - _t0
                                continue
                            cx2 = (bbox2[0] + bbox2[2]) / 2.0
                            cy2 = (bbox2[1] + bbox2[3]) / 2.0
                            bc2 = undistort_points(np.array([[cx2, cy2]]), K2, dist2) \
                                if K2 is not None else np.array([[cx2, cy2]])
                            pt_3d = triangulate_point_dlt(P1, P2, bc1[0], bc2[0])
                            if pt_3d is None or max(
                                compute_reprojection_error(P1, pt_3d, bc1[0]),
                                compute_reprojection_error(P2, pt_3d, bc2[0])
                            ) >= repr_threshold:
                                repr_gate_rejected += 1
                                gate_time_sec += time.perf_counter() - _t0
                                continue

                        if matching_mode in ('epi_gate', 'epi_gate_only'):
                            mesh_2_raw = poses2[p2_idx].get('pred_keypoints_2d_verts')
                            mesh_2 = undistort_points(mesh_2_raw, K2, dist2) \
                                if (K2 is not None and mesh_2_raw is not None) else mesh_2_raw
                            epi_cost = compute_epipolar_cost(mesh_1, mesh_2, F, epi_threshold, logger)
                            if F is None or epi_cost >= 1.0:
                                epi_gate_rejected += 1
                                gate_time_sec += time.perf_counter() - _t0
                                continue

                        gate_time_sec += time.perf_counter() - _t0
                        cost_matrix[p1_idx, p2_idx] = compute_pa_mpjpe(kp1, kp2)
                    else:
                        cost_matrix[p1_idx, p2_idx] = _compute_hybrid_cost(
                            kp1, kp2, verts_2d_1, verts_2d_2, F,
                            matching_mode, mpjpe_weight,
                            max_mpjpe=max_error_threshold or 300.0,
                            epi_threshold=epi_threshold,
                            logger=logger
                        )

            if matching_mode in ('epi_gate', 'repr_gate', 'epi_gate_only') and logger:
                total = num_p1 * num_p2
                logger.info(f"    {view1} <-> {view2}: reproj_gate rejected {repr_gate_rejected}/{total}, "
                            f"epi_gate rejected {epi_gate_rejected}/{total - repr_gate_rejected}")

            valid_costs = cost_matrix[cost_matrix < INVALID_COST]
            if logger and len(valid_costs) > 0:
                logger.info(f"    {view1} <-> {view2}: {len(valid_costs)}/{num_p1*num_p2} valid costs, "
                            f"min={np.min(valid_costs):.4f}, max={np.max(valid_costs):.4f}")
            elif logger:
                logger.info(f"    {view1} <-> {view2}: ALL costs INVALID")

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matches = []
            for p1_idx, p2_idx in zip(row_ind, col_ind):
                if cost_matrix[p1_idx, p2_idx] >= INVALID_COST:
                    continue
                pa_mpjpe = compute_pa_mpjpe(poses1[p1_idx]['keypoints'], poses2[p2_idx]['keypoints'])
                matches.append((p1_idx, p2_idx, pa_mpjpe))

            if matches:
                pairwise_matches[(view1, view2)] = matches
                if logger:
                    logger.info(f"    {view1} <-> {view2}: {len(matches)} matches "
                                f"(max_mpjpe={max(m[2] for m in matches):.1f}mm)")
            elif logger:
                logger.info(f"    {view1} <-> {view2}: 0 matches")

    return pairwise_matches, gate_time_sec
