"""Triangulation: RANSAC per-keypoint DLT and xrmocap AniposelibTriangulator."""

import json
import os
from itertools import combinations
from typing import Dict, List, Optional

import cv2
import numpy as np


# ============================================================================
# Camera loading helpers (for triangulation)
# ============================================================================

def load_projection_matrices(camera_params_dict: Dict, camera_param_dir: str):
    """Load projection matrices P = K @ [R | t] from JSON files.

    Supports two naming conventions:
        fisheye_param_{i:02d}.json  (SportCenter)
        camera_{i}.json             (Human-M3)

    Returns:
        proj_matrices: dict view_name -> P (3, 4)
        K_dict:        dict view_name -> K (3, 3)
        dist_dict:     dict view_name -> dist coefficients (8,)
    """
    proj_matrices, K_dict, dist_dict = {}, {}, {}

    for view_name in sorted(camera_params_dict.keys()):
        view_idx = int(view_name.split('_')[1])
        param_file = os.path.join(camera_param_dir, f'fisheye_param_{view_idx:02d}.json')
        if not os.path.exists(param_file):
            param_file = os.path.join(camera_param_dir, f'camera_{view_idx}.json')

        with open(param_file) as f:
            d = json.load(f)

        K = np.array(d['intrinsic'], dtype=np.float64)[:3, :3]
        if 'extrinsic_r' in d and 'extrinsic_t' in d:
            R = np.array(d['extrinsic_r'], dtype=np.float64)
            t = np.array(d['extrinsic_t'], dtype=np.float64)
        else:
            extr = np.array(d['extrinsic'], dtype=np.float64)
            R, t = extr[:3, :3], extr[:3, 3]

        P = K @ np.hstack([R, t.reshape(3, 1)])
        dist = np.array([d.get(k, 0.0) for k in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')],
                        dtype=np.float64)
        proj_matrices[view_name] = P
        K_dict[view_name] = K
        dist_dict[view_name] = dist

    return proj_matrices, K_dict, dist_dict


def load_camera_parameters_xrmocap(camera_params_dict: Dict, camera_param_dir: str):
    """Load camera parameters as a list of FisheyeCameraParameter objects for xrmocap.

    Returns:
        List[FisheyeCameraParameter] in sorted view order
    """
    from xrprimer.data_structure.camera import FisheyeCameraParameter

    camera_params = []
    for view_name in sorted(camera_params_dict.keys()):
        view_idx = int(view_name.split('_')[1])
        param_file = os.path.join(camera_param_dir, f'fisheye_param_{view_idx:02d}.json')
        if not os.path.exists(param_file):
            param_file = os.path.join(camera_param_dir, f'camera_{view_idx}.json')

        with open(param_file) as f:
            d = json.load(f)

        K = np.array(d['intrinsic'], dtype=np.float64)[:3, :3]
        if 'extrinsic_r' in d and 'extrinsic_t' in d:
            R = np.array(d['extrinsic_r'], dtype=np.float64)
            t = np.array(d['extrinsic_t'], dtype=np.float64).ravel()
        else:
            extr = np.array(d['extrinsic'], dtype=np.float64)
            R, t = extr[:3, :3], extr[:3, 3]

        cam = FisheyeCameraParameter(name=view_name)
        cam.set_intrinsic(mat3x3=K.tolist())
        cam.extrinsic_r = R.tolist()
        cam.extrinsic_t = t.tolist()
        cam.world2cam = True
        for key in ('k1', 'k2', 'p1', 'p2', 'k3'):
            if key in d:
                setattr(cam, key, float(d[key]))

        camera_params.append(cam)

    return camera_params


# ============================================================================
# RANSAC triangulation
# ============================================================================

def _dlt_triangulate_point(pts2d: np.ndarray, P_list: List[np.ndarray],
                           weights: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Confidence-weighted DLT triangulation for one 3D point from N >= 2 views.

    Each view's pair of DLT equations is scaled by sqrt(weight) before the SVD,
    so higher-confidence views (per-view detection score) pull the solution harder.
    """
    A = []
    for i, ((x, y), P) in enumerate(zip(pts2d, P_list)):
        w = 1.0 if weights is None else max(float(weights[i]), 1e-6) ** 0.5
        A.append(w * (x * P[2] - P[0]))
        A.append(w * (y * P[2] - P[1]))
    _, _, Vt = np.linalg.svd(np.array(A))
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


def _triangulate_point_ransac(pts2d_all: np.ndarray, P_all: List[np.ndarray],
                              valid_mask: np.ndarray,
                              reproj_threshold: float = 20.0,
                              weights_all: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Per-keypoint RANSAC triangulation.

    Enumerates all view-pair hypotheses, picks the one with the most inliers,
    then re-triangulates from all inlier views via confidence-weighted DLT.
    Inlier consensus search itself is unweighted (purely geometric); only the
    final triangulation math uses weights_all (per-view detection confidence).
    """
    valid_idx = np.where(valid_mask)[0]
    n_valid = len(valid_idx)
    if n_valid < 2:
        return None

    def _w(idx):
        return weights_all[idx] if weights_all is not None else None

    if n_valid == 2:
        return _dlt_triangulate_point(pts2d_all[valid_idx], [P_all[i] for i in valid_idx],
                                      weights=_w(valid_idx))

    best_pt3d = best_inlier_idx = None
    best_n_inliers = -1
    best_mean_err = float('inf')

    for pair in combinations(valid_idx, 2):
        pt3d = _dlt_triangulate_point(pts2d_all[list(pair)], [P_all[i] for i in pair],
                                      weights=_w(list(pair)))
        if pt3d is None or not np.all(np.isfinite(pt3d)):
            continue
        errors = [_reproj_error(pt3d, pts2d_all[i], P_all[i]) for i in valid_idx]
        inlier_flags = [e < reproj_threshold for e in errors]
        n_inliers = sum(inlier_flags)
        mean_err = float(np.mean([e for e, ok in zip(errors, inlier_flags) if ok])) \
                   if n_inliers > 0 else float('inf')
        if n_inliers > best_n_inliers or (n_inliers == best_n_inliers and mean_err < best_mean_err):
            best_n_inliers = n_inliers
            best_mean_err = mean_err
            best_pt3d = pt3d
            best_inlier_idx = [valid_idx[k] for k, ok in enumerate(inlier_flags) if ok]

    if best_inlier_idx and len(best_inlier_idx) >= 2:
        refined = _dlt_triangulate_point(pts2d_all[best_inlier_idx],
                                         [P_all[i] for i in best_inlier_idx],
                                         weights=_w(best_inlier_idx))
        if refined is not None and np.all(np.isfinite(refined)):
            return refined

    if best_pt3d is not None:
        return best_pt3d
    return _dlt_triangulate_point(pts2d_all[valid_idx], [P_all[i] for i in valid_idx],
                                  weights=_w(valid_idx))


def triangulate_cluster_ransac(matched_cluster: Dict, view_names_sorted: List[str],
                               proj_matrices: Dict[str, np.ndarray],
                               K_dict: Optional[Dict] = None,
                               dist_dict: Optional[Dict] = None,
                               reproj_threshold: float = 20.0,
                               n_keypoints: int = 13,
                               logger=None) -> Optional[np.ndarray]:
    """Triangulate all keypoints for one cluster using per-keypoint RANSAC.

    Each view is weighted by its bbox detection confidence (the only per-detection
    confidence this pipeline produces) in the final DLT solve.

    Returns:
        (n_keypoints, 3) array or None
    """
    poses_dict = matched_cluster['poses']
    n_views = len(view_names_sorted)

    pts2d_all = np.zeros((n_views, n_keypoints, 2), dtype=np.float64)
    valid_mask = np.zeros((n_views, n_keypoints), dtype=bool)
    P_all = [proj_matrices[v] for v in view_names_sorted]
    weights_all = np.ones(n_views, dtype=np.float64)

    for vi, view_name in enumerate(view_names_sorted):
        if view_name not in poses_dict:
            continue
        weights_all[vi] = float(poses_dict[view_name].get('scores', 1.0))
        kpts = poses_dict[view_name].get('pred_keypoints_2d')
        if kpts is None or kpts.shape[1] < 2:
            continue

        valid_ki = [ki for ki in range(n_keypoints)
                    if not (np.all(kpts[ki, :2] == 0) or not np.all(np.isfinite(kpts[ki, :2])))]
        raw_pts = [kpts[ki, :2] for ki in valid_ki]

        if raw_pts and K_dict is not None and dist_dict is not None:
            K = K_dict[view_name]
            dist = dist_dict[view_name]
            pts_arr = np.array(raw_pts, dtype=np.float32).reshape(-1, 1, 2)
            undist = cv2.undistortPoints(pts_arr, K, dist, P=K).reshape(-1, 2)
            for idx, ki in enumerate(valid_ki):
                pts2d_all[vi, ki] = undist[idx]
                valid_mask[vi, ki] = True
        else:
            for idx, ki in enumerate(valid_ki):
                pts2d_all[vi, ki] = raw_pts[idx]
                valid_mask[vi, ki] = True

    if logger:
        valid_views = [v for v in view_names_sorted if v in poses_dict]
        logger.info(f"      RANSAC triangulation from {len(valid_views)} views "
                    f"(thresh={reproj_threshold}px): {', '.join(valid_views)}")

    points3d = np.zeros((n_keypoints, 3), dtype=np.float64)
    for ki in range(n_keypoints):
        pt = _triangulate_point_ransac(pts2d_all[:, ki, :], P_all, valid_mask[:, ki],
                                       reproj_threshold=reproj_threshold,
                                       weights_all=weights_all)
        if pt is not None:
            points3d[ki] = pt

    return points3d


# ============================================================================
# xrmocap triangulation
# ============================================================================

def triangulate_cluster_xrmocap(matched_cluster: Dict, triangulator,
                                 view_names_sorted: List[str],
                                 n_keypoints: int = 13,
                                 logger=None) -> Optional[np.ndarray]:
    """Triangulate 3D pose using xrmocap's AniposelibTriangulator.

    Returns:
        (n_keypoints, 3) array or None
    """
    poses_dict = matched_cluster['poses']
    all_views_points, all_views_mask, valid_view_names = [], [], []

    for view_name in view_names_sorted:
        if view_name not in poses_dict:
            all_views_points.append(np.zeros((n_keypoints, 2)))
            all_views_mask.append(np.zeros((n_keypoints, 1)))
            continue

        kpts2d = poses_dict[view_name].get('pred_keypoints_2d')
        if kpts2d is None or kpts2d.shape[1] < 2:
            all_views_points.append(np.zeros((n_keypoints, 2)))
            all_views_mask.append(np.zeros((n_keypoints, 1)))
            continue

        points_2d = kpts2d[:, :2]
        mask = np.array([[0 if (np.all(points_2d[k] == 0) or np.any(~np.isfinite(points_2d[k])))
                          else 1] for k in range(n_keypoints)], dtype=np.float64)
        all_views_points.append(points_2d)
        all_views_mask.append(mask)
        valid_view_names.append(view_name)

    try:
        points2d = np.array(all_views_points, dtype=np.float64)
        mask = np.array(all_views_mask, dtype=np.int32)
        if logger:
            logger.info(f"        xrmocap input: points2d={points2d.shape}, "
                        f"valid_views={', '.join(valid_view_names)}")
        points3d = triangulator.triangulate(points2d, points_mask=mask)
        return points3d
    except Exception as e:
        if logger:
            logger.error(f"        Triangulation failed: {e}")
        return None


def fuse_poses(matched_clusters: List[Dict], triangulator,
               view_names_sorted: List[str],
               proj_matrices: Optional[Dict] = None,
               K_dict: Optional[Dict] = None,
               dist_dict: Optional[Dict] = None,
               reproj_threshold: float = 20.0,
               n_keypoints: int = 13,
               logger=None) -> List[np.ndarray]:
    """Triangulate 3D poses for all matched clusters.

    Uses RANSAC when proj_matrices is provided; falls back to AniposelibTriangulator.

    Returns:
        List of (n_keypoints, 3) arrays
    """
    fused_poses = []
    for person_idx, cluster in enumerate(matched_clusters):
        if logger:
            logger.info(f"    Processing person {person_idx}")

        if proj_matrices is not None:
            pose_3d = triangulate_cluster_ransac(
                cluster, view_names_sorted, proj_matrices,
                K_dict=K_dict, dist_dict=dist_dict,
                reproj_threshold=reproj_threshold,
                n_keypoints=n_keypoints, logger=logger
            )
        else:
            pose_3d = triangulate_cluster_xrmocap(
                cluster, triangulator, view_names_sorted,
                n_keypoints=n_keypoints, logger=logger
            )

        if pose_3d is not None:
            fused_poses.append(pose_3d)
        elif logger:
            logger.warning(f"      Failed to triangulate person {person_idx}, skipping")

    return fused_poses
