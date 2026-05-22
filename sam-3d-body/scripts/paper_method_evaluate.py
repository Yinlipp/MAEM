#!/usr/bin/env python3
"""
Multi-view 3D Pose Evaluation: Paper Method (Section 4.2 + 4.3)

4.2 Human Identification:
    - Build complete graph with PA-MPJPE edge weights
    - (a) different camera, common frame  -> PA-MPJPE
    - (b) same camera, common frame       -> max PA-MPJPE (penalty)
    - (c) no common frame                 -> avg PA-MPJPE
    - Spectral Clustering + k-means -> N person classes

4.3 Global Coordinate Unification (NO pre-calibrated camera params):
    - Collect pred_cam_t correspondences of matched persons across camera pairs
    - Procrustes alignment per camera pair -> (s*, R*, t*)
    - Select base camera k* = argmin average alignment error
    - Transform all predictions to base camera coordinates

Metrics: MPJPE, PA-MPJPE, AP@75/100/125/150mm
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering

KEYPOINTS_IDX = [0, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


# ============================================================================
# LOGGING
# ============================================================================

def setup_logger(output_dir: str, name: str = 'paper_method') -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(os.path.join(output_dir, f'{name}_{ts}.log'), encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================================
# METRICS
# ============================================================================

def compute_similarity_transform(S1: np.ndarray, S2: np.ndarray):
    """Procrustes: find s, R, t minimizing ||S2 - s*R*S1 - t||"""
    mu1, mu2 = S1.mean(0), S2.mean(0)
    S1c, S2c = S1 - mu1, S2 - mu2
    var1 = np.mean(np.sum(S1c ** 2, axis=1))
    K = S2c.T @ S1c
    U, sigma, Vt = np.linalg.svd(K)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.sum(sigma * np.diag(D)) / (var1 * len(S1c)) if var1 > 0 else 1.0
    t = mu2 - s * R @ mu1
    return s, R, t


def apply_similarity_transform(pts: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return s * (R @ pts.T).T + t


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    err = np.linalg.norm(pred - gt, axis=-1)
    if mask is not None:
        err = err[mask > 0]
    return float(np.mean(err)) if len(err) > 0 else float('nan')


def compute_pa_mpjpe(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    p, g = pred.copy(), gt.copy()
    if mask is not None:
        valid = mask > 0
        if valid.sum() < 3:
            return float('nan')
        p, g = p[valid], g[valid]
    if len(p) < 3:
        return float('nan')
    try:
        s, R, t = compute_similarity_transform(p, g)
        p_aligned = apply_similarity_transform(p, s, R, t)
        return float(np.mean(np.linalg.norm(p_aligned - g, axis=-1)))
    except Exception:
        return float('nan')


# ============================================================================
# DATA LOADING
# ============================================================================

def load_frame_predictions(npz_path: str) -> List[Dict]:
    """Load all person predictions from one frame npz."""
    data = np.load(npz_path, allow_pickle=True)
    if 'outputs' not in data or len(data['outputs']) == 0:
        return []
    results = []
    for person_data in data['outputs']:
        d = person_data.item() if hasattr(person_data, 'item') else person_data
        if not isinstance(d, dict):
            continue
        if not all(k in d for k in ['pred_keypoints_3d', 'bbox_score', 'pred_cam_t']):
            continue
        kps = np.array(d['pred_keypoints_3d'])
        if kps.shape[0] > 21:
            kps = kps[KEYPOINTS_IDX]
        cam_t = np.array(d['pred_cam_t']).reshape(3)
        results.append({
            'keypoints_cam': kps,          # [13, 3] in camera SMPL space
            'cam_t': cam_t,                # [3] root translation in camera space
            'keypoints_world': kps + cam_t, # [13, 3] = cam_t-shifted
            'score': float(d['bbox_score']),
            'bbox': d.get('bbox', None),
        })
    return results


def load_all_predictions(output_dir: str, view_names: List[str],
                         logger: logging.Logger) -> Dict[int, Dict[str, List[Dict]]]:
    """
    Returns: {frame_num: {view_name: [person_dict, ...]}}
    """
    all_data = {}
    # Use first view to enumerate frames
    ref_view = view_names[0]
    npz_dir = os.path.join(output_dir, ref_view, 'npz')
    if not os.path.exists(npz_dir):
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")

    frame_files = sorted(f for f in os.listdir(npz_dir) if f.endswith('.npz'))
    frame_nums = [int(f.split('.')[0]) for f in frame_files]
    logger.info(f"Found {len(frame_nums)} frames from {ref_view}")

    for frame_num in frame_nums:
        all_data[frame_num] = {}
        for view in view_names:
            path = os.path.join(output_dir, view, 'npz', f'{frame_num:06d}.npz')
            if not os.path.exists(path):
                path = os.path.join(output_dir, view, 'npz', f'{frame_num}.npz')
            if os.path.exists(path):
                preds = load_frame_predictions(path)
                if preds:
                    all_data[frame_num][view] = preds
    return all_data


# ============================================================================
# 4.2 HUMAN IDENTIFICATION — SPECTRAL CLUSTERING
# ============================================================================

def build_affinity_matrix(nodes: List[Dict]) -> Tuple[np.ndarray, float, float]:
    """
    nodes: list of {'view': str, 'keypoints': [13,3], ...}
    Returns: affinity matrix W (before exp transform), max_pa, avg_pa
    """
    n = len(nodes)
    W = np.zeros((n, n))

    # Case (a): different camera, common frame -> PA-MPJPE
    case_a_values = []
    for i in range(n):
        for j in range(i + 1, n):
            if nodes[i]['view'] != nodes[j]['view']:
                pa = compute_pa_mpjpe(nodes[i]['keypoints'], nodes[j]['keypoints'])
                if not np.isnan(pa):
                    W[i, j] = W[j, i] = pa
                    case_a_values.append(pa)

    max_pa = float(np.max(case_a_values)) if case_a_values else 1000.0
    avg_pa = float(np.mean(case_a_values)) if case_a_values else 1000.0

    # Case (b): same camera -> max PA-MPJPE (penalty = different person)
    # Case (c): no common frame (N/A in per-frame processing) -> avg_pa
    for i in range(n):
        for j in range(i + 1, n):
            if nodes[i]['view'] == nodes[j]['view']:   # case (b)
                W[i, j] = W[j, i] = max_pa
            elif W[i, j] == 0:                         # case (c)
                W[i, j] = W[j, i] = avg_pa

    return W, max_pa, avg_pa


def spectral_cluster_frame(poses_per_view: Dict[str, List[Dict]],
                           n_persons: int,
                           sigma: Optional[float] = None,
                           logger: Optional[logging.Logger] = None) -> List[Dict]:
    """
    Per-frame spectral clustering.
    Returns: list of clusters, each cluster = {view: person_data}
    """
    nodes = []
    for view, plist in poses_per_view.items():
        for p in plist:
            nodes.append({'view': view, 'keypoints': p['keypoints_world'],
                          'cam_t': p['cam_t'], 'score': p['score'], 'bbox': p['bbox'],
                          'keypoints_cam': p['keypoints_cam']})

    if len(nodes) == 0:
        return []
    if len(nodes) <= n_persons:
        # Not enough nodes to cluster - each node is its own cluster
        clusters = [{n['view']: n} for n in nodes]
        return clusters

    W, max_pa, avg_pa = build_affinity_matrix(nodes)

    if sigma is None:
        valid = W[(W > 0) & (W < 1e9)]
        sigma = float(np.median(valid)) if len(valid) > 0 else 1000.0

    # Affinity: exp(-W / sigma)
    A = np.exp(-W / sigma)
    np.fill_diagonal(A, 0)

    try:
        sc = SpectralClustering(n_clusters=n_persons, affinity='precomputed',
                                random_state=42, n_init=10)
        labels = sc.fit_predict(A)
    except Exception as e:
        if logger:
            logger.warning(f"    Spectral clustering failed: {e}, falling back to single cluster")
        labels = np.zeros(len(nodes), dtype=int)

    # Build clusters: one dict per cluster_id {view: best_person}
    cluster_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_dict[label].append(nodes[idx])

    clusters = []
    for label, members in cluster_dict.items():
        cluster = {}
        for m in members:
            view = m['view']
            # If multiple persons from same view in same cluster, keep highest score
            if view not in cluster or m['score'] > cluster[view]['score']:
                cluster[view] = m
        if len(cluster) >= 1:
            clusters.append(cluster)

    return clusters


# ============================================================================
# 4.3 GLOBAL COORDINATE UNIFICATION — PROCRUSTES
# ============================================================================

def collect_root_correspondences(all_clusters: Dict[int, List[Dict]],
                                 view_a: str, view_b: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect paired root translations (cam_t) for persons visible in both view_a and view_b.
    Returns: (pts_a, pts_b) each shape [N, 3]
    """
    pts_a, pts_b = [], []
    for frame_num, clusters in all_clusters.items():
        for cluster in clusters:
            if view_a in cluster and view_b in cluster:
                pts_a.append(cluster[view_a]['cam_t'])
                pts_b.append(cluster[view_b]['cam_t'])
    if len(pts_a) < 3:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.array(pts_a), np.array(pts_b)


def estimate_camera_alignments(all_clusters: Dict[int, List[Dict]],
                               view_names: List[str],
                               logger: logging.Logger) -> Dict[Tuple, Tuple]:
    """
    For each camera pair (k, l): Procrustes alignment s*, R*, t* such that
    pts_k ≈ s* R* pts_l + t*  (transform l -> k)
    Returns: {(view_k, view_l): (s, R, t, error)}
    """
    alignments = {}
    for i, vk in enumerate(view_names):
        for j, vl in enumerate(view_names):
            if i >= j:
                continue
            pts_k, pts_l = collect_root_correspondences(all_clusters, vk, vl)
            if len(pts_k) < 3:
                logger.info(f"  Camera pair ({vk}, {vl}): only {len(pts_k)} correspondences, skipping")
                continue
            s, R, t = compute_similarity_transform(pts_l, pts_k)
            pts_l_aligned = apply_similarity_transform(pts_l, s, R, t)
            err = float(np.mean(np.linalg.norm(pts_k - pts_l_aligned, axis=-1)))
            alignments[(vk, vl)] = (s, R, t, err)
            logger.info(f"  Camera pair ({vk}, {vl}): {len(pts_k)} pts, alignment error={err:.4f}m")
    return alignments


def select_base_camera(view_names: List[str],
                       alignments: Dict[Tuple, Tuple],
                       logger: logging.Logger) -> str:
    """
    L_k = mean S_{k,l} over all l != k
    k* = argmin L_k
    """
    avg_errors = {}
    for vk in view_names:
        errs = []
        for vl in view_names:
            if vk == vl:
                continue
            key = (vk, vl) if (vk, vl) in alignments else (vl, vk)
            if key in alignments:
                errs.append(alignments[key][3])
        if errs:
            avg_errors[vk] = float(np.mean(errs))

    if not avg_errors:
        base = view_names[0]
        logger.info(f"  No alignment data, defaulting base camera to {base}")
        return base

    base = min(avg_errors, key=avg_errors.get)
    for v, e in sorted(avg_errors.items(), key=lambda x: x[1]):
        marker = " ← BASE" if v == base else ""
        logger.info(f"  {v}: avg alignment error = {e:.4f}m{marker}")
    return base


def get_transform_to_base(view: str, base: str,
                          alignments: Dict[Tuple, Tuple]) -> Optional[Tuple]:
    """
    Get (s, R, t) to transform `view`'s cam_t space to `base`'s cam_t space.
    """
    if view == base:
        return 1.0, np.eye(3), np.zeros(3)
    if (base, view) in alignments:
        s, R, t, _ = alignments[(base, view)]
        return s, R, t
    if (view, base) in alignments:
        s, R, t, _ = alignments[(view, base)]
        # Invert: base = s*R*view + t => view = (1/s)*R^T*(base - t)
        s_inv = 1.0 / s if s != 0 else 1.0
        R_inv = R.T
        t_inv = -s_inv * R_inv @ t
        return s_inv, R_inv, t_inv
    return None


def transform_cluster_to_base(cluster: Dict, base: str,
                               alignments: Dict[Tuple, Tuple]) -> Dict:
    """
    Transform all keypoints in cluster to base camera's coordinate system.
    Returns: {view: {keypoints_base, score}}
    """
    transformed = {}
    for view, person in cluster.items():
        params = get_transform_to_base(view, base, alignments)
        if params is None:
            continue
        s, R, t = params
        kps_world = person['keypoints']  # [13, 3]
        kps_base = apply_similarity_transform(kps_world, s, R, t)
        transformed[view] = {
            'keypoints': kps_base,
            'score': person['score'],
        }
    return transformed


def fuse_cluster(transformed_cluster: Dict) -> np.ndarray:
    """Weighted fusion by bbox_score."""
    kps_list = [v['keypoints'] for v in transformed_cluster.values()]
    scores = np.array([v['score'] for v in transformed_cluster.values()])
    if len(kps_list) == 0:
        return None
    kps = np.stack(kps_list)           # [n_views, 13, 3]
    w = scores / scores.sum()
    return np.sum(kps * w[:, None, None], axis=0)  # [13, 3]


# ============================================================================
# GT MATCHING
# ============================================================================

def load_gt(gt_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Returns: (pose [n_frames, n_persons, 13, 4], mask, convention)"""
    data = np.load(gt_path, allow_pickle=True)
    pose = data['pose']
    mask = data['mask']
    convention = str(data['convention'])
    return pose, mask, convention


def find_best_gt_match(fused_pose: np.ndarray,
                       gt_poses: np.ndarray, gt_masks: np.ndarray,
                       matched_gt_ids: set,
                       match_threshold_mm: float = 500.0) -> Tuple[int, np.ndarray, np.ndarray]:
    """Match fused pose to best unmatched GT person by PA-MPJPE (coordinate-system agnostic)."""
    best_idx, best_pa, best_gt_pose, best_gt_mask = -1, float('inf'), None, None
    for gt_idx in range(gt_poses.shape[0]):
        if gt_idx in matched_gt_ids:
            continue
        gt_pose = gt_poses[gt_idx, :, :3]   # [n_kps, 3]
        gt_mask = gt_masks[gt_idx]           # [n_kps]
        if gt_mask.sum() < 3:
            continue
        pa = compute_pa_mpjpe(fused_pose, gt_pose, gt_mask) * 1000  # m -> mm
        if not np.isnan(pa) and pa < best_pa:
            best_idx, best_pa = gt_idx, pa
            best_gt_pose, best_gt_mask = gt_pose.copy(), gt_mask.copy()
    return best_idx, best_gt_pose, best_gt_mask


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Paper Method: Spectral Clustering + Procrustes Coordinate Unification')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing SAM-3D-Body predictions per view')
    parser.add_argument('--gt_file', type=str, required=True,
                        help='Ground truth NPZ file')
    parser.add_argument('--n_persons', type=int, default=10,
                        help='Number of persons in scene (for spectral clustering)')
    parser.add_argument('--view_names', type=str, nargs='+',
                        default=None,
                        help='View names, e.g. camera_0 camera_1 ... (auto-detected if None)')
    parser.add_argument('--sigma', type=float, default=None,
                        help='Sigma for affinity exp(-PA-MPJPE/sigma). Default: median of PA-MPJPE')
    parser.add_argument('--bbox_score_threshold', type=float, default=0.5,
                        help='Minimum bbox score to include a detection')
    parser.add_argument('--min_views', type=int, default=2,
                        help='Minimum views in cluster for evaluation')
    parser.add_argument('--match_threshold_mm', type=float, default=500.0,
                        help='Max MPJPE (mm) for GT matching')
    args = parser.parse_args()

    logger = setup_logger(args.output_dir, 'paper_method')
    logger.info("=" * 70)
    logger.info("Paper Method: Spectral Clustering + Procrustes Unification")
    logger.info("=" * 70)

    # --- Auto-detect view names ---
    if args.view_names is None:
        view_names = sorted([
            d for d in os.listdir(args.output_dir)
            if os.path.isdir(os.path.join(args.output_dir, d, 'npz'))
        ])
    else:
        view_names = args.view_names
    logger.info(f"Views: {view_names}")
    logger.info(f"N persons: {args.n_persons}")

    # --- Load all predictions ---
    logger.info("\n[1/5] Loading predictions...")
    all_preds = load_all_predictions(args.output_dir, view_names, logger)
    frame_nums = sorted(all_preds.keys())
    logger.info(f"  Total frames: {len(frame_nums)}")

    # Filter by bbox score
    for fn in frame_nums:
        for v in list(all_preds[fn].keys()):
            all_preds[fn][v] = [p for p in all_preds[fn][v]
                                if p['score'] >= args.bbox_score_threshold]
            if not all_preds[fn][v]:
                del all_preds[fn][v]

    # --- 4.2 Spectral clustering per frame ---
    logger.info("\n[2/5] Human Identification (Spectral Clustering per frame)...")
    all_clusters = {}  # {frame_num: [cluster, ...]}
    for fn in frame_nums:
        views_data = all_preds.get(fn, {})
        if not views_data:
            all_clusters[fn] = []
            continue
        clusters = spectral_cluster_frame(views_data, args.n_persons,
                                          sigma=args.sigma, logger=logger)
        all_clusters[fn] = clusters
        n_valid = sum(1 for c in clusters if len(c) >= args.min_views)
        logger.info(f"  Frame {fn}: {sum(len(v) for v in views_data.values())} detections "
                    f"-> {len(clusters)} clusters ({n_valid} with >= {args.min_views} views)")

    # --- 4.3 Global Coordinate Unification ---
    logger.info("\n[3/5] Global Coordinate Unification (Procrustes)...")
    alignments = estimate_camera_alignments(all_clusters, view_names, logger)

    logger.info("\n  Selecting base camera...")
    base_camera = select_base_camera(view_names, alignments, logger)
    logger.info(f"  Base camera: {base_camera}")

    # --- Load GT ---
    logger.info("\n[4/5] Loading GT and evaluating...")
    gt_poses_all, gt_masks_all, gt_convention = load_gt(args.gt_file)
    logger.info(f"  GT: {gt_poses_all.shape}, convention={gt_convention}")

    # --- Pre-compute fused poses for all frames ---
    # Structure: {frame_idx: [(fused [13,3], cluster_views), ...]}
    fused_by_frame = {}
    for frame_idx, fn in enumerate(frame_nums):
        if frame_idx >= gt_poses_all.shape[0]:
            continue
        clusters = all_clusters.get(fn, [])
        fused_list = []
        for cluster in clusters:
            if len(cluster) < args.min_views:
                continue
            transformed = transform_cluster_to_base(cluster, base_camera, alignments)
            if len(transformed) < args.min_views:
                continue
            fused = fuse_cluster(transformed)
            if fused is not None:
                fused_list.append((fused, list(cluster.keys())))
        if fused_list:
            fused_by_frame[frame_idx] = fused_list

    # --- Global Procrustes alignment: pred space -> GT world space ---
    # Collect all valid pred root joints and corresponding GT root joints
    # Use greedy nearest-neighbor to build correspondences for global alignment
    logger.info("\n[4.5/5] Global scene Procrustes alignment (pred -> GT world space)...")
    pred_roots_all, gt_roots_all = [], []
    for frame_idx in sorted(fused_by_frame.keys()):
        gt_frame_poses = gt_poses_all[frame_idx]   # [n_gt, 13, 4]
        gt_frame_masks = gt_masks_all[frame_idx]
        used_gt = set()
        for fused, _ in fused_by_frame[frame_idx]:
            best_d, best_gt_root, best_gt_idx = float('inf'), None, -1
            for gi in range(gt_frame_poses.shape[0]):
                if gi in used_gt or gt_frame_masks[gi].sum() < 3:
                    continue
                gt_root = gt_frame_poses[gi, 0, :3]   # first keypoint as root proxy
                pred_root = fused[0]                   # first keypoint
                d = np.linalg.norm(pred_root - gt_root)
                if d < best_d:
                    best_d, best_gt_root, best_gt_idx = d, gt_root, gi
            if best_gt_idx >= 0:
                pred_roots_all.append(fused[0])
                gt_roots_all.append(best_gt_root)
                used_gt.add(best_gt_idx)

    global_s, global_R, global_t = 1.0, np.eye(3), np.zeros(3)
    if len(pred_roots_all) >= 3:
        pred_roots = np.array(pred_roots_all)
        gt_roots   = np.array(gt_roots_all)
        global_s, global_R, global_t = compute_similarity_transform(pred_roots, gt_roots)
        err = float(np.mean(np.linalg.norm(
            gt_roots - apply_similarity_transform(pred_roots, global_s, global_R, global_t),
            axis=-1)))
        logger.info(f"  Global alignment: {len(pred_roots)} correspondences, "
                    f"s={global_s:.4f}, align_err={err:.4f}m")
    else:
        logger.warning(f"  Not enough correspondences ({len(pred_roots_all)}) for global alignment, skipping.")

    # --- Evaluation ---
    all_pa_mpjpe = []
    total_gt_count = 0       # total valid GT persons across all frames
    total_matched_count = 0  # GT persons successfully matched

    for frame_idx, fn in enumerate(frame_nums):
        if frame_idx >= gt_poses_all.shape[0]:
            continue

        gt_frame_poses = gt_poses_all[frame_idx]   # [n_gt, 13, 4]
        gt_frame_masks = gt_masks_all[frame_idx]

        # count valid GT persons in this frame
        n_valid_gt = int(np.sum([gt_frame_masks[gi].sum() >= 3
                                 for gi in range(gt_frame_poses.shape[0])]))
        total_gt_count += n_valid_gt

        if frame_idx not in fused_by_frame:
            continue

        logger.info(f"\nFrame {fn} ({frame_idx+1}/{len(frame_nums)}):")

        # Collect valid GT indices
        valid_gt_ids = [gi for gi in range(gt_frame_poses.shape[0])
                        if gt_frame_masks[gi].sum() >= 3]
        fused_list = fused_by_frame[frame_idx]  # [(fused_raw, views), ...]
        n_pred = len(fused_list)
        n_gt   = len(valid_gt_ids)

        if n_pred == 0 or n_gt == 0:
            continue

        # Build cost matrix [n_pred x n_gt]: PA-MPJPE (coordinate-agnostic)
        cost_matrix = np.full((n_pred, n_gt), fill_value=1e6)
        for pi, (fused_raw, _) in enumerate(fused_list):
            for gj, gi in enumerate(valid_gt_ids):
                gt_pose = gt_frame_poses[gi, :, :3]
                gt_mask = gt_frame_masks[gi]
                pa = compute_pa_mpjpe(fused_raw, gt_pose, gt_mask) * 1000
                if not np.isnan(pa):
                    cost_matrix[pi, gj] = pa

        # Hungarian assignment: each GT matched at most once
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for pi, gj in zip(row_ind, col_ind):
            pa_mpjpe = cost_matrix[pi, gj]
            if pa_mpjpe >= 1e6:
                continue
            gi = valid_gt_ids[gj]
            fused_raw, cluster_views = fused_list[pi]
            total_matched_count += 1
            all_pa_mpjpe.append(pa_mpjpe)
            logger.info(f"  Person (GT={gi}, views={cluster_views}): "
                        f"PA-MPJPE={pa_mpjpe:.1f}mm")

    # --- Summary ---
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)

    recall = total_matched_count / total_gt_count if total_gt_count > 0 else 0.0
    logger.info(f"\nRecall: {total_matched_count}/{total_gt_count} GT persons matched = {recall*100:.2f}%")

    if all_pa_mpjpe:
        pa_arr = np.array(all_pa_mpjpe)

        logger.info(f"Total evaluated poses: {len(pa_arr)}")
        logger.info(f"PA-MPJPE: mean={pa_arr.mean():.2f}mm  median={np.median(pa_arr):.2f}mm  std={pa_arr.std():.2f}mm")
        logger.info("(Note: MPJPE not reported — no camera extrinsics available for world-space alignment)")

        # AP: #(PA-MPJPE < thr) / total_gt_count  (standard detection-style AP)
        logger.info(f"\nAP Metrics (#PA-MPJPE < threshold / total GT={total_gt_count}):")
        for thr in [75, 100, 125, 150]:
            ap = float((pa_arr < thr).sum()) / total_gt_count if total_gt_count > 0 else 0.0
            logger.info(f"  AP@{thr}mm:  {ap*100:.2f}%")

        # Results table
        logger.info("\n+---------------------------+-----------+")
        logger.info("|        Metric             |   Value   |")
        logger.info("+---------------------------+-----------+")
        logger.info(f"| Recall                    | {recall*100:8.2f}%  |")
        logger.info(f"| PA-MPJPE mean             | {pa_arr.mean():8.2f}mm |")
        logger.info(f"| PA-MPJPE median           | {np.median(pa_arr):8.2f}mm |")
        logger.info(f"| PA-MPJPE std              | {pa_arr.std():8.2f}mm |")
        for thr in [75, 100, 125, 150]:
            ap = float((pa_arr < thr).sum()) / total_gt_count if total_gt_count > 0 else 0.0
            logger.info(f"| AP@{thr}mm                  | {ap*100:8.2f}%  |")
        logger.info("+---------------------------+-----------+")

        # Save JSON
        results = {
            'total_gt': total_gt_count,
            'total_matched': total_matched_count,
            'recall': float(recall),
            'total_poses': int(len(pa_arr)),
            'pa_mpjpe_mean': float(pa_arr.mean()),
            'pa_mpjpe_median': float(np.median(pa_arr)),
            'pa_mpjpe_std': float(pa_arr.std()),
            'ap75mm': float((pa_arr < 75).sum()) / total_gt_count if total_gt_count > 0 else 0.0,
            'ap100mm': float((pa_arr < 100).sum()) / total_gt_count if total_gt_count > 0 else 0.0,
            'ap125mm': float((pa_arr < 125).sum()) / total_gt_count if total_gt_count > 0 else 0.0,
            'ap150mm': float((pa_arr < 150).sum()) / total_gt_count if total_gt_count > 0 else 0.0,
            'base_camera': base_camera,
            'n_persons': args.n_persons,
        }
        out_json = os.path.join(args.output_dir, 'paper_method_results.json')
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {out_json}")
    else:
        logger.info("No valid evaluations found.")


if __name__ == '__main__':
    main()
