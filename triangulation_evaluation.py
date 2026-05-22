#!/usr/bin/env python3
"""
Stage 3 — RANSAC Triangulation and Evaluation

Loads matched clusters from Stage 2, triangulates 3D poses via per-keypoint
RANSAC (with xrmocap AniposelibTriangulator as fallback), and evaluates
against ground-truth keypoints3d_GT.npz.

Output:
    predicted_3d_poses_xrmocap.npz   — padded (n_frames, n_persons, n_kpts, 3)
    predicted_3d_poses_gt_format.npz — same shape as GT for direct comparison
    evaluation_results_xrmocap.json  — MPJPE / PA-MPJPE / AP metrics
    part2_triangulation_<ts>.log
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

# Ensure xrmocap source is on the path
_XRMOCAP_PATH = os.path.join(os.path.dirname(__file__), 'xrmocap')
if _XRMOCAP_PATH not in sys.path:
    sys.path.insert(0, _XRMOCAP_PATH)

from xrmocap.ops.triangulation.aniposelib_triangulator import AniposelibTriangulator

try:
    from scipy.optimize import linear_sum_assignment as _lsa
    def _hungarian(cost_matrix):
        row_ind, col_ind = _lsa(cost_matrix)
        return list(zip(row_ind, col_ind))
except ImportError:
    from munkres import Munkres
    _munkres = Munkres()
    def _hungarian(cost_matrix):
        return _munkres.compute(cost_matrix.tolist())

from maem.logging_utils import setup_logger
from maem.metrics import compute_mpjpe, compute_pa_mpjpe
from maem.triangulation import (load_projection_matrices,
                                load_camera_parameters_xrmocap,
                                fuse_poses)


def main():
    parser = argparse.ArgumentParser(
        description='Stage 3: RANSAC triangulation and evaluation'
    )
    parser.add_argument('--intermediate_input', type=str,
                    #    default='/home/y_li/workspace7/MAEM/sam-3d-body/output/sportscenter_30000/intermediate_matched_clusters.pkl',
                    #    default='/home/y_li/workspace7/sam-3d-body/output/sportscenter_21380/intermediate_matched_clusters.pkl',
                    #    default='/home/y_li/workspace7/sam-3d-body/output/humanm3/basketball2/intermediate_matched_clusters.pkl',
                        help='Path to matched_clusters.pkl from Stage 2')
    parser.add_argument('--output_dir', type=str,
                    #    default='/home/y_li/workspace7/MAEM/sam-3d-body/output/sportscenter_21380',
                    #    default='/home/y_li/workspace7/sam-3d-body/output/sportscenter_30000',
                    #    default='/home/y_li/workspace7/sam-3d-body/output/humanm3/basketball2/',
                        help='Output directory for results')
    parser.add_argument('--gt_file', type=str,
                    #    default='/home/y_li/workspace7/xrmocap/xrmocap_data/xrmocap_sportscenter/xrmocap_meta_sportscenter/scene_0/keypoints3d_GT.npz',
                    #    default='/home/y_li/workspace7/xrmocap/xrmocap_data/xrmocap_sportscenter_30000/xrmocap_meta_sportscenter_30000/scene_0/keypoints3d_GT.npz',
                    #    default='/home/y_li/workspace7/humanm3/test/basketball2/keypoints3d_GT.npz',
                        help='Ground truth NPZ file (keypoints3d_GT.npz)')
    parser.add_argument('--save_3d_poses', action='store_true', default=True,
                        help='Save triangulated 3D poses to NPZ')
    parser.add_argument('--reproj_threshold', type=float, default=20.0,
                        help='RANSAC reprojection inlier threshold (pixels)')
    args = parser.parse_args()

    logger, log_file = setup_logger(args.output_dir, 'part2_triangulation')
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)
    logger.info("Stage 3: Triangulation and Evaluation")
    logger.info("=" * 80)
    logger.info(f"Intermediate input: {args.intermediate_input}")
    logger.info(f"Output dir:         {args.output_dir}")
    logger.info(f"GT file:            {args.gt_file}")
    logger.info("")

    # Load Stage 2 output
    logger.info("Loading Stage 2 data...")
    import numpy.core._multiarray_umath as _mu
    sys.modules['numpy._core._multiarray_umath'] = _mu
    sys.modules['numpy._core'] = np.core

    with open(args.intermediate_input, 'rb') as f:
        intermediate_data = pickle.load(f)

    all_frame_data = intermediate_data['all_frame_data']
    camera_params_dict = intermediate_data['camera_params_dict']
    camera_param_dir = intermediate_data.get('args', {}).get('camera_param_dir')

    logger.info(f"  {len(all_frame_data)} frames, "
                f"{sum(fd['num_persons'] for fd in all_frame_data)} clusters total")
    logger.info("")

    # Load camera parameters
    logger.info("Loading camera parameters...")
    camera_params_xrmocap = load_camera_parameters_xrmocap(camera_params_dict, camera_param_dir)
    proj_matrices, K_dict, dist_dict = load_projection_matrices(camera_params_dict, camera_param_dir)
    view_names_sorted = sorted(camera_params_dict.keys())
    logger.info(f"  {len(camera_params_xrmocap)} cameras: {', '.join(view_names_sorted)}")
    logger.info("")

    # Create fallback triangulator
    triangulator = AniposelibTriangulator(camera_parameters=camera_params_xrmocap)

    # Load GT
    logger.info(f"Loading GT from {args.gt_file}...")
    gt_data = np.load(args.gt_file)
    gt_poses = gt_data['pose']
    gt_mask = gt_data.get('mask') if 'mask' in gt_data else None
    logger.info(f"  GT shape: {gt_poses.shape}")
    logger.info("")

    # Per-frame processing
    all_mpjpe, all_pa_mpjpe, all_frame_summaries = [], [], []
    total_gt_count = total_matched_count = 0
    all_frames_poses = []
    frame_to_idx_mapping = {}

    MATCH_THRESHOLD_MM = 500.0
    INF_COST = 1e9
    N_KEYPOINTS = gt_poses.shape[2]

    logger.info("Processing frames...")
    for frame_data in all_frame_data:
        frame_idx = frame_data['frame_idx']
        frame_num = frame_data['frame_num']
        matched_clusters = frame_data['matched_clusters']

        logger.info(f"\n--- Frame {frame_num} ({frame_idx}/{len(all_frame_data)}) ---")

        # Triangulate
        frame_avg_poses = fuse_poses(
            matched_clusters, triangulator, view_names_sorted,
            proj_matrices=proj_matrices, K_dict=K_dict, dist_dict=dist_dict,
            reproj_threshold=args.reproj_threshold,
            n_keypoints=N_KEYPOINTS, logger=logger,
        )

        if not frame_avg_poses:
            logger.info(f"  No valid poses for frame {frame_num}")
            continue

        logger.info(f"  Triangulated {len(frame_avg_poses)} pose(s)")

        if args.save_3d_poses:
            all_frames_poses.append(np.array(frame_avg_poses))
            frame_to_idx_mapping[frame_num] = len(all_frames_poses) - 1

        # GT matching
        if frame_idx >= len(gt_poses):
            logger.info(f"  Warning: frame_idx {frame_idx} out of GT range")
            continue

        gt_pose_frame = gt_poses[frame_idx]
        gt_mask_frame = gt_mask[frame_idx] if gt_mask is not None else None

        gt_poses_clean, gt_masks_clean = {}, {}
        for gt_idx in range(gt_pose_frame.shape[0]):
            mask = gt_mask_frame[gt_idx] if gt_mask_frame is not None else None
            if mask is not None and mask.sum() == 0:
                continue
            gt_pose = gt_pose_frame[gt_idx, :, :3].copy()
            if mask is not None:
                gt_pose[mask == 0] = 0
            if not np.all(np.isfinite(gt_pose)):
                continue
            gt_poses_clean[gt_idx] = gt_pose
            gt_masks_clean[gt_idx] = mask

        gt_valid_indices = sorted(gt_poses_clean.keys())
        total_gt_count += len(gt_valid_indices)

        n_pred = len(frame_avg_poses)
        cost_matrix = np.full((n_pred, len(gt_valid_indices)), INF_COST)
        for pi, fused_pose in enumerate(frame_avg_poses):
            for col, gt_idx in enumerate(gt_valid_indices):
                cost_matrix[pi, col] = compute_mpjpe(
                    fused_pose, gt_poses_clean[gt_idx], gt_masks_clean[gt_idx]
                )

        frame_person_results = []
        for pred_idx, col in _hungarian(cost_matrix):
            cost = cost_matrix[pred_idx, col]
            gt_idx = gt_valid_indices[col]
            if cost >= MATCH_THRESHOLD_MM:
                logger.info(f"    Person {pred_idx}: no GT match (MPJPE={cost:.1f}mm)")
                continue

            fused_pose = frame_avg_poses[pred_idx]
            mpjpe = compute_mpjpe(fused_pose, gt_poses_clean[gt_idx], gt_masks_clean[gt_idx])
            pa_mpjpe = compute_pa_mpjpe(fused_pose, gt_poses_clean[gt_idx], gt_masks_clean[gt_idx])

            logger.info(f"    Person {pred_idx} → GT {gt_idx}: "
                        f"MPJPE={mpjpe:.2f}mm, PA-MPJPE={pa_mpjpe:.2f}mm")
            total_matched_count += 1
            all_mpjpe.append(mpjpe)
            all_pa_mpjpe.append(pa_mpjpe)
            frame_person_results.append({
                'person_idx': int(pred_idx), 'gt_person_idx': int(gt_idx),
                'mpjpe': float(mpjpe), 'pa_mpjpe': float(pa_mpjpe),
            })

        if frame_person_results:
            all_frame_summaries.append({
                'frame_idx': frame_idx, 'frame_num': frame_num,
                'num_matched': len(frame_person_results),
                'persons': frame_person_results,
            })

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("Evaluation Results")
    logger.info("=" * 80)

    if all_mpjpe:
        mean_mpjpe     = float(np.mean(all_mpjpe))
        median_mpjpe   = float(np.median(all_mpjpe))
        mean_pa_mpjpe  = float(np.mean(all_pa_mpjpe))
        median_pa_mpjpe= float(np.median(all_pa_mpjpe))
        recall = total_matched_count / total_gt_count if total_gt_count > 0 else 0.0
        pa_arr = np.array(all_pa_mpjpe)
        ap_thresholds = [75, 100, 125, 150]
        ap_values = {t: float((pa_arr < t).sum()) / total_gt_count
                     if total_gt_count > 0 else 0.0 for t in ap_thresholds}

        logger.info(f"GT persons:   {total_gt_count}")
        logger.info(f"Matched:      {total_matched_count}  (Recall: {recall*100:.2f}%)")
        logger.info(f"MPJPE:        {mean_mpjpe:.2f}mm mean, {median_mpjpe:.2f}mm median")
        logger.info(f"PA-MPJPE:     {mean_pa_mpjpe:.2f}mm mean, {median_pa_mpjpe:.2f}mm median")
        for t in ap_thresholds:
            logger.info(f"AP@{t}mm:      {ap_values[t]*100:.2f}%")

        results_json = {
            'summary': {
                'total_gt': total_gt_count,
                'total_matched': total_matched_count,
                'recall': recall,
                'mean_mpjpe': mean_mpjpe,
                'median_mpjpe': median_mpjpe,
                'mean_pa_mpjpe': mean_pa_mpjpe,
                'median_pa_mpjpe': median_pa_mpjpe,
                **{f'ap{t}mm': ap_values[t] for t in ap_thresholds},
            },
            'frame_summaries': all_frame_summaries,
        }
        out_json = os.path.join(args.output_dir, 'evaluation_results_xrmocap.json')
        with open(out_json, 'w') as f:
            json.dump(results_json, f, indent=2)
        logger.info(f"\nResults saved to: {out_json}")
    else:
        logger.info("No valid evaluations performed.")

    # Save 3D poses
    if args.save_3d_poses and all_frames_poses:
        max_persons = max(p.shape[0] for p in all_frames_poses)
        n_frames = len(all_frames_poses)
        poses_padded = np.zeros((n_frames, max_persons, N_KEYPOINTS, 3), dtype=np.float32)
        poses_mask = np.zeros((n_frames, max_persons), dtype=np.int32)
        for fi, fp in enumerate(all_frames_poses):
            n = fp.shape[0]
            poses_padded[fi, :n] = fp
            poses_mask[fi, :n] = 1

        out_npz = os.path.join(args.output_dir, 'predicted_3d_poses_xrmocap.npz')
        np.savez(out_npz, pose=poses_padded, mask=poses_mask,
                 frame_mapping=frame_to_idx_mapping, convention='13points')
        logger.info(f"\nSaved 3D poses: {out_npz}  shape={poses_padded.shape}")

        # GT-format output
        if gt_poses is not None:
            gt_nf, gt_np, gt_nk, _ = gt_poses.shape
            poses_gt_fmt = np.zeros((gt_nf, gt_np, gt_nk, 3), dtype=np.float32)
            for frame_num, pred_idx in frame_to_idx_mapping.items():
                if pred_idx < gt_nf:
                    fp = all_frames_poses[pred_idx]
                    n = min(fp.shape[0], gt_np)
                    poses_gt_fmt[pred_idx, :n] = fp[:n]
            out_gt = os.path.join(args.output_dir, 'predicted_3d_poses_gt_format.npz')
            np.savez(out_gt, pose=poses_gt_fmt, convention='13points')
            logger.info(f"Saved GT-format poses: {out_gt}  shape={poses_gt_fmt.shape}")

    logger.info(f"\n{'=' * 80}")
    logger.info("Stage 3 Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
