#!/usr/bin/env python3
"""Stage 3 — RANSAC triangulation of Stage 2 clusters, evaluated against keypoints3d_GT.npz."""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

# Ensure xrmocap source is on the path
_XRMOCAP_PATH = os.path.join(os.path.dirname(__file__), 'xrmocap')
if _XRMOCAP_PATH not in sys.path:
    sys.path.insert(0, _XRMOCAP_PATH)

from xrmocap.ops.triangulation.aniposelib_triangulator import AniposelibTriangulator

from maem.logging_utils import setup_logger
from maem.metrics import (compute_mpjpe, compute_pa_mpjpe, eval_list_to_ap_official,
                          eval_list_to_mpjpe_official, eval_list_to_recall_official)
from maem.triangulation import (load_projection_matrices,
                                load_camera_parameters_xrmocap,
                                fuse_poses)

MATCH_THRESHOLD_MM = 500.0
AP_THRESHOLDS = [25, 50, 75, 100, 125, 150]  # Human-M3: np.arange(0.025, 0.155, 0.025) in mm


def compute_stats(eval_list, total_gt):
    """Recall/MPJPE/PA-MPJPE/AP off one eval_list, Human-M3-protocol (see maem.metrics)."""
    recall = eval_list_to_recall_official(eval_list, total_gt, MATCH_THRESHOLD_MM)
    mean_mpjpe = eval_list_to_mpjpe_official(eval_list, MATCH_THRESHOLD_MM)
    ap_values = {t: eval_list_to_ap_official(eval_list, total_gt, t) for t in AP_THRESHOLDS}

    ranked = sorted(eval_list, key=lambda k: k['score'], reverse=True)
    gt_det = []
    tp_mpjpes, tp_pa_mpjpes = [], []
    for item in ranked:
        if item['mpjpe'] < MATCH_THRESHOLD_MM and item['gt_id'] is not None and item['gt_id'] not in gt_det:
            gt_det.append(item['gt_id'])
            tp_mpjpes.append(item['mpjpe'])
            tp_pa_mpjpes.append(item['pa_mpjpe'])

    return {
        'total_gt': total_gt,
        'total_matched': len(tp_mpjpes),
        'recall': recall,
        'mean_mpjpe': mean_mpjpe,
        'median_mpjpe': float(np.median(tp_mpjpes)) if tp_mpjpes else float('inf'),
        'mean_pa_mpjpe': float(np.mean(tp_pa_mpjpes)) if tp_pa_mpjpes else float('inf'),
        'median_pa_mpjpe': float(np.median(tp_pa_mpjpes)) if tp_pa_mpjpes else float('inf'),
        **{f'ap{t}mm': ap_values[t] for t in AP_THRESHOLDS},
    }


def log_stats(logger, label, stats):
    logger.info(f"[{label}] GT persons: {stats['total_gt']}")
    logger.info(f"[{label}] Matched@{MATCH_THRESHOLD_MM:.0f}mm: {stats['total_matched']}  "
                f"(Recall: {stats['recall']*100:.2f}%)")
    logger.info(f"[{label}] MPJPE@{MATCH_THRESHOLD_MM:.0f}mm: {stats['mean_mpjpe']:.2f}mm mean, "
                f"{stats['median_mpjpe']:.2f}mm median")
    logger.info(f"[{label}] PA-MPJPE (bonus): {stats['mean_pa_mpjpe']:.2f}mm mean, "
                f"{stats['median_pa_mpjpe']:.2f}mm median")
    for t in AP_THRESHOLDS:
        logger.info(f"[{label}] AP@{t}mm: {stats[f'ap{t}mm']*100:.2f}%")


def process_dataset(name, intermediate_input, gt_file, reproj_threshold,
                    output_dir, save_3d_poses, logger):
    """Triangulate + match one dataset; gt_id is locally 0-based, main() offsets it when merging."""
    logger.info(f"\n{'=' * 80}\nDataset: {name}\n{'=' * 80}")
    logger.info(f"Intermediate input: {intermediate_input}")
    logger.info(f"GT file:            {gt_file}")

    with open(intermediate_input, 'rb') as f:
        intermediate_data = pickle.load(f)

    all_frame_data = intermediate_data['all_frame_data']
    camera_params_dict = intermediate_data['camera_params_dict']
    camera_param_dir = intermediate_data.get('args', {}).get('camera_param_dir')

    logger.info(f"  {len(all_frame_data)} frames, "
                f"{sum(fd['num_persons'] for fd in all_frame_data)} clusters total")

    camera_params_xrmocap = load_camera_parameters_xrmocap(camera_params_dict, camera_param_dir)
    proj_matrices, K_dict, dist_dict = load_projection_matrices(camera_params_dict, camera_param_dir)
    view_names_sorted = sorted(camera_params_dict.keys())
    logger.info(f"  {len(camera_params_xrmocap)} cameras: {', '.join(view_names_sorted)}")

    triangulator = AniposelibTriangulator(camera_parameters=camera_params_xrmocap)

    gt_data = np.load(gt_file)
    gt_poses = gt_data['pose']
    gt_mask = gt_data.get('mask') if 'mask' in gt_data else None
    logger.info(f"  GT shape: {gt_poses.shape}")
    N_KEYPOINTS = gt_poses.shape[2]

    assert len(all_frame_data) == len(gt_poses), (
        f"[{name}] Frame-completeness check FAILED: Stage 2 output has {len(all_frame_data)} "
        f"frames, GT has {len(gt_poses)} frames")
    logger.info(f"  Frame-completeness check PASSED: {len(all_frame_data)} frames match GT")

    eval_list = []
    total_gt_count = 0
    all_frame_summaries = []
    all_frames_poses = []
    frame_to_idx_mapping = {}
    frame_idx_by_pred_idx = []
    triangulation_time_sec = 0.0

    for frame_data in all_frame_data:
        frame_idx = frame_data['frame_idx']
        frame_num = frame_data['frame_num']
        matched_clusters = frame_data['matched_clusters']

        if frame_idx >= len(gt_poses):
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
        gt_id_offset = total_gt_count
        total_gt_count += len(gt_valid_indices)

        _t0 = time.perf_counter()
        frame_avg_poses, frame_confidences = fuse_poses(
            matched_clusters, triangulator, view_names_sorted,
            proj_matrices=proj_matrices, K_dict=K_dict, dist_dict=dist_dict,
            reproj_threshold=reproj_threshold,
            n_keypoints=N_KEYPOINTS, logger=logger,
        )
        triangulation_time_sec += time.perf_counter() - _t0

        if not frame_avg_poses:
            continue

        if save_3d_poses:
            all_frames_poses.append(np.array(frame_avg_poses))
            frame_to_idx_mapping[frame_num] = len(all_frames_poses) - 1
            frame_idx_by_pred_idx.append(frame_idx)

        # Human-M3 protocol: each prediction finds its own nearest GT, no Hungarian
        frame_person_results = []
        for pred_idx, fused_pose in enumerate(frame_avg_poses):
            conf = frame_confidences[pred_idx] if pred_idx < len(frame_confidences) else 0.0

            if not gt_valid_indices:
                eval_list.append({
                    'mpjpe': float('inf'), 'pa_mpjpe': float('inf'),
                    'score': conf, 'gt_id': None, 'frame_idx': frame_idx,
                })
                continue

            cand_mpjpes = [
                compute_mpjpe(fused_pose, gt_poses_clean[gt_idx], gt_masks_clean[gt_idx])
                for gt_idx in gt_valid_indices
            ]
            best_local = int(np.argmin(cand_mpjpes))
            best_gt_idx = gt_valid_indices[best_local]
            best_mpjpe = float(cand_mpjpes[best_local])
            best_pa_mpjpe = float(compute_pa_mpjpe(
                fused_pose, gt_poses_clean[best_gt_idx], gt_masks_clean[best_gt_idx]
            ))
            global_gt_id = gt_id_offset + best_local

            eval_list.append({
                'mpjpe': best_mpjpe, 'pa_mpjpe': best_pa_mpjpe,
                'score': conf, 'gt_id': global_gt_id, 'frame_idx': frame_idx,
            })
            frame_person_results.append({
                'person_idx': int(pred_idx), 'gt_person_idx': int(best_gt_idx),
                'mpjpe': best_mpjpe, 'pa_mpjpe': best_pa_mpjpe, 'score': conf,
            })

        if frame_person_results:
            all_frame_summaries.append({
                'frame_idx': frame_idx, 'frame_num': frame_num,
                'num_matched': len(frame_person_results),
                'persons': frame_person_results,
            })

    logger.info(f"  Triangulation time: {triangulation_time_sec:.1f}s total, "
                f"{triangulation_time_sec / max(len(all_frame_data), 1) * 1000:.1f}ms/frame")

    stats = compute_stats(eval_list, total_gt_count)
    log_stats(logger, name, stats)

    dataset_dir = os.path.join(output_dir, name)
    os.makedirs(dataset_dir, exist_ok=True)
    with open(os.path.join(dataset_dir, 'evaluation_results_xrmocap.json'), 'w') as f:
        json.dump({
            'summary': {
                'total_frames': len(all_frame_data),
                'evaluated_frames': len(all_frame_summaries),
                'reproj_threshold': reproj_threshold,
                'triangulation_time_sec': triangulation_time_sec,
                **stats,
            },
            'frame_summaries': all_frame_summaries,
        }, f, indent=2)

    if save_3d_poses and all_frames_poses:
        max_persons = max(p.shape[0] for p in all_frames_poses)
        n_frames = len(all_frames_poses)
        poses_padded = np.zeros((n_frames, max_persons, N_KEYPOINTS, 3), dtype=np.float32)
        poses_mask = np.zeros((n_frames, max_persons), dtype=np.int32)
        for fi, fp in enumerate(all_frames_poses):
            n = fp.shape[0]
            poses_padded[fi, :n] = fp
            poses_mask[fi, :n] = 1
        np.savez(os.path.join(dataset_dir, 'predicted_3d_poses_xrmocap.npz'),
                 pose=poses_padded, mask=poses_mask,
                 frame_mapping=frame_to_idx_mapping, convention=f'{N_KEYPOINTS}points')

        # Indexed by frame_idx (same convention as gt_poses), not pred_idx — pred_idx is a
        # position in the compacted all_frames_poses list (skips frames with zero triangulated
        # poses), so using it here would misalign every frame after the first skip.
        gt_nf, gt_np_, gt_nk, _ = gt_poses.shape
        poses_gt_fmt = np.zeros((gt_nf, gt_np_, gt_nk, 3), dtype=np.float32)
        for pred_idx, fp in enumerate(all_frames_poses):
            fidx = frame_idx_by_pred_idx[pred_idx]
            if fidx < gt_nf:
                n = min(fp.shape[0], gt_np_)
                poses_gt_fmt[fidx, :n] = fp[:n]
        np.savez(os.path.join(dataset_dir, 'predicted_3d_poses_gt_format.npz'),
                 pose=poses_gt_fmt, convention=f'{N_KEYPOINTS}points')

    logger.info(f"  Saved per-dataset outputs to: {dataset_dir}")
    return eval_list, total_gt_count, triangulation_time_sec


def main():
    parser = argparse.ArgumentParser(
        description='Stage 3: RANSAC triangulation and evaluation'
    )
    # required, no hard-coded defaults; pass multiple to combine datasets
    parser.add_argument('--intermediate_input', type=str, required=True, nargs='+',
                        help='One or more matched_clusters.pkl paths from Stage 2')
    parser.add_argument('--gt_file', type=str, required=True, nargs='+',
                        help='One keypoints3d_GT.npz per --intermediate_input, same order')
    parser.add_argument('--dataset_name', type=str, nargs='+', default=None,
                        help='One label per dataset (default: dataset0, dataset1, ...)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--save_3d_poses', action='store_true', default=True,
                        help='Save triangulated 3D poses to NPZ')
    parser.add_argument('--reproj_threshold', type=float, default=20.0,
                        help='RANSAC reprojection inlier threshold (pixels)')
    args = parser.parse_args()

    if len(args.intermediate_input) != len(args.gt_file):
        parser.error('--intermediate_input and --gt_file must have the same count')
    names = args.dataset_name or [f'dataset{i}' for i in range(len(args.intermediate_input))]
    if len(names) != len(args.intermediate_input):
        parser.error('--dataset_name must have the same count as --intermediate_input')

    logger, log_file = setup_logger(args.output_dir, 'part2_triangulation')
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)
    logger.info("Stage 3: Triangulation and Evaluation")
    logger.info("=" * 80)
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Datasets:   {', '.join(names)}")

    import numpy.core._multiarray_umath as _mu
    sys.modules['numpy._core._multiarray_umath'] = _mu
    sys.modules['numpy._core'] = np.core

    combined_eval_list = []
    combined_total_gt = 0
    per_dataset_stats = {}
    per_dataset_timing = {}

    for name, intermediate_input, gt_file in zip(names, args.intermediate_input, args.gt_file):
        eval_list, total_gt, timing_sec = process_dataset(
            name, intermediate_input, gt_file, args.reproj_threshold,
            args.output_dir, args.save_3d_poses, logger,
        )
        per_dataset_stats[name] = compute_stats(eval_list, total_gt)
        per_dataset_timing[name] = timing_sec

        for item in eval_list:
            if item['gt_id'] is not None:
                item['gt_id'] += combined_total_gt
        combined_eval_list.extend(eval_list)
        combined_total_gt += total_gt

    logger.info(f"\n{'=' * 80}\nCombined Results ({', '.join(names)})\n{'=' * 80}")
    combined_stats = compute_stats(combined_eval_list, combined_total_gt)
    log_stats(logger, 'combined', combined_stats)

    combined_json = {
        'datasets': names,
        'per_dataset': per_dataset_stats,
        'per_dataset_triangulation_time_sec': per_dataset_timing,
        'combined': combined_stats,
    }
    out_json = os.path.join(args.output_dir, 'evaluation_results_combined.json')
    with open(out_json, 'w') as f:
        json.dump(combined_json, f, indent=2)
    logger.info(f"\nCombined results saved to: {out_json}")

    logger.info(f"\n{'=' * 80}")
    logger.info("Stage 3 Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
