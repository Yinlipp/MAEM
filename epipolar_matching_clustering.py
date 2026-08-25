#!/usr/bin/env python3
"""Stage 2 — cross-view pose matching and clustering; outputs matched_clusters.pkl for Stage 3."""

import argparse
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from maem.camera_utils import load_camera_params
from maem.data_io import load_view_data
from maem.logging_utils import setup_logger
from maem.pose_clustering import match_poses_across_views
from maem.visualization import visualize_matched_clusters


def main():
    parser = argparse.ArgumentParser(
        description='Stage 2: Multi-view pose matching and clustering'
    )
    # I/O — required, no hard-coded defaults
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing Stage 1 predictions (per-view NPZ)')
    parser.add_argument('--camera_param_dir', type=str, required=True,
                        help='Directory containing camera parameter JSON files')
    parser.add_argument('--scene_dir', type=str, required=True,
                        help='Root folder with per-view image subfolders')
    parser.add_argument('--intermediate_output', type=str, required=True,
                        help='Output path for matched_clusters.pkl')
    # View naming
    parser.add_argument('--view_name_pattern', type=str, default='ace_{i}',
                        help='View folder name template, e.g. "camera_{i}" or "ace_{i}"')
    parser.add_argument('--num_views', type=int, default=6,
                        help='Number of camera views')
    parser.add_argument('--camera_param_pattern', type=str,
                        default='fisheye_param_{i:02d}.json',
                        help='Camera JSON filename template')
    # Matching
    parser.add_argument('--min_views_cluster', type=int, default=4,
                        help='Minimum views required to form a valid cluster')
    parser.add_argument('--max_pa_mpjpe_threshold', type=float, default=300,
                        help='Maximum PA-MPJPE for a valid match (mm)')
    parser.add_argument('--bbox_score_threshold', type=float, default=0.9,
                        help='Minimum detection confidence score')
    parser.add_argument('--matching_mode', type=str, default='epi_gate',
                        choices=['pa_mpjpe', 'sparse', 'epi_gate',
                                 'repr_gate', 'epi_gate_only'],
                        help='Cross-view matching mode')
    parser.add_argument('--epi_threshold', type=float, default=8.0,
                        help='Epipolar distance threshold (pixels)')
    parser.add_argument('--repr_threshold', type=float, default=10.0,
                        help='Reprojection error threshold (pixels)')
    # Explicit frame range, not scanned from NPZ files (Stage 1 skips zero-detection frames)
    parser.add_argument('--num_frames', type=int, required=True,
                        help='Total number of frames to process')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='First frame number')
    parser.add_argument('--frame_step', type=int, default=1,
                        help='Stride between frame numbers (SportCenter is sampled every 3rd frame)')
    parser.add_argument('--visualize_matches', action='store_true', default=False,
                        help='Save per-frame match visualization images')
    parser.add_argument('--keypoint_convention', type=str, default='sportcenter_13',
                        choices=['sportcenter_13', 'humanm3_15'],
                        help='Keypoint set extracted from SAM-3D-Body predictions')
    parser.add_argument('--vertex_sample_rate', type=int, default=1,
                        help='Use every Nth mesh vertex for the epipolar cost (1=all)')

    args = parser.parse_args()

    logger, log_file = setup_logger(args.output_dir, 'part1_matching')
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)
    logger.info("Stage 2: Multi-view Pose Matching and Clustering")
    logger.info("=" * 80)
    logger.info(f"Output dir:         {args.output_dir}")
    logger.info(f"Intermediate output:{args.intermediate_output}")
    logger.info(f"Matching mode:      {args.matching_mode}")
    logger.info(f"Min views/cluster:  {args.min_views_cluster}")
    logger.info("")

    # Build view names and load camera parameters
    view_names = [args.view_name_pattern.format(i=i) for i in range(args.num_views)]
    logger.info("Loading camera parameters...")
    camera_params_dict = {}
    for i, view_name in enumerate(view_names):
        param_file = os.path.join(args.camera_param_dir,
                                  args.camera_param_pattern.format(i=i))
        camera_params_dict[view_name] = load_camera_params(param_file)
        logger.info(f"  {view_name}: {param_file}")

    frame_numbers = list(range(args.start_frame,
                               args.start_frame + args.num_frames * args.frame_step,
                               args.frame_step))
    logger.info(f"Processing {len(frame_numbers)} frames...\n")

    all_frame_data = []
    total_repr_gate_time = 0.0
    total_epi_gate_time = 0.0
    total_clustering_time = 0.0

    for frame_idx, frame_num in enumerate(frame_numbers):
        logger.info(f"--- Frame {frame_num} ({frame_idx + 1}/{len(frame_numbers)}) ---")

        # Load all views in parallel
        poses_world_dict = {}
        with ThreadPoolExecutor(max_workers=len(view_names)) as executor:
            futures = {
                executor.submit(load_view_data, vn, frame_num, args.output_dir,
                                args.scene_dir, camera_params_dict,
                                args.keypoint_convention, args.vertex_sample_rate): vn
                for vn in view_names
            }
            for future in as_completed(futures):
                vn_result, view_poses = future.result()
                if view_poses is not None:
                    poses_world_dict[vn_result] = view_poses
                    logger.info(f"  {vn_result}: {len(view_poses)} person(s)")

        if not poses_world_dict:
            logger.info("  No valid predictions")
            matched_clusters = []
        else:
            logger.info("  Performing cross-view matching...")
            matched_clusters, repr_gate_time, epi_gate_time, clustering_time = match_poses_across_views(
                poses_world_dict,
                min_views=args.min_views_cluster,
                max_error_threshold=args.max_pa_mpjpe_threshold,
                bbox_score_threshold=args.bbox_score_threshold,
                matching_mode=args.matching_mode,
                camera_params_dict=camera_params_dict,
                epi_threshold=args.epi_threshold,
                repr_threshold=args.repr_threshold,
                logger=logger,
            )
            total_repr_gate_time += repr_gate_time
            total_epi_gate_time += epi_gate_time
            total_clustering_time += clustering_time
            logger.info(f"  Gate time: repr={repr_gate_time * 1000:.2f}ms, "
                        f"epi={epi_gate_time * 1000:.2f}ms, "
                        f"clustering={clustering_time * 1000:.2f}ms")
            if not matched_clusters:
                logger.info("  No matched clusters found")

        logger.info(f"  Found {len(matched_clusters)} matched person(s)")
        all_frame_data.append({
            'frame_idx': frame_idx,
            'frame_num': frame_num,
            'matched_clusters': matched_clusters,
            'num_persons': len(matched_clusters),
        })

        if args.visualize_matches and matched_clusters:
            visualize_matched_clusters(
                matched_clusters, frame_num, args.scene_dir,
                os.path.join(args.output_dir, 'matched_visualizations'),
                view_names=view_names, logger=logger,
            )

    # Timing summary — averaged over the full input sequence (len(frame_numbers)), not just
    # frames that had a valid prediction, so this matches a full-sequence-average Table 8 number.
    logger.info(f"\n{'=' * 80}")
    n_total_frames = len(frame_numbers)
    total_gate_time = total_repr_gate_time + total_epi_gate_time
    logger.info(f"Bbox reprojection filtering timing: {n_total_frames} frames, "
                f"total={total_repr_gate_time * 1000:.1f}ms, "
                f"avg={total_repr_gate_time * 1000.0 / n_total_frames:.2f}ms/frame")
    logger.info(f"Dense epipolar matching timing: {n_total_frames} frames, "
                f"total={total_epi_gate_time * 1000:.1f}ms, "
                f"avg={total_epi_gate_time * 1000.0 / n_total_frames:.2f}ms/frame")
    logger.info(f"Clustering timing: {n_total_frames} frames, "
                f"total={total_clustering_time * 1000:.1f}ms, "
                f"avg={total_clustering_time * 1000.0 / n_total_frames:.2f}ms/frame")
    logger.info(f"Gate timing ({args.matching_mode}, repr+epi combined): "
                f"{n_total_frames} frames, "
                f"total={total_gate_time * 1000:.1f}ms, "
                f"avg={total_gate_time * 1000.0 / n_total_frames:.2f}ms/frame")

    # Save matched clusters
    os.makedirs(os.path.dirname(args.intermediate_output), exist_ok=True)
    with open(args.intermediate_output, 'wb') as f:
        pickle.dump({
            'all_frame_data': all_frame_data,
            'camera_params_dict': camera_params_dict,
            'args': vars(args),
        }, f)
    logger.info(f"\nSaved {len(all_frame_data)} frames → {args.intermediate_output}")
    logger.info(f"Total persons: {sum(fd['num_persons'] for fd in all_frame_data)}")
    assert len(all_frame_data) == args.num_frames, (
        f"Frame-completeness check FAILED: {len(all_frame_data)} of {args.num_frames} frames saved")
    logger.info(f"Frame-completeness check PASSED: all {args.num_frames} frames saved")

    # Save JSON summary
    matching_json = os.path.join(args.output_dir, 'matching_results.json')
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_frames': len(all_frame_data),
            'total_persons': sum(fd['num_persons'] for fd in all_frame_data),
            'min_views_cluster': args.min_views_cluster,
            'matching_mode': args.matching_mode,
            'bbox_score_threshold': args.bbox_score_threshold,
        },
        'frames': [
            {
                'frame_idx': fd['frame_idx'],
                'frame_num': fd['frame_num'],
                'num_persons': fd['num_persons'],
                'persons': [
                    {
                        'person_idx': pi,
                        'num_views': cl['num_views'],
                        'avg_error': float(cl.get('avg_error', 0.0)),
                        'views': [
                            {'view_name': vn,
                             'bbox_score': float(vd.get('scores', 0.0)),
                             'has_keypoints': 'keypoints' in vd}
                            for vn, vd in cl['poses'].items()
                        ],
                    }
                    for pi, cl in enumerate(fd['matched_clusters'])
                ],
            }
            for fd in all_frame_data
        ],
    }
    with open(matching_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON summary → {matching_json}")
    logger.info("=" * 80)
    logger.info("Stage 2 Complete!")


if __name__ == '__main__':
    main()
