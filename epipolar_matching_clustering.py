#!/usr/bin/env python3
"""
Stage 2 — Cross-View Pose Matching and Clustering

Loads per-view NPZ predictions from Stage 1 (SAM-3D-Body), matches persons
across views using epipolar geometry, and saves matched clusters for Stage 3.

Output:
    matched_clusters.pkl   — per-frame person clusters with 2D keypoints
    matching_results.json  — human-readable summary
    part1_matching_<ts>.log
"""

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
    # I/O
    parser.add_argument('--output_dir', type=str,
                    #    default='/home/y_li/workspace7/MAEM/sam-3d-body/output/sportscenter_21380/',
                    #    default='/work7/y_li/sam-3d-body/output/humanm3/basketball1/split1',
                    #    default='/work7/y_li/sam-3d-body/output/humanm3/basketball2/',
                    #    default='/home/y_li/workspace7/sam-3d-body/output/sportscenter_30000/',
                        help='Directory containing Stage 1 predictions (per-view NPZ)')
    parser.add_argument('--camera_param_dir', type=str,
                    #    default='/home/y_li/workspace7/xrmocap/xrmocap_data/xrmocap_sportscenter/xrmocap_meta_sportscenter/scene_0/camera_parameters',
                    #    default='/home/y_li/workspace7/humanm3/test/basketball1/split1/camera_calibration',
                    #    default='/home/y_li/workspace7/humanm3/test/basketball2/camera_calibration',
                        help='Directory containing camera parameter JSON files')
    parser.add_argument('--scene_dir', type=str,
                    #    default='/home/y_li/workspace7/xrmocap/xrmocap_data/xrmocap_sportscenter/scene_0',
                    #    default='/home/y_li/workspace7/humanm3/test/basketball1/split1/images',
                    #    default='/home/y_li/workspace7/humanm3/test/basketball2/images',
                        help='Root folder with per-view image subfolders')
    parser.add_argument('--intermediate_output', type=str,
                    #    default='/home/y_li/workspace7/MAEM/sam-3d-body/output/sportscenter_21380/intermediate_matched_clusters.pkl',
                    #    default='/work7/y_li/sam-3d-body/output/humanm3/basketball1/split1/intermediate_matched_clusters.pkl',
                    #    default='/work7/y_li/sam-3d-body/output/humanm3/basketball2/intermediate_matched_clusters.pkl',
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
                        choices=['pa_mpjpe', 'epipolar_only', 'hybrid',
                                 'epi_gate', 'repr_gate', 'epi_gate_only'],
                        help='Cross-view matching mode')
    parser.add_argument('--mpjpe_weight', type=float, default=0.3,
                        help='PA-MPJPE weight in hybrid mode')
    parser.add_argument('--epi_threshold', type=float, default=8.0,
                        help='Epipolar distance threshold (pixels)')
    parser.add_argument('--repr_threshold', type=float, default=10.0,
                        help='Reprojection error threshold (pixels)')
    # Visualization
    parser.add_argument('--visualize_matches', action='store_true', default=False,
                        help='Save per-frame match visualization images')

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

    # Discover frames from the view with most NPZ files
    ref_npz_dir = None
    for vn in view_names:
        candidate = os.path.join(args.output_dir, vn, 'npz')
        if os.path.exists(candidate):
            if ref_npz_dir is None or len(os.listdir(candidate)) > len(os.listdir(ref_npz_dir)):
                ref_npz_dir = candidate
    if not ref_npz_dir:
        raise FileNotFoundError(f"No NPZ directory found under {args.output_dir}")
    logger.info(f"Scanning frames from: {ref_npz_dir}")

    frame_numbers = sorted(
        int(f.split('.')[0])
        for f in os.listdir(ref_npz_dir)
        if f.endswith('.npz')
    )
    logger.info(f"Processing {len(frame_numbers)} frames...\n")

    all_frame_data = []
    total_gate_time = 0.0
    frames_with_matching = 0

    for frame_idx, frame_num in enumerate(frame_numbers):
        logger.info(f"--- Frame {frame_num} ({frame_idx + 1}/{len(frame_numbers)}) ---")

        # Load all views in parallel
        poses_world_dict = {}
        with ThreadPoolExecutor(max_workers=len(view_names)) as executor:
            futures = {
                executor.submit(load_view_data, vn, frame_num, args.output_dir,
                                args.scene_dir, camera_params_dict): vn
                for vn in view_names
            }
            for future in as_completed(futures):
                vn_result, view_poses = future.result()
                if view_poses is not None:
                    poses_world_dict[vn_result] = view_poses
                    logger.info(f"  {vn_result}: {len(view_poses)} person(s)")

        if not poses_world_dict:
            logger.info("  No valid predictions, skipping")
            continue

        # Match poses
        logger.info("  Performing cross-view matching...")
        matched_clusters, gate_time = match_poses_across_views(
            poses_world_dict,
            min_views=args.min_views_cluster,
            max_error_threshold=args.max_pa_mpjpe_threshold,
            bbox_score_threshold=args.bbox_score_threshold,
            matching_mode=args.matching_mode,
            camera_params_dict=camera_params_dict,
            mpjpe_weight=args.mpjpe_weight,
            epi_threshold=args.epi_threshold,
            repr_threshold=args.repr_threshold,
            logger=logger,
        )
        total_gate_time += gate_time
        frames_with_matching += 1
        logger.info(f"  Gate time: {gate_time * 1000:.2f}ms")

        if not matched_clusters:
            logger.info("  No matched clusters found")
            continue

        logger.info(f"  Found {len(matched_clusters)} matched person(s)")
        all_frame_data.append({
            'frame_idx': frame_idx,
            'frame_num': frame_num,
            'matched_clusters': matched_clusters,
            'num_persons': len(matched_clusters),
        })

        if args.visualize_matches:
            visualize_matched_clusters(
                matched_clusters, frame_num, args.scene_dir,
                os.path.join(args.output_dir, 'matched_visualizations'),
                view_names=view_names, logger=logger,
            )

    # Timing summary
    logger.info(f"\n{'=' * 80}")
    if frames_with_matching > 0:
        avg_ms = total_gate_time * 1000.0 / frames_with_matching
        logger.info(f"Gate timing ({args.matching_mode}): "
                    f"{frames_with_matching} frames, "
                    f"total={total_gate_time * 1000:.1f}ms, "
                    f"avg={avg_ms:.2f}ms/frame")

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
