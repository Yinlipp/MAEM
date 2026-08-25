#!/bin/bash
# Sweep --vertex_sample_rate around a baseline.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$SCRIPT_DIR/output/vertex_downsample_experiment"

OUTPUT_DIR="/path/to/stage1_output/basketball1/split1"
CAMERA_PARAM_DIR="/path/to/humanm3/test/basketball1/split1/camera_calibration"
SCENE_DIR="/path/to/humanm3/test/basketball1/split1/images"
GT_FILE="/path/to/humanm3/test/basketball1/split1/keypoints3d_GT.npz"
VIEW_NAME_PATTERN="camera_{i}"
NUM_VIEWS=4
CAMERA_PARAM_PATTERN="fisheye_param_{i:02d}.json"
START_FRAME=1800
NUM_FRAMES=200

SAM3D_CONDA_ENV="sam_3d_body"
XRMOCAP_CONDA_ENV="xrmocap"

BBOX=0.7
REPR=30.0
EPI=8.0
KMIN=2
RANSAC=20.0

RATES="1 2 4 8 16 32 64"

mkdir -p "$EXP_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "================ Stage 2 (parallel over rates) ================"
for RATE in $RATES; do
    DIR="$EXP_DIR/rate_${RATE}"
    mkdir -p "$DIR"
    if [ -s "$DIR/intermediate_matched_clusters.pkl" ]; then
        echo "  [skip] Stage 2 already done: rate=$RATE"
        continue
    fi
    (
        conda activate "$SAM3D_CONDA_ENV"
        { time python3 "$SCRIPT_DIR/epipolar_matching_clustering.py" \
            --output_dir "$OUTPUT_DIR" \
            --camera_param_dir "$CAMERA_PARAM_DIR" \
            --scene_dir "$SCENE_DIR" \
            --intermediate_output "$DIR/intermediate_matched_clusters.pkl" \
            --view_name_pattern "$VIEW_NAME_PATTERN" \
            --num_views "$NUM_VIEWS" \
            --camera_param_pattern "$CAMERA_PARAM_PATTERN" \
            --start_frame "$START_FRAME" \
            --num_frames "$NUM_FRAMES" \
            --bbox_score_threshold "$BBOX" \
            --repr_threshold "$REPR" \
            --epi_threshold "$EPI" \
            --min_views_cluster "$KMIN" \
            --matching_mode epi_gate \
            --vertex_sample_rate "$RATE" \
            --keypoint_convention humanm3_15 \
            ; } > "$DIR/driver_stage2.log" 2>&1
        if [ $? -ne 0 ]; then
            echo "  [FAIL] Stage 2 failed: rate=$RATE (see $DIR/driver_stage2.log)"
        else
            echo "  [ok] Stage 2: rate=$RATE"
        fi
    ) &
done
wait

echo "================ Stage 3 (parallel over rates) ================"
for RATE in $RATES; do
    DIR="$EXP_DIR/rate_${RATE}"
    if [ -s "$DIR/run/evaluation_results_xrmocap.json" ]; then
        echo "  [skip] Stage 3 already done: rate=$RATE"
        continue
    fi
    if [ ! -s "$DIR/intermediate_matched_clusters.pkl" ]; then
        echo "  [skip] Stage 3: no Stage 2 output for rate=$RATE"
        continue
    fi
    (
        conda activate "$XRMOCAP_CONDA_ENV"
        { time python3 "$SCRIPT_DIR/triangulation_evaluation.py" \
            --intermediate_input "$DIR/intermediate_matched_clusters.pkl" \
            --output_dir "$DIR" \
            --gt_file "$GT_FILE" \
            --dataset_name run \
            --reproj_threshold "$RANSAC" \
            ; } > "$DIR/driver_stage3.log" 2>&1
        if [ $? -ne 0 ]; then
            echo "  [FAIL] Stage 3 failed: rate=$RATE (see $DIR/driver_stage3.log)"
        else
            echo "  [ok] Stage 3: rate=$RATE"
        fi
    ) &
done
wait

echo "================ ALL VERTEX DOWNSAMPLE RUNS COMPLETE ================"
