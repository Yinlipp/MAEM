#!/bin/bash
# One-at-a-time sweep of 5 pipeline thresholds around a baseline, evaluated end to end.
# Each (parameter, value) run is independent, so Stage 2 runs (then Stage 3 runs) in parallel.
#
# Runs on the TRAIN split (basketball1/split1's train/ frames, never test/): thresholds are
# picked here, then fixed and applied unchanged to test/ for every number reported in the
# paper (see README's Train/Test Split section) — this script must never touch test/.
#
# Swept parameters:
#   1. detection confidence   -> --bbox_score_threshold   (Stage 2, quality filter)
#   2. bbox reprojection      -> --repr_threshold          (Stage 2, Stage-1 gate)
#   3. mesh epipolar distance -> --epi_threshold           (Stage 2, Stage-2 gate)
#   4. Kmin                   -> --min_views_cluster       (Stage 2, cluster validity)
#   5. RANSAC inlier thresh   -> --reproj_threshold        (Stage 3, triangulation only)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$SCRIPT_DIR/output/sensitivity_experiment"

# ==== Edit these for your layout ====
OUTPUT_DIR="/path/to/stage1_output/basketball1/split1_train"
CAMERA_PARAM_DIR="/path/to/humanm3/train/basketball1/split1/camera_calibration"
SCENE_DIR="/path/to/humanm3/train/basketball1/split1/images"
GT_FILE="/path/to/humanm3/train/basketball1/split1/keypoints3d_GT.npz"
VIEW_NAME_PATTERN="camera_{i}"
NUM_VIEWS=4
CAMERA_PARAM_PATTERN="fisheye_param_{i:02d}.json"
START_FRAME=0
NUM_FRAMES=1800

SAM3D_CONDA_ENV="sam_3d_body"   # conda env name from SAM 3D Body's INSTALL.md (README Stage 1)
XRMOCAP_CONDA_ENV="xrmocap"     # conda env name from xrMoCap's install docs (README Stage 3)
# =====================================

BASE_BBOX=0.7
BASE_REPR=30.0
BASE_EPI=8.0
BASE_KMIN=2
BASE_RANSAC=20.0

mkdir -p "$EXP_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"

run_stage2 () {
    local DIR="$1" BBOX="$2" REPR="$3" EPI="$4" KMIN="$5"
    mkdir -p "$DIR"
    if [ -s "$DIR/intermediate_matched_clusters.pkl" ]; then
        echo "  [skip] Stage 2 already done: $DIR"
        return 0
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
            --keypoint_convention humanm3_15 \
            ; } > "$DIR/driver_stage2.log" 2>&1
        if [ $? -ne 0 ]; then
            echo "  [FAIL] Stage 2 failed: $DIR (see $DIR/driver_stage2.log)"
        else
            echo "  [ok] Stage 2: $DIR"
        fi
    ) &
}

run_stage3 () {
    local DIR="$1" SRC_PKL="$2" RANSAC="$3"
    if [ -s "$DIR/run/evaluation_results_xrmocap.json" ]; then
        echo "  [skip] Stage 3 already done: $DIR"
        return 0
    fi
    mkdir -p "$DIR"
    (
        conda activate "$XRMOCAP_CONDA_ENV"
        { time python3 "$SCRIPT_DIR/triangulation_evaluation.py" \
            --intermediate_input "$SRC_PKL" \
            --output_dir "$DIR" \
            --gt_file "$GT_FILE" \
            --dataset_name run \
            --reproj_threshold "$RANSAC" \
            ; } > "$DIR/driver_stage3.log" 2>&1
        if [ $? -ne 0 ]; then
            echo "  [FAIL] Stage 3 failed: $DIR (see $DIR/driver_stage3.log)"
        else
            echo "  [ok] Stage 3: $DIR"
        fi
    ) &
}

echo "================ baseline (Stage 2) ================"
BASELINE_DIR="$EXP_DIR/baseline"
run_stage2 "$BASELINE_DIR" "$BASE_BBOX" "$BASE_REPR" "$BASE_EPI" "$BASE_KMIN"
wait
BASELINE_PKL="$BASELINE_DIR/intermediate_matched_clusters.pkl"

echo "================ Stage 2 sweeps (parallel) ================"
for V in 0.3 0.4 0.6 0.8 0.9; do
    run_stage2 "$EXP_DIR/detection_confidence/${V}" "$V" "$BASE_REPR" "$BASE_EPI" "$BASE_KMIN"
done
for V in 10 15 20 40 60 100; do
    run_stage2 "$EXP_DIR/bbox_reprojection/${V}" "$BASE_BBOX" "$V" "$BASE_EPI" "$BASE_KMIN"
done
for V in 3 8 10 15 20 30; do
    run_stage2 "$EXP_DIR/mesh_epipolar_distance/${V}" "$BASE_BBOX" "$BASE_REPR" "$V" "$BASE_KMIN"
done
for V in 3 4; do
    run_stage2 "$EXP_DIR/kmin/${V}" "$BASE_BBOX" "$BASE_REPR" "$BASE_EPI" "$V"
done
wait

echo "================ Stage 3 (parallel, baseline + all Stage-2 sweeps + ransac sweep) ================"
run_stage3 "$BASELINE_DIR" "$BASELINE_PKL" "$BASE_RANSAC"
for V in 0.3 0.4 0.6 0.8 0.9; do
    DIR="$EXP_DIR/detection_confidence/${V}"
    run_stage3 "$DIR" "$DIR/intermediate_matched_clusters.pkl" "$BASE_RANSAC"
done
for V in 10 15 20 40 60 100; do
    DIR="$EXP_DIR/bbox_reprojection/${V}"
    run_stage3 "$DIR" "$DIR/intermediate_matched_clusters.pkl" "$BASE_RANSAC"
done
for V in 3 8 10 15 20 30; do
    DIR="$EXP_DIR/mesh_epipolar_distance/${V}"
    run_stage3 "$DIR" "$DIR/intermediate_matched_clusters.pkl" "$BASE_RANSAC"
done
for V in 3 4; do
    DIR="$EXP_DIR/kmin/${V}"
    run_stage3 "$DIR" "$DIR/intermediate_matched_clusters.pkl" "$BASE_RANSAC"
done
for V in 5 10 15 30 50 80; do
    run_stage3 "$EXP_DIR/ransac_inlier_threshold/${V}" "$BASELINE_PKL" "$V"
done
wait

echo "================ ALL SENSITIVITY RUNS COMPLETE ================"
