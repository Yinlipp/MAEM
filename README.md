# MAEM: Mesh-Aware Epipolar Matching forMulti-View Multi-Person 3D Pose Estimation in Basketball

## Overview
MAEM (Mesh-Aware Epipolar Matching) is a retraining-free multi-view multi-person 3D pose estimation framework explicitly designed for team basketball scenarios. MAEM achieves cross-view association purely through geometric constraints, requiring zero target-domain network retraining. MAEM was evaluated on two public multi-view datasets: 
- [SportCenter Multi-View Human Pose Estimation Dataset](https://www.epfl.ch/labs/cvlab/data/sportcenter-dataset/)
- [Human-M3 Dataset](https://github.com/soullessrobot/Human-M3-Dataset)

![fig](figs/fig.png)


## Usage Pipeline
## Stage 1: Single &mdash; View 3D Mesh Recovery.

We employ the [SAM 3D Body model](https://github.com/facebookresearch/sam-3d-body) as the single-view perception frontend to extract a rich suite of 2D and 3D spatial attributes, including 3D human mesh vertex coordinates, their corresponding 2D projections, as well as 2D/3D joint keypoints and bounding boxes.

Please clone the official repository and set up environment configurations following the SAM 3D Body [official guidelines](https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md).

    conda activate sam_3d_body
    cd sam-3d-body

    python sam-3d-body/sam_prediction.py \
    --checkpoints_dir /path/to/checkpoints \
    --img_root_folder /path/to/images \
    --output_dir      /path/to/output
   

```text
Input:
├── images/
│   ├── camera_0/   # *.jpg / *.png
│   ├── camera_1/
│   └── ...

Output:
└── output/
    ├── camera_0/
    │   ├── <frame>.jpg        # rendered overlay image
    │   └── npz/
    │       └── <frame>.npz    # pred_keypoints_3d, pred_keypoints_2d,
    │                            pred_keypoints_2d_verts, bbox, bbox_score, pred_cam_t
    └── ...

```
**Note:** `pred_keypoints_2d_verts` (2D projections of all 18,439 mesh vertices) is added by modifying SAM-3D-Body source and is required by the epipolar matching stage.


## Stage 2 &mdash; Cross-view Matching and Clustering
cd ..

**SportCenter dataset** (6 fisheye cameras):
```
python epipolar_matching_clustering.py \
    --output_dir           /path/to/stage1_output \
    --camera_param_dir     /path/to/camera_params \
    --scene_dir            /path/to/images \
    --intermediate_output  /path/to output/intermediate_matched_clusters.pkl \
    --view_name_pattern    "ace_{i}" \
    --num_views            6 \
    --camera_param_pattern "fisheye_param_{i:02d}.json" \
    --matching_mode        epi_gate \
    --bbox_score_threshold 0.9 \
    --epi_threshold        8.0 \
    --repr_threshold       10.0 \
    --min_views_cluster    4
```

**Human-M3 dataset** (3 or 4 standard cameras):
```
python epipolar_matching_clustering.py \
    --output_dir           /path/to/stage1_output \
    --camera_param_dir     /path/to/camera_params \
    --scene_dir            /path/to/images \
    --intermediate_output  /path/to output/intermediate_matched_clusters.pkl \
    --view_name_pattern    "camera_{i}" \
    --num_views            4 \
    --camera_param_pattern "camera_{i}.json" \
    --matching_mode        epi_gate \
    --bbox_score_threshold 0.7 \
    --epi_threshold        8.0 \
    --repr_threshold       30.0 \
    --min_views_cluster    2
```
**Key parameters:**

| Parameter | Description | Default |
|---|---|---|
| `--view_name_pattern` | View folder name template(two datasets use distinct file names and folder names), e.g. `"ace_{i}"` or `"camera_{i}"` | `ace_{i}` |
| `--num_views` | Number of camera views | `6` |
| `--camera_param_pattern` | Camera JSON filename template, e.g. `"fisheye_param_{i:02d}.json"` | `fisheye_param_{i:02d}.json` |
| `--matching_mode` | `epi_gate` (reprojection + epipolar matching) | `epi_gate` |
| `--bbox_score_threshold` | Detection confidence threshold | `0.9` |
| `--epi_threshold` | Epipolar distance threshold (pixels) | `8.0` |
| `--repr_threshold` | BBox center reprojection error threshold | `10.0` |
| `--min_views_cluster` | Minimum views required to form a valid person cluster | `4` |
| `--visualize_matches` | Save per-frame match visualization images | `False` |

**Output:**

| File | Description |
|---|---|
| `intermediate_matched_clusters.pkl` | Per-frame person clusters, each with 2D keypoints per view, used as input to Stage 3 |
| `matching_results.json` | Human-readable summary: per-frame match info, scores, cluster sizes |

   
## Stage 3 — Triangulation and Evaluation

Triangulate matched clusters into 3D poses using RANSAC, then compute MPJPE, PA-MPJPE and AP against ground truth.
This step utilizes [xrMoCap](https://github.com/openxrlab/xrmocap) as an external dependency. Please ensure you have cloned their official repository and configured the necessary environment following [official instruction](https://github.com/openxrlab/xrmocap/blob/main/docs/en/installation.md).

```
conda deactivate sam_3d_body
conda activate xrmocap

python projection_visualization.py \
    --intermediate_input  /path/to/intermediate_matched_clusters.pkl \
    --output_dir          /path/to/output \
    --gt_file             /path/to/keypoints3d_GT.npz
```

**Ground truth file format (`keypoints3d_GT.npz`):**

| Key | Shape | Description |
|---|---|---|
| `pose` | `(n_frames, n_persons, n_keypoints, 4)` | World-space 3D coordinates `[x, y, z, 1.0]`, unit: metres |
| `mask` | `(n_frames, n_persons, n_keypoints)` | Validity mask — `1` valid, `0` occluded / missing |
| `convention` | string scalar | Keypoint format name |

Persons with `mask.sum() == 0` are placeholder slots and are skipped during evaluation. Only joints where `mask == 1` contribute to MPJPE.
For more file format information, please refer to [xrmocap official instructions](https://github.com/openxrlab/xrmocap/blob/main/docs/en/data_structure/keypoints.md)


**Output:**

| File | Description |
|---|---|
| `pred_keypoints3d.npz` | Triangulated 3D poses per frame |
| `triangulation_<timestamp>.log` | Evaluation log with per-frame MPJPE and PA-MPJPE |

**Metrics report:**

- **MPJPE** — Mean Per Joint Position Error (mm)
- **PA-MPJPE** — Procrustes-Aligned MPJPE (mm), removes global rotation/scale/translation
- **APδ**  — Average Precision at threshold δ mm

---

## Camera Parameter Format

Camera JSON files are expected to contain the following fields:

```json
{
    "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "extrinsic_r": [[...], [...], [...]],
    "extrinsic_t": [tx, ty, tz],
    "k1": 0.0, "k2": 0.0, "k3": 0.0,
    "p1": 0.0, "p2": 0.0,
    "height": height,
    "width": width,
}
```

Alternatively, a 4×4 homogeneous `extrinsic` matrix can be used in place of `extrinsic_r` + `extrinsic_t`:

```json
{
    "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "extrinsic": [[r00, r01, r02, tx], [r10, r11, r12, ty],
                  [r20, r21, r22, tz], [0,   0,   0,   1]],
    "k1": 0.0, "k2": 0.0,
    "height": height,
    "width": width,
}
```

All distortion coefficients are optional and default to `0.0` if absent.


## Citation

If you find this work useful, please cite:

```bibtex

```
