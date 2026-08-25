"""Data loading: NPZ predictions, frame images, and per-view detections."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .camera_utils import get_transform_matrix, transform_to_world

# SportCenter's 13-point convention (MHR indices, see sam_3d_body/metadata/mhr70.py)
KEYPOINTS_IDX = [0, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]

# Human-M3's 15-point convention (lib/dataset/human_m3.py's valid_joints_def); pelvis
# is derived as the left/right hip midpoint, same as SAM-3D-Body's own pelvis_idx
_HUMANM3_RAW_IDX = [9, 10, 11, 12, 13, 14, 69, 0, 5, 6, 7, 8, 62, 41]


def _extract_humanm3_keypoints(full_kpts: np.ndarray) -> np.ndarray:
    pelvis = (full_kpts[9] + full_kpts[10]) / 2.0
    return np.concatenate([pelvis[None], full_kpts[_HUMANM3_RAW_IDX]], axis=0)


def downsample_vertices(verts: Optional[np.ndarray], rate: int) -> Optional[np.ndarray]:
    """Take every `rate`-th vertex (fixed stride keeps the same indices across views)."""
    if verts is None or rate <= 1:
        return verts
    return verts[::rate]


def load_prediction_npz(npz_path: str, person_idx: int = 0,
                        keypoint_convention: str = 'sportcenter_13',
                        vertex_sample_rate: int = 1) -> Dict:
    """Load one person's prediction (by index into 'outputs') from a SAM-3D-Body NPZ."""
    data = np.load(npz_path, allow_pickle=True)
    if 'outputs' not in data:
        raise ValueError(f"'outputs' key not found in {npz_path}")

    outputs = data['outputs']
    if len(outputs) == 0:
        raise ValueError(f"No detections in {npz_path}")
    if person_idx >= len(outputs):
        raise ValueError(f"Person index {person_idx} out of range ({len(outputs)} detections)")

    person_data = outputs[person_idx]
    frame_dict = (person_data.item()
                  if hasattr(person_data, 'item') and isinstance(person_data.item(), dict)
                  else person_data)
    if not isinstance(frame_dict, dict):
        raise ValueError(f"Unexpected format for outputs[{person_idx}]: {type(person_data)}")

    for field in ('pred_keypoints_3d', 'bbox_score', 'pred_cam_t'):
        if field not in frame_dict:
            raise ValueError(f"'{field}' missing in outputs[{person_idx}]")

    extract = _extract_humanm3_keypoints if keypoint_convention == 'humanm3_15' \
        else (lambda kpts: kpts[KEYPOINTS_IDX])

    keypoints3d = np.array(frame_dict['pred_keypoints_3d'])
    if keypoints3d.shape[0] > 21:
        keypoints3d = extract(keypoints3d)

    pred_keypoints_2d = frame_dict.get('pred_keypoints_2d')
    if pred_keypoints_2d is not None:
        pred_keypoints_2d = np.array(pred_keypoints_2d)
        if pred_keypoints_2d.shape[0] > 21:
            pred_keypoints_2d = extract(pred_keypoints_2d)

    pred_keypoints_2d_verts = frame_dict.get('pred_keypoints_2d_verts')
    if pred_keypoints_2d_verts is not None:
        pred_keypoints_2d_verts = downsample_vertices(np.array(pred_keypoints_2d_verts), vertex_sample_rate)

    return {
        'keypoints3d': keypoints3d,
        'bbox_score': frame_dict['bbox_score'],
        'pred_cam_t': np.array(frame_dict['pred_cam_t']),
        'bbox': frame_dict.get('bbox'),
        'pred_keypoints_2d': pred_keypoints_2d,
        'pred_keypoints_2d_verts': pred_keypoints_2d_verts,
        'pred_vertices': frame_dict.get('pred_vertices'),
    }


def find_frame_image(folder: str, frame_num: int) -> Optional[str]:
    """Find the image for a frame number, trying '30048'/'030048'/'img_030048'/'img_30048'."""
    IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    if not os.path.isdir(folder):
        return None
    target_stems = {
        str(frame_num),
        f'{frame_num:06d}',
        f'img_{frame_num:06d}',
        f'img_{frame_num}',
    }
    for f in os.listdir(folder):
        name, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXTS and name in target_stems:
            return os.path.join(folder, f)
    return None


def load_view_data(view_name: str, frame_num: int, output_dir: str, scene_dir: str,
                   camera_params_dict: Dict,
                   keypoint_convention: str = 'sportcenter_13',
                   vertex_sample_rate: int = 1) -> Tuple[str, Optional[List[Dict]]]:
    """Load all person detections for one view/frame; (view_name, None) on failure."""
    npz_path = os.path.join(output_dir, view_name, 'npz', f'{frame_num:06d}.npz')
    if not os.path.exists(npz_path):
        npz_path = os.path.join(output_dir, view_name, 'npz', f'{frame_num}.npz')
    if not os.path.exists(npz_path):
        return view_name, None

    try:
        data = np.load(npz_path, allow_pickle=True)
        if 'outputs' not in data:
            return view_name, None
        num_detections = len(data['outputs'])

        cam_params = camera_params_dict[view_name]
        R, t, _ = get_transform_matrix(cam_params)

        view_poses = []
        for person_idx in range(num_detections):
            try:
                pred_data = load_prediction_npz(npz_path, person_idx=person_idx,
                                                keypoint_convention=keypoint_convention,
                                                vertex_sample_rate=vertex_sample_rate)
                keypoints3d_cam = pred_data['keypoints3d']

                if keypoints3d_cam.ndim != 2 or keypoints3d_cam.shape[1] != 3:
                    continue

                keypoints3d_world = transform_to_world(
                    keypoints3d_cam + pred_data['pred_cam_t'], R, t
                )
                view_poses.append({
                    'keypoints': keypoints3d_world,
                    'scores': pred_data['bbox_score'],
                    'bbox': pred_data.get('bbox'),
                    'pred_keypoints_2d': pred_data.get('pred_keypoints_2d'),
                    'pred_keypoints_2d_verts': pred_data.get('pred_keypoints_2d_verts'),
                })
            except Exception as e:
                print(f"Warning: Failed to process person {person_idx} in {view_name}: {e}")

        return view_name, view_poses if view_poses else None

    except Exception as e:
        print(f"Error loading view {view_name}: {e}")
        return view_name, None
