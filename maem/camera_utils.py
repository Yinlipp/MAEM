"""Camera parameter loading and coordinate transformation utilities."""

import json
from typing import Dict, Tuple

import numpy as np


def load_camera_params(camera_param_path: str) -> Dict:
    """Load camera parameters from JSON file."""
    with open(camera_param_path, 'r') as f:
        return json.load(f)


def get_transform_matrix(camera_params: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (R, t, K); accepts either {extrinsic_r, extrinsic_t} or a 4x4 {extrinsic}."""
    if 'extrinsic_r' in camera_params and 'extrinsic_t' in camera_params:
        R = np.array(camera_params['extrinsic_r'])
        t = np.array(camera_params['extrinsic_t'])
    else:
        extrinsic = np.array(camera_params['extrinsic'])
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
    K = np.array(camera_params['intrinsic'])[:3, :3]
    return R, t, K


def transform_to_world(keypoints3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera to world: P_world = R^T @ (P_cam - t)."""
    t = np.array(t).ravel()
    return (keypoints3d - t) @ R
