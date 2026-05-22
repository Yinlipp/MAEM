"""Epipolar geometry: fundamental matrix, epipolar cost, DLT triangulation."""

from typing import Optional

import numpy as np


def compute_fundamental_matrix(K1: np.ndarray, R1: np.ndarray, t1: np.ndarray,
                               K2: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Compute fundamental matrix F such that x2^T F x1 = 0.

    Camera model: P_cam = R @ P_world + t  (world-to-camera).
    """
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    tx, ty, tz = t_rel.ravel()
    t_cross = np.array([[0, -tz, ty],
                         [tz,  0, -tx],
                         [-ty, tx,  0]])
    E = t_cross @ R_rel
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F


def compute_epipolar_cost(verts_2d_1: np.ndarray, verts_2d_2: np.ndarray,
                          F: np.ndarray, epi_threshold: float = 20.0, logger=None) -> float:
    """Compute normalized mean symmetric epipolar distance in [0, 1].

    Args:
        verts_2d_1:    (N, 2) points in view 1 (pixel coords)
        verts_2d_2:    (N, 2) points in view 2 (pixel coords)
        F:             (3, 3) fundamental matrix (view1 → view2)
        epi_threshold: pixel threshold for normalization

    Returns:
        Normalized cost in [0, 1]. Returns 1.0 if inputs are invalid.
    """
    if verts_2d_1 is None or verts_2d_2 is None:
        return 1.0

    N = min(len(verts_2d_1), len(verts_2d_2))
    p1 = verts_2d_1[:N]
    p2 = verts_2d_2[:N]
    valid = (~np.any(np.isnan(p1), axis=1)) & (~np.any(np.isnan(p2), axis=1))
    if valid.sum() < 1:
        return 1.0

    p1, p2 = p1[valid], p2[valid]
    ones = np.ones((len(p1), 1))
    p1_h = np.hstack([p1, ones])
    p2_h = np.hstack([p2, ones])

    l2 = (F @ p1_h.T).T
    l1 = (F.T @ p2_h.T).T

    d2 = np.abs(np.sum(p2_h * l2, axis=1)) / (np.sqrt(l2[:, 0]**2 + l2[:, 1]**2) + 1e-8)
    d1 = np.abs(np.sum(p1_h * l1, axis=1)) / (np.sqrt(l1[:, 0]**2 + l1[:, 1]**2) + 1e-8)
    mean_dist = (np.mean(d1) + np.mean(d2)) / 2.0

    return float(min(mean_dist / epi_threshold, 1.0))


def triangulate_point_dlt(P1: np.ndarray, P2: np.ndarray,
                          pt1: np.ndarray, pt2: np.ndarray) -> Optional[np.ndarray]:
    """Triangulate a 3D point from two undistorted 2D observations using DLT.

    Args:
        P1, P2: (3, 4) projection matrices K @ [R | t]
        pt1, pt2: (2,) undistorted pixel coordinates

    Returns:
        (3,) 3D point in world coordinates, or None if degenerate.
    """
    A = np.array([
        pt1[0] * P1[2] - P1[0],
        pt1[1] * P1[2] - P1[1],
        pt2[0] * P2[2] - P2[0],
        pt2[1] * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    if abs(X_h[3]) < 1e-10:
        return None
    return X_h[:3] / X_h[3]


def compute_reprojection_error(P: np.ndarray, pt_3d: np.ndarray, pt_2d: np.ndarray) -> float:
    """Reproject a 3D point and compute pixel distance to the observed 2D point.

    Args:
        P:     (3, 4) projection matrix K @ [R | t]
        pt_3d: (3,) 3D point in world coordinates
        pt_2d: (2,) undistorted observed pixel coordinates

    Returns:
        Reprojection error in pixels.
    """
    X_h = np.append(pt_3d, 1.0)
    x_h = P @ X_h
    if abs(x_h[2]) < 1e-10:
        return float('inf')
    x_proj = x_h[:2] / x_h[2]
    return float(np.linalg.norm(x_proj - pt_2d))
