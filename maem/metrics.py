"""Evaluation metrics: MPJPE and PA-MPJPE."""

from typing import Optional, Tuple
import numpy as np


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute Mean Per Joint Position Error (MPJPE) in mm."""
    if mask is None:
        mask = np.ones(pred.shape[0], dtype=bool)
    distances = np.linalg.norm((pred - gt) * 1000, axis=1)
    valid_distances = distances[mask.astype(bool)]
    return float('inf') if len(valid_distances) == 0 else float(np.mean(valid_distances))


def _compute_similarity_transform(S1: np.ndarray, S2: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute a similarity transform (sR, t) that takes S1 closest to S2 (Procrustes)."""
    mu1, mu2 = S1.mean(axis=0), S2.mean(axis=0)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = np.sum(X1 ** 2)
    K = X1.T @ X2
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U @ V.T))
    R = V @ Z @ U.T
    scale = np.trace(R @ K) / var1 if var1 >= 1e-10 else 1.0
    t = mu2 - scale * (R @ mu1)
    return scale, R, t


def compute_pa_mpjpe(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute Procrustes Aligned MPJPE in mm."""
    if len(pred) == 0 or len(gt) == 0:
        return float('inf')
    finite_pred = np.all(np.isfinite(pred), axis=1)
    finite_gt = np.all(np.isfinite(gt), axis=1)
    valid = finite_pred & finite_gt
    if mask is not None:
        valid = valid & mask.astype(bool)
    if valid.sum() < 3:
        return float('inf')
    s, R, t = _compute_similarity_transform(pred[valid], gt[valid])
    pred_aligned = s * (R @ pred.T).T + t
    distances = np.linalg.norm(pred_aligned - gt, axis=1) * 1000
    return float(np.mean(distances[valid]))
