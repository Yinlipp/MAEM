"""Evaluation metrics: MPJPE, PA-MPJPE, and AP_delta."""

from typing import List, Optional, Tuple
import numpy as np


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute Mean Per Joint Position Error (MPJPE) in mm."""
    if mask is None:
        mask = np.ones(pred.shape[0], dtype=bool)
    distances = np.linalg.norm((pred - gt) * 1000, axis=1)
    valid_distances = distances[mask.astype(bool)]
    return float('inf') if len(valid_distances) == 0 else float(np.mean(valid_distances))


def _compute_similarity_transform(S1: np.ndarray, S2: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Procrustes: similarity transform (sR, t) taking S1 closest to S2."""
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


def compute_ap_delta(entries: List[Tuple[float, float, int, Optional[int]]],
                     delta: float, total_gt: int) -> float:
    """AP_delta via all-point interpolation over confidence-ranked Hungarian matches.

    TP iff MPJPE < delta and the GT isn't already claimed by a higher-confidence
    prediction; unassigned (surplus) predictions are always FP.
    """
    if total_gt == 0 or not entries:
        return 0.0

    ranked = sorted(entries, key=lambda e: e[0], reverse=True)
    claimed = set()
    tp = np.zeros(len(ranked))
    fp = np.zeros(len(ranked))

    for i, (conf, mpjpe, frame_idx, gt_idx) in enumerate(ranked):
        key = (frame_idx, gt_idx)
        if gt_idx is not None and mpjpe < delta and key not in claimed:
            tp[i] = 1
            claimed.add(key)
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # envelope precision non-increasing, then integrate
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
