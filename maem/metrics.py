"""Evaluation metrics: MPJPE, PA-MPJPE, and Human-M3's official AP/recall protocol."""

from typing import Dict, List, Optional, Tuple
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


def eval_list_to_ap_official(eval_list: List[Dict], total_gt: int, threshold_mm: float) -> float:
    """Human-M3's AP formula (lib/dataset/human_m3.py: _eval_list_to_ap), reproduced verbatim."""
    if total_gt == 0 or not eval_list:
        return 0.0
    ranked = sorted(eval_list, key=lambda k: k['score'], reverse=True)
    total_num = len(ranked)
    tp = np.zeros(total_num)
    fp = np.zeros(total_num)
    gt_det = []
    for i, item in enumerate(ranked):
        if item['mpjpe'] < threshold_mm and item['gt_id'] is not None and item['gt_id'] not in gt_det:
            tp[i] = 1
            gt_det.append(item['gt_id'])
        else:
            fp[i] = 1
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / (total_gt + 1e-5)
    precise = tp / (tp + fp + 1e-5)
    for n in range(total_num - 2, -1, -1):
        precise[n] = max(precise[n], precise[n + 1])
    precise = np.concatenate(([0], precise, [0]))
    recall = np.concatenate(([0], recall, [1]))
    index = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[index + 1] - recall[index]) * precise[index + 1]))


def eval_list_to_mpjpe_official(eval_list: List[Dict], threshold_mm: float = 500.0) -> float:
    """Human-M3's _eval_list_to_mpjpe: confidence-sorted, greedy-deduped mean MPJPE of TPs."""
    ranked = sorted(eval_list, key=lambda k: k['score'], reverse=True)
    gt_det = []
    mpjpes = []
    for item in ranked:
        if item['mpjpe'] < threshold_mm and item['gt_id'] is not None and item['gt_id'] not in gt_det:
            mpjpes.append(item['mpjpe'])
            gt_det.append(item['gt_id'])
    return float(np.mean(mpjpes)) if mpjpes else float('inf')


def eval_list_to_recall_official(eval_list: List[Dict], total_gt: int, threshold_mm: float = 500.0) -> float:
    """Human-M3's _eval_list_to_recall: any prediction within threshold counts, no dedup."""
    gt_ids = [e['gt_id'] for e in eval_list if e['mpjpe'] < threshold_mm and e['gt_id'] is not None]
    return len(set(gt_ids)) / total_gt if total_gt > 0 else 0.0
