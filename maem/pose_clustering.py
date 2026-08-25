"""Union-Find clustering, conflict resolution, and cluster validation."""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .camera_utils import get_transform_matrix
from .epipolar_geometry import compute_fundamental_matrix, compute_epipolar_cost
from .metrics import compute_pa_mpjpe


def _build_person_clusters(pairwise_matches):
    """Union-Find clustering; same-camera conflicts resolved later, not here."""
    person_to_cluster = {}
    cluster_to_persons = defaultdict(list)
    next_cluster_id = [0]

    def get_cluster(view, person_idx):
        key = (view, person_idx)
        if key not in person_to_cluster:
            person_to_cluster[key] = next_cluster_id[0]
            cluster_to_persons[next_cluster_id[0]].append(key)
            next_cluster_id[0] += 1
        return person_to_cluster[key]

    def merge_clusters(cluster1, cluster2):
        if cluster1 == cluster2:
            return cluster1
        for person_key in cluster_to_persons[cluster2]:
            person_to_cluster[person_key] = cluster1
            cluster_to_persons[cluster1].append(person_key)
        cluster_to_persons[cluster2] = []
        return cluster1

    for (view1, view2), matches in pairwise_matches.items():
        for p1_idx, p2_idx, _ in matches:
            merge_clusters(get_cluster(view1, p1_idx), get_cluster(view2, p2_idx))

    return cluster_to_persons, person_to_cluster


def _resolve_cluster_conflicts(person_list, pairwise_matches, logger=None):
    """Drop the conflicting detection with highest max-PA-MPJPE cost (Eq. 8), not mean."""
    current = list(person_list)

    while True:
        view_to_persons = defaultdict(list)
        for view, person_idx in current:
            view_to_persons[view].append(person_idx)

        conflicting_views = {v: ps for v, ps in view_to_persons.items() if len(ps) > 1}
        if not conflicting_views:
            break

        conflicting_detections = {(v, p) for v, ps in conflicting_views.items() for p in ps}
        current_set = set(current)

        detection_costs = {}
        for det in conflicting_detections:
            view, person_idx = det
            costs = []
            for (v1, v2), matches in pairwise_matches.items():
                for p1_idx, p2_idx, pa_cost in matches:
                    if v1 == view and p1_idx == person_idx and (v2, p2_idx) in current_set:
                        costs.append(pa_cost)
                    elif v2 == view and p2_idx == person_idx and (v1, p1_idx) in current_set:
                        costs.append(pa_cost)
            detection_costs[det] = float(max(costs)) if costs else float('inf')

        worst = max(detection_costs, key=lambda d: detection_costs[d])
        if logger:
            logger.info(f"    Conflict resolved: removed {worst} "
                        f"(c_PA={detection_costs[worst]:.1f}mm)")
        current.remove(worst)

    return current


def _get_epipolar_cost_for_pair(v1, v2, view_to_person, poses_world_dict,
                                camera_params_dict, epi_threshold, logger):
    """Compute epipolar cost for one view pair. Returns cost in [0,1], or None."""
    if v1 not in camera_params_dict or v2 not in camera_params_dict:
        return None
    p1_data = poses_world_dict[v1][view_to_person[v1]]
    p2_data = poses_world_dict[v2][view_to_person[v2]]
    verts1 = p1_data.get('pred_keypoints_2d_verts') or p1_data.get('pred_keypoints_2d')
    verts2 = p2_data.get('pred_keypoints_2d_verts') or p2_data.get('pred_keypoints_2d')
    if verts1 is None or verts2 is None:
        return None
    R1, t1, K1 = get_transform_matrix(camera_params_dict[v1])
    R2, t2, K2 = get_transform_matrix(camera_params_dict[v2])
    F = compute_fundamental_matrix(K1, R1, t1, K2, R2, t2)
    return compute_epipolar_cost(verts1, verts2, F, epi_threshold, logger)


def _trim_cluster_for_consistency(view_to_person, poses_world_dict, camera_params_dict,
                                  epi_threshold, min_views, logger):
    """Trim inconsistent views until all remaining pairs are epipolar-consistent."""
    if camera_params_dict is None:
        return view_to_person

    current = dict(view_to_person)

    while len(current) >= min_views:
        views = sorted(current.keys())
        failed_pairs = [
            (v1, v2)
            for i, v1 in enumerate(views)
            for v2 in views[i + 1:]
            if _get_epipolar_cost_for_pair(v1, v2, current, poses_world_dict,
                                           camera_params_dict, epi_threshold, logger) is not None
            and _get_epipolar_cost_for_pair(v1, v2, current, poses_world_dict,
                                            camera_params_dict, epi_threshold, logger) >= 1.0
        ]
        if not failed_pairs:
            return current

        fail_count = {v: 0 for v in views}
        for v1, v2 in failed_pairs:
            fail_count[v1] += 1
            fail_count[v2] += 1

        worst_view = max(fail_count, key=lambda v: (fail_count[v], v))
        if logger:
            logger.info(f"    Cluster trimmed: removed {worst_view} "
                        f"({fail_count[worst_view]} failed pair(s)), "
                        f"{len(current) - 1} views remaining")
        del current[worst_view]

    return None


def _filter_and_build_valid_clusters(cluster_to_persons, poses_world_dict, pairwise_matches,
                                     min_views, logger,
                                     camera_params_dict=None, epi_threshold=10.0):
    """Filter clusters by min_views and same-camera conflicts; return valid cluster list."""
    valid_clusters = []

    for cluster_id, person_list in cluster_to_persons.items():
        if not person_list:
            continue
        if len(set(v for v, _ in person_list)) < min_views:
            continue

        view_counts = defaultdict(int)
        for view, _ in person_list:
            view_counts[view] += 1
        if any(c > 1 for c in view_counts.values()):
            if logger:
                logger.info(f"    Cluster {cluster_id}: conflict detected, resolving...")
            person_list = _resolve_cluster_conflicts(person_list, pairwise_matches, logger)

        if len(set(v for v, _ in person_list)) < min_views:
            continue

        view_to_person = {view: person_idx for view, person_idx in person_list}
        cluster_poses = {view: poses_world_dict[view][person_idx]
                         for view, person_idx in view_to_person.items()}

        total_error = error_count = 0.0
        for (v1, v2), matches in pairwise_matches.items():
            if v1 not in view_to_person or v2 not in view_to_person:
                continue
            p1_idx, p2_idx = view_to_person[v1], view_to_person[v2]
            for m_p1, m_p2, error in matches:
                if m_p1 == p1_idx and m_p2 == p2_idx:
                    total_error += error
                    error_count += 1
                    break

        valid_clusters.append({
            'poses': cluster_poses,
            'num_views': len(view_to_person),
            'avg_error': total_error / error_count if error_count > 0 else 0.0
        })

    if logger:
        logger.info(f"    Found {len(valid_clusters)} valid person clusters")
        for i, cluster in enumerate(valid_clusters):
            views = ', '.join(cluster['poses'].keys())
            logger.info(f"      Person {i}: {cluster['num_views']} views ({views}), "
                        f"avg_error={cluster['avg_error']:.1f}mm")

    return valid_clusters


def match_poses_across_views(poses_world_dict_all_people: Dict,
                             min_views: int = 3,
                             max_error_threshold: Optional[float] = None,
                             bbox_score_threshold: float = 0.9,
                             matching_mode: str = 'epi_gate',
                             camera_params_dict: Optional[Dict] = None,
                             epi_threshold: float = 8.0,
                             repr_threshold: float = 30.0,
                             logger=None) -> Tuple[List[Dict], float, float, float]:
    """Hungarian matching + Union-Find clustering across views.

    matching_mode: pa_mpjpe | sparse | epi_gate (repr+epipolar) | repr_gate | epi_gate_only

    Returns (valid_clusters, repr_gate_time_sec, epi_gate_time_sec, clustering_time_sec).
    """
    from .pose_matching import _filter_poses_by_quality, _compute_pairwise_matches

    if logger:
        logger.info(f"    Starting cross-view matching: min_views={min_views}, "
                    f"bbox_score_threshold={bbox_score_threshold}")

    poses_world_dict_all_people = _filter_poses_by_quality(
        poses_world_dict_all_people, bbox_score_threshold, logger
    )

    view_names = sorted(poses_world_dict_all_people.keys())
    if len(view_names) < 2:
        if logger:
            logger.info(f"    Only {len(view_names)} view(s) available, skipping")
        return [], 0.0, 0.0, 0.0

    pairwise_matches, repr_gate_time_sec, epi_gate_time_sec = _compute_pairwise_matches(
        poses_world_dict_all_people, view_names, matching_mode,
        max_error_threshold, camera_params_dict,
        epi_threshold=epi_threshold,
        repr_threshold=repr_threshold, logger=logger
    )

    if not pairwise_matches:
        if logger:
            logger.info(f"    No pairwise matches found")
        return [], repr_gate_time_sec, epi_gate_time_sec, 0.0

    _t0 = time.perf_counter()
    cluster_to_persons, _ = _build_person_clusters(pairwise_matches)

    valid_clusters = _filter_and_build_valid_clusters(
        cluster_to_persons, poses_world_dict_all_people, pairwise_matches, min_views, logger,
        camera_params_dict=camera_params_dict, epi_threshold=epi_threshold
    )
    clustering_time_sec = time.perf_counter() - _t0

    return valid_clusters, repr_gate_time_sec, epi_gate_time_sec, clustering_time_sec
