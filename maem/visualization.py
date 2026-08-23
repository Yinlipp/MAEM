"""Visualization: draw matched clusters on images."""

import os
from typing import Dict, List, Optional

import cv2
import numpy as np

from .data_io import find_frame_image


def _draw_bbox_with_label(img: np.ndarray, bbox, label: str, color, font_scale=0.6, thickness=2):
    """Draw a bounding box with a filled label on an image."""
    if bbox is None or len(bbox) < 4:
        return
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - baseline - 5), font, font_scale, (255, 255, 255), thickness)


def visualize_matched_clusters(matched_clusters: List[Dict], frame_num: int,
                               scene_dir: str, output_dir: str,
                               view_names: Optional[List[str]] = None,
                               n_views: int = 8, logger=None):
    """Draw bboxes for each matched person cluster on that frame's per-view images."""
    os.makedirs(output_dir, exist_ok=True)

    if view_names is None:
        view_names = [f'camera_{i}' for i in range(n_views)]

    if logger:
        logger.info(f"  Visualizing {len(matched_clusters)} matched clusters for frame {frame_num}")

    view_images = {}
    for view_name in view_names:
        img_path = find_frame_image(os.path.join(scene_dir, view_name), frame_num)
        if img_path is None:
            continue
        img = cv2.imread(img_path)
        if img is not None:
            view_images[view_name] = img

    if not view_images:
        if logger:
            logger.info(f"    No images loaded, skipping visualization")
        return

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 128, 0), (0, 128, 255), (128, 255, 0)
    ]

    for person_idx, cluster in enumerate(matched_clusters):
        poses = cluster['poses']
        matched_views = sorted(v for v in poses if v in view_images)
        if not matched_views:
            continue

        color = colors[person_idx % len(colors)]
        person_view_images = []

        for view_name in matched_views:
            img_copy = view_images[view_name].copy()
            bbox = poses[view_name].get('bbox')
            score = poses[view_name].get('scores', 0.0)
            label = f'P{person_idx} {view_name} ({score:.2f})'
            _draw_bbox_with_label(img_copy, bbox, label, color)
            person_view_images.append(img_copy)

        if not person_view_images:
            continue

        max_h = max(img.shape[0] for img in person_view_images)
        resized = []
        for img in person_view_images:
            if img.shape[0] != max_h:
                scale = max_h / img.shape[0]
                resized.append(cv2.resize(img, (int(img.shape[1] * scale), max_h)))
            else:
                resized.append(img)

        concatenated = cv2.hconcat(resized)
        frame_folder = os.path.join(output_dir, f'frame_{frame_num:06d}')
        os.makedirs(frame_folder, exist_ok=True)
        cv2.imwrite(os.path.join(frame_folder, f'person_{person_idx}_concat.png'), concatenated)
