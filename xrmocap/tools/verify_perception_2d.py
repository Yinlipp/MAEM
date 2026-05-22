"""Verify and inspect perception_2d.npz file format.

This script helps verify that the saved perception_2d.npz file has the correct
format and can be loaded by the evaluation dataset.
"""

import argparse
import numpy as np
import os.path as osp
from prettytable import PrettyTable


def verify_perception_2d(npz_path):
    """Verify perception_2d.npz file format.

    Args:
        npz_path (str): Path to perception_2d.npz file.
    """
    print(f"\n{'='*60}")
    print(f"Verifying: {npz_path}")
    print(f"{'='*60}\n")

    # Check file exists
    if not osp.exists(npz_path):
        print(f"❌ File not found: {npz_path}")
        return False

    # Load file
    try:
        data = np.load(npz_path, allow_pickle=True)
        print("✅ File loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return False

    # Check required fields
    print("📋 Checking required fields...")
    required_fields = ['bbox_convention', 'kps2d_convention']
    for field in required_fields:
        if field in data:
            print(f"  ✅ {field}: {data[field]}")
        else:
            print(f"  ❌ Missing required field: {field}")
            return False

    # Find all views
    view_indices = set()
    for key in data.keys():
        if key.startswith('bbox2d_view_'):
            view_idx = int(key.split('_')[-1])
            view_indices.add(view_idx)
        elif key.startswith('kps2d_view_'):
            view_idx = int(key.split('_')[-1])
            view_indices.add(view_idx)

    n_views = len(view_indices)
    print(f"\n📹 Found {n_views} views: {sorted(view_indices)}\n")

    # Create summary table
    table = PrettyTable()
    table.field_names = [
        'View', 'BBox Shape', 'Max Persons', 'Kps2d Shape',
        'Mask Shape', 'Keypoints'
    ]

    all_valid = True
    for view_idx in sorted(view_indices):
        bbox_key = f'bbox2d_view_{view_idx:02d}'
        kps2d_key = f'kps2d_view_{view_idx:02d}'
        mask_key = f'kps2d_mask_view_{view_idx:02d}'

        # Check bbox
        if bbox_key not in data:
            print(f"  ❌ Missing {bbox_key}")
            all_valid = False
            continue

        bbox_arr = data[bbox_key]
        n_frame, max_person, bbox_dim = bbox_arr.shape

        # Count actual persons (non-zero bboxes)
        bbox_scores = bbox_arr[..., -1]
        persons_per_frame = (bbox_scores > 0).sum(axis=1)
        max_persons = persons_per_frame.max()

        # Check keypoints
        if kps2d_key not in data:
            print(f"  ❌ Missing {kps2d_key}")
            all_valid = False
            continue

        kps2d_arr = data[kps2d_key]
        kps2d_shape = kps2d_arr.shape
        n_kps = kps2d_shape[-2] if len(kps2d_shape) >= 2 else 0

        # Check mask
        mask_shape = "N/A"
        if mask_key in data:
            mask_arr = data[mask_key]
            mask_shape = str(mask_arr.shape)
        else:
            print(f"  ⚠️  Missing {mask_key} (optional)")

        table.add_row([
            f'View {view_idx:02d}',
            f'{bbox_arr.shape}',
            f'{max_persons}',
            f'{kps2d_arr.shape}',
            mask_shape,
            f'{n_kps}'
        ])

    print(table)

    # Detailed info for first view
    if len(view_indices) > 0:
        first_view = sorted(view_indices)[0]
        print(f"\n📊 Detailed info for View {first_view:02d}:")

        bbox_key = f'bbox2d_view_{first_view:02d}'
        kps2d_key = f'kps2d_view_{first_view:02d}'

        bbox_arr = data[bbox_key]
        kps2d_arr = data[kps2d_key]

        print(f"  Frames: {bbox_arr.shape[0]}")
        print(f"  Max persons per frame: {bbox_arr.shape[1]}")
        print(f"  BBox format: {data['bbox_convention']}")
        print(f"  Keypoints convention: {data['kps2d_convention']}")
        print(f"  Keypoints per person: {kps2d_arr.shape[-2]}")

        # Show first frame stats
        print(f"\n  📸 First frame statistics:")
        frame_0_bbox = bbox_arr[0]
        valid_detections = frame_0_bbox[frame_0_bbox[:, -1] > 0]
        print(f"    Valid detections: {len(valid_detections)}")

        if len(valid_detections) > 0:
            print(f"    BBox scores: {valid_detections[:, -1]}")
            print(f"    First bbox: {valid_detections[0]}")

            # Show keypoints for first person
            frame_0_kps = kps2d_arr[0, 0]  # First frame, first person
            visible_kps = frame_0_kps[frame_0_kps[:, -1] > 0]
            print(f"    Visible keypoints (person 0): {len(visible_kps)}/{len(frame_0_kps)}")

    # Check scene directory structure
    scene_dir = osp.dirname(npz_path)
    print(f"\n📁 Checking scene directory: {scene_dir}")

    # Check for image_list files
    for view_idx in sorted(view_indices):
        img_list_path = osp.join(scene_dir, f'image_list_view_{view_idx:02d}.txt')
        if osp.exists(img_list_path):
            with open(img_list_path, 'r') as f:
                n_lines = sum(1 for _ in f)
            print(f"  ✅ image_list_view_{view_idx:02d}.txt ({n_lines} images)")
        else:
            print(f"  ⚠️  image_list_view_{view_idx:02d}.txt not found")

    if all_valid:
        print(f"\n{'='*60}")
        print("✅ Verification passed! File format is correct.")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("❌ Verification failed! Please check the errors above.")
        print(f"{'='*60}\n")

    return all_valid


def compare_with_reference(npz_path, reference_path):
    """Compare saved perception_2d with a reference file.

    Args:
        npz_path (str): Path to perception_2d.npz to verify.
        reference_path (str): Path to reference perception_2d.npz.
    """
    print(f"\n{'='*60}")
    print("Comparing with reference file...")
    print(f"{'='*60}\n")

    data = np.load(npz_path, allow_pickle=True)
    ref_data = np.load(reference_path, allow_pickle=True)

    print(f"Your file: {npz_path}")
    print(f"Reference: {reference_path}\n")

    # Compare keys
    your_keys = set(data.keys())
    ref_keys = set(ref_data.keys())

    print("🔑 Key comparison:")
    print(f"  Common keys: {len(your_keys & ref_keys)}")
    print(f"  Only in your file: {your_keys - ref_keys}")
    print(f"  Only in reference: {ref_keys - your_keys}")

    # Compare conventions
    print(f"\n📐 Convention comparison:")
    if 'bbox_convention' in data and 'bbox_convention' in ref_data:
        match = data['bbox_convention'] == ref_data['bbox_convention']
        symbol = "✅" if match else "❌"
        print(f"  {symbol} bbox_convention: {data['bbox_convention'].item()} vs {ref_data['bbox_convention'].item()}")

    if 'kps2d_convention' in data and 'kps2d_convention' in ref_data:
        match = data['kps2d_convention'] == ref_data['kps2d_convention']
        symbol = "✅" if match else "❌"
        print(f"  {symbol} kps2d_convention: {data['kps2d_convention'].item()} vs {ref_data['kps2d_convention'].item()}")

    # Compare shapes
    print(f"\n📏 Shape comparison:")
    common_keys = sorted(your_keys & ref_keys)
    for key in common_keys:
        if key.startswith('bbox2d_view_') or key.startswith('kps2d_view_') or key.startswith('kps2d_mask_view_'):
            your_shape = data[key].shape
            ref_shape = ref_data[key].shape
            match = your_shape == ref_shape
            symbol = "✅" if match else "⚠️"
            print(f"  {symbol} {key}: {your_shape} vs {ref_shape}")


def main():
    parser = argparse.ArgumentParser(
        description='Verify perception_2d.npz file format')
    parser.add_argument(
        '--npz_path',
        type=str,
        required=True,
        help='Path to perception_2d.npz file to verify')
    parser.add_argument(
        '--reference',
        type=str,
        default=None,
        help='Optional reference perception_2d.npz for comparison')
    args = parser.parse_args()

    # Verify file
    success = verify_perception_2d(args.npz_path)

    # Compare with reference if provided
    if args.reference and osp.exists(args.reference):
        compare_with_reference(args.npz_path, args.reference)
    elif args.reference:
        print(f"\n⚠️  Reference file not found: {args.reference}")

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
