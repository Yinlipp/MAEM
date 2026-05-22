import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together
import argparse

logger = logging.getLogger(__name__)

def save_outputs(outputs: Any, output_path: str) -> None:
    """Save model outputs to npz file."""
    try:
        # Convert outputs to numpy arrays if needed
        def convert_to_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj
            elif isinstance(obj, list):
                return np.array(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_numpy(value) for key, value in obj.items()}
            else:
                return obj

        # Prepare data for npz saving
        # outputs is a list of dictionaries, one per detected person
        npz_data = {
            'outputs': np.array(outputs, dtype=object)
        }

        # Save to NPZ file
        np.savez(output_path, **npz_data)
        logger.info(f"Saved outputs to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save outputs to {output_path}: {e}")

def process_image(img_path: str, img_output_path: str, npz_output_path: str,
                  estimator, save_visualization: bool = True) -> bool:
    """
    Process a single image and save results.

    Returns:
        bool: True if processing succeeded, False otherwise
    """
    try:
        # Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            logger.error(f"Failed to load image: {img_path}")
            return False

        logger.info(f"Processing {img_path}")

        # Process image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        outputs = estimator.process_one_image(img_rgb)

        # Handle results
        if len(outputs) == 0:
            logger.warning(f"No person detected in {img_path}")
            if save_visualization:
                cv2.imwrite(img_output_path, img_bgr)
            return False

        # Visualize and save
        if save_visualization:
            rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
            cv2.imwrite(img_output_path, rend_img.astype(np.uint8))

        # Save outputs to npz
        save_outputs(outputs, npz_output_path)

        return True

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return False

def get_frame_number(filename: str) -> str:
    """Extract frame number from filename."""
    try:
        return filename.split('_')[1].split('.')[0]
    except IndexError:
        logger.warning(f"Could not extract frame number from {filename}, using full name")
        return Path(filename).stem

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Initializing SAM-3D-Body estimator...")

    parser = argparse.ArgumentParser(description='SAM 3d body inference')

    parser.add_argument('--checkpoints_dir', type=str, default='/home/y_li/workspace7/MAEM/sam-3d-body/checkpoints/',
                         help='checkpoints folder path')
    parser.add_argument('--img_root_folder', type=str, default='/home/y_li/workspace7/humanm3/test/plaza/images',
                        help='the folder path containing subfolders of images from different views')
    parser.add_argument('--output_dir', default='/home/y_li/workspace7/MAEM/sam-3d-body/output/plaza',
                         help='output dir')
    
    args = parser.parse_args()

    # Configuration
    checkpoint_path = os.path.join(args.checkpoints_dir, "model.ckpt")
    mhr_path = os.path.join(args.checkpoints_dir, "assets", "mhr_model.pt")

    # Dataset Humanm3
    imgs_root_folder = args.img_root_folder
    # imgs_root_folder = '/home/y_li/workspace7/humanm3/test/basketball1/split2/images/'

    # sportscenter 21380
    imgs_root_folder = '/home/y_li/workspace7/xrmocap/xrmocap_data/xrmocap_sportscenter/scene_0'
    
    out_folder = args.output_dir
    out_folder = "/home/y_li/workspace7/sam-3d-body/output/sportscenter_21380/"
    # out_folder = "/home/y_li/workspace7/sam-3d-body/output/humanm3//basketball1/split2/"

    # Create output folder
    os.makedirs(out_folder, exist_ok=True)


    try:
        estimator = setup_sam_3d_body(
            checkpoint_path=checkpoint_path,
            mhr_path=mhr_path,
            device="cuda"
        )
    except Exception as e:
        logger.error(f"Failed to initialize estimator: {e}")
        return

    logger.info(f"Processing images from {imgs_root_folder}")

    # Get all folders
    folders = [f for f in os.listdir(imgs_root_folder)
               if os.path.isdir(os.path.join(imgs_root_folder, f))]

    total_processed = 0
    total_failed = 0

    # Process each folder
    for folder_name in tqdm(folders, desc="Processing folders"):
        img_folder_path = os.path.join(imgs_root_folder, folder_name)

        # Create output folder for this camera view
        img_output_folder = os.path.join(out_folder, folder_name)
        npz_output_folder = os.path.join(img_output_folder, "npz")
        os.makedirs(img_output_folder, exist_ok=True)
        os.makedirs(npz_output_folder, exist_ok=True)

        # Get all image files
        image_files = [img for img in os.listdir(img_folder_path)
                       if img.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # Process each image
        for img_name in tqdm(image_files, desc=f"  {folder_name}", leave=False):
            img_path = os.path.join(img_folder_path, img_name)
            img_output_path = os.path.join(img_output_folder, img_name)

            frame_num = get_frame_number(img_name)
            npz_output_path = os.path.join(npz_output_folder, f'{frame_num}.npz')

            success = process_image(img_path, img_output_path, npz_output_path, estimator)

            if success:
                total_processed += 1
            else:
                total_failed += 1

    logger.info(f"Processing complete! Processed: {total_processed}, Failed: {total_failed}")

if __name__ == "__main__":
    main()


