#!/bin/bash
#SBATCH -p ubuntu
#SBATCH -w nevera
#SBATCH -c 16
#SBATCH --gres=gpu:1


PYTHONPATH=/home/y_li/workspace7/xrmocap:$PYTHONPATH python tools/mview_mperson_topdown_estimator.py \
      --estimator_config 'configs/sportscenter/mview_mperson_topdown_estimator.py' \
      --image_and_camera_param 'xrmocap_data/xrmocap_humanm3_basketball1_split2/image_and_cam_param.txt' \
      --output_dir 'output/estimation/humanm3-bas1' \
      --enable_log_file \
      --save_perception_2d \
      --save_association
    
   