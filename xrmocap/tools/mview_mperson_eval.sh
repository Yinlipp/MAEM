#!/bin/bash
#SBATCH -p ubuntu
#SBATCH -w nevera
#SBATCH -c 16
#SBATCH --gres=gpu:1

PYTHONPATH=/home/y_li/workspace7/xrmocap:$PYTHONPATH python tools/mview_mperson_evaluation.py \
      --evaluation_config 'configs/mvpose_tracking/sportscenter/eval_keypoints3d.py' \
      --enable_log_file
