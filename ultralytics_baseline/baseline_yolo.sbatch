#!/bin/bash
#SBATCH --job-name=logc-yolo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/logc-%j.out

# 1) Load shell (so conda works)
source ~/.bashrc

# 2) Activate conda env
conda activate SR-shadow-removal

# 3) Go to project dir that has train_baseline_resume.py
cd /projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics_baseline

# 4) Train (auto-resume; NO per-epoch validation now)
python train_baseline_resume.py --epochs 300

# sbatch baseline_yolo.sbatch
