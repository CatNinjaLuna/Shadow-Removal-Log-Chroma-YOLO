#!/bin/bash
#SBATCH --job-name=fuse-yolo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/fuse-yolo-%j.out

# 1) Load shell (so conda works)
source ~/.bashrc

# 2) Activate conda env
conda activate SR-shadow-removal

# 3) Go to project directory that contains train_fuse_log_chroma_resume.py
cd /projects/SuperResolutionData/carolinali-shadowRemoval/Y_Channel_Exchange

# 4) Train (auto-resume if runs/train/exp/weights/last.pt exists)
python train_fuse_log_chroma_resume.py --epochs 300

# Submit with:
#   sbatch fuse_log_chroma_yolo.sbatch
