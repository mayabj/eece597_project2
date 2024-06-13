#!/bin/bash
#SBATCH --time 48:00:00
#SBATCH --gres gpu:0
#SBATCH -p batch
#SBATCH -c 1

eval "$(conda shell.bash hook)"
conda activate env1
python3 /home/mayabj/project_scripts/train_knn_save.py --decision_point $1
