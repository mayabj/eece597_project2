#!/usr/bin/env python3

import subprocess
import time

decision_points_list = [
    "100",
    "200",
    "300" 
]

for decision_point in decision_points_list:
    subprocess.run(f'echo "Training for decision point {decision_point}"', shell=True)
    subprocess.run(
            f'echo "sbatch --job-name=train_{decision_point} --mem=8G --time=48:00:00 --output=ml_out/train_knn_{decision_point}.out ml_compile.sh {decision_point}"',
            shell=True
        )
    subprocess.run(
            f"sbatch --job-name=train_{decision_point} --mem=8G --time=48:00:00 --output=ml_out/train_knn_{decision_point}.out ml_compile.sh {decision_point}",
            shell=True
        )
