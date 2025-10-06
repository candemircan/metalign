#!/bin/bash
#SBATCH --account=hai_1065
#SBATCH --nodes=1
#SBATCH --output=logs/train_notmeta_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

module load CUDA/12

for file in data/configs/notmeta*toml; do

srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/train_notmeta.py --config_file $file

done

wait
