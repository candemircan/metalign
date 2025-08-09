#!/bin/bash
#SBATCH --account=hai_1065
#SBATCH --nodes=1
#SBATCH --output=logs/extract_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

module load CUDA/12

srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/extract_backbone.py coco &
srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/extract_backbone.py things &
srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/extract_sae.py coco &
srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/extract_sae.py things &

wait