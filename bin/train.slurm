#!/bin/bash
#SBATCH --account=hai_1065
#SBATCH --nodes=1
#SBATCH --output=logs/train_%j.out
#SBATCH --time=10:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

module load CUDA/12

srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/train.py --config_file data/configs/0.toml &
srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/train.py --config_file data/configs/1.toml &
srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/train.py --config_file data/configs/2.toml &
srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/train.py --config_file data/configs/3.toml &

wait