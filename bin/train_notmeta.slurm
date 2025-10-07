#!/bin/bash
#SBATCH --account=hai_1065
#SBATCH --nodes=1
#SBATCH --output=logs/train_notmeta_%j.out
#SBATCH --time=10:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch train_notmeta.slurm {ce|focal}"
    echo "  ce    - Run cross-entropy loss experiments"
    echo "  focal - Run focal loss experiments"
    exit 1
fi

LOSS_TYPE=$1

if [ "$LOSS_TYPE" != "ce" ] && [ "$LOSS_TYPE" != "focal" ]; then
    echo "Error: Invalid loss type '$LOSS_TYPE'. Must be 'ce' or 'focal'"
    exit 1
fi

module load CUDA/12

if [ "$LOSS_TYPE" = "ce" ]; then
    for file in data/configs/notmeta_ce_*.toml; do
        srun --exclusive -n 1 --gres=gpu:1 --cpus-per-task=4 uv run bin/train_notmeta.py --config_file "$file" &
    done
elif [ "$LOSS_TYPE" = "focal" ]; then
    for file in data/configs/notmeta_focal_*.toml; do
        srun --exclusive -n 1 --gres=gpu:1 --cpus-per-task=4 uv run bin/train_notmeta.py --config_file "$file" &
    done
fi

wait
