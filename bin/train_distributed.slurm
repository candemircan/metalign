#!/bin/bash
#SBATCH --account=hai_1065
#SBATCH --nodes=1
#SBATCH --output=logs/train_%j.out
#SBATCH --time=10:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

config_file="$1"

if [[ -z "$config_file" ]]; then
    echo "Usage: sbatch bin/train_distributed.slurm <config_file>"
    exit 1
fi

module load CUDA/12

export OMP_NUM_THREADS=8
uv run torchrun --nproc_per_node=4 bin/train.py --config_file "$config_file"