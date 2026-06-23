#!/bin/bash
#SBATCH --account=hai_1149
#SBATCH --nodes=1
#SBATCH --output=logs/train_%j.out
#SBATCH --time=05:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

config_file="$1"

if [[ -z "$config_file" ]]; then
    echo "Usage: sbatch bin/train.slurm <config_file>"
    exit 1
fi

module load CUDA-Python/13
export OMP_NUM_THREADS=8
unset PYTHONPATH

uv run torchrun --nproc_per_node=4 bin/train.py --config_file "$config_file"
