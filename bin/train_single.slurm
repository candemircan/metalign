#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000
#SBATCH --output=logs/train_%j.out

config_file="$1"

# Check if the arguments are provided
if [[ -z "$config_file" ]]; then
    echo "Error: You need to provide a config file as a positional argument. e.g. sbatch bin/train_single.slurm <config1>"
    exit 1
fi

export OMP_NUM_THREADS=8
uv run torchrun --nproc_per_node=1 bin/train.py --config_file "$config_file"