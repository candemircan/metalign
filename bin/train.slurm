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
    echo "Error: You need to provide a config file as a positional argument. e.g. sbatch bin/train.slurm data/example_transformer_config.toml"
    exit 1
fi

uv run bin/train.py --config_file "$config_file"