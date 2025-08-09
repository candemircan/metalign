#!/bin/bash
#SBATCH --account=hai_1065
#SBATCH --nodes=1
#SBATCH --output=logs/train_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

if [ "$#" -ne 4 ]; then
    echo "Usage: sbatch bin/train.slurm <config1> <config2> <config3> <config4>"
    exit 1
fi

module load CUDA/12

srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run torchrun --nproc_per_node=1 bin/train.py --config_file $1 &
srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run torchrun --nproc_per_node=1 bin/train.py --config_file $2 &
srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run torchrun --nproc_per_node=1 bin/train.py --config_file $3 &
srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run torchrun --nproc_per_node=1 bin/train.py --config_file $4 &

wait