#!/bin/bash

for f in data/configs/*toml; do
    config_name=$(basename "$f" .toml)
    sbatch --output="logs/${config_name}.out" bin/train.slurm "$f"
done