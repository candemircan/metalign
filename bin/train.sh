#!/bin/bash

for f in data/configs/*toml; do
    sbatch bin/train.slurm "$f"
done