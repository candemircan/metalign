#!/bin/bash

for f in data/configs/*toml; do
    if [[ "$f" == *"notmeta"* ]]; then
        continue
    fi
    sbatch bin/train.slurm "$f"
done