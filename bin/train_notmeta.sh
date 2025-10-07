#!/bin/bash

for loss in ce focal; do
    sbatch bin/train_notmeta.slurm $loss
done