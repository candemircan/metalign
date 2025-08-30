#!/bin/bash

for experiment in main raw midsae; do
    for backbone in vit clip siglip2 dinov2; do
        uv run bin/eval_rewardlearning.py $experiment $backbone
    done
done