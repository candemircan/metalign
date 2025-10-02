#!/bin/bash

for experiment in main raw midsae; do
    for backbone in mae clip siglip2 dinov3; do
        uv run bin/eval_rewardlearning.py $experiment $backbone
    done
done