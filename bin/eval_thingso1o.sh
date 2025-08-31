#!/bin/bash

for experiment in main raw midsae; do
    for backbone in mae vit clip siglip2 dinov2; do
        uv run bin/eval_thingso1o.py $experiment $backbone
    done
done