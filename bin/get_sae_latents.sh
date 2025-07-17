#!/bin/bash

for dataset in things coco; do
    uv run bin/get_sae_latents.py $dataset
done