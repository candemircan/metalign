#!/bin/bash

source .venv/bin/activate

uv pip install -e .

for dataset in things coco; do
    python bin/get_sae_latents.py $dataset
done