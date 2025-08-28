#!/bin/bash
#SBATCH --account=hai_1065
#SBATCH --nodes=1
#SBATCH --output=logs/extract_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

module load CUDA/12

for model in facebook/dinov3-vitb16-pretrain-lvd1689m google/vit-base-patch16-224 openai/clip-vit-base-patch16 google/siglip2-base-patch16-224; do
    srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/extract_backbone.py coco --repo_id $model &
    srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/extract_backbone.py things --repo_id $model &
done

for layer in 6 11; do
    srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/extract_sae.py coco --repo_id Prisma-Multimodal/sae-top_k-64-cls_only-layer_${layer}-hook_resid_post &
    srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/extract_sae.py things --repo_id Prisma-Multimodal/sae-top_k-64-cls_only-layer_${layer}-hook_resid_post &
done

wait