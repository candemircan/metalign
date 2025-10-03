#!/bin/bash
#SBATCH --account=hai_1065
#SBATCH --nodes=1
#SBATCH --output=logs/extract_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

module load CUDA/12

for data in levels coco things; do
    for model in timm/vit_base_patch16_224.mae timm/vit_base_patch16_dinov3.lvd1689m timm/vit_base_patch16_siglip_256.v2_webli; do
        srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/extract_backbone.py $data $model &
    done
done

for data in levels coco things; do
    for layer in 6 11; do
        srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/extract_sae.py $data Prisma-Multimodal/sae-top_k-64-cls_only-layer_${layer}-hook_resid_post &
    done
done

wait