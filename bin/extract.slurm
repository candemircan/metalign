#!/bin/bash
#SBATCH --account=hai_1065
#SBATCH --nodes=1
#SBATCH --output=logs/extract_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

module load CUDA/12

for dataset in things coco; do
    for model in timm/vit_base_patch14_reg4_dinov2.lvd142m timm/ViT-B-16-SigLIP2-512 timm/vit_base_patch16_rope_reg1_gap_256.sbb_in1k; do
        srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/extract_backbone.py $dataset $model &
    done
done

for layer in 6 11; do
    srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/extract_sae.py coco Prisma-Multimodal/sae-top_k-64-cls_only-layer_${layer}-hook_resid_post &
    srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run bin/extract_sae.py things Prisma-Multimodal/sae-top_k-64-cls_only-layer_${layer}-hook_resid_post &
done

wait