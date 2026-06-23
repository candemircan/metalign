#!/bin/bash

experiments=$(jq -r 'keys[]' "data/checkpoints/best_models.json")
backbones=$(jq -r 'to_entries[0].value | keys[]' "data/checkpoints/best_models.json")

eval_types=("$@")
if [ ${#eval_types[@]} -eq 0 ]; then
    eval_types=("levelso1o" "thingso1o" "rewardlearning" "categorylearning" "brain" "icl")
fi

for eval_type in "${eval_types[@]}"; do
    if [ "$eval_type" = "brain" ]; then
        # Brain eval only uses these 4 experiments + participants 1-3
        brain_exps=("main" "raw" "midsae" "static_linear")
        for experiment in "${brain_exps[@]}"; do
            for backbone in $backbones; do
                for par in 1 2 3; do
                    job_name="brain_${experiment}_${backbone}_p${par}"
                    sbatch --job-name="$job_name" bin/eval.slurm "$eval_type" "$experiment" "$backbone" "$par"
                done
            done
        done
        continue
    fi
    if [ "$eval_type" = "icl" ]; then
        # ICL eval only uses main model for each backbone
        for backbone in $backbones; do
            job_name="icl_main_${backbone}"
            sbatch --job-name="$job_name" bin/eval.slurm "$eval_type" "main" "$backbone"
        done
        continue
    fi
    for experiment in $experiments; do
        experiment=$(echo "$experiment" | tr -d "[]")
        for backbone in $backbones; do
            job_name="${eval_type}_${experiment}_${backbone}"
            sbatch --job-name="$job_name" bin/eval.slurm "$eval_type" "$experiment" "$backbone"
        done
    done
done
