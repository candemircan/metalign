#!/bin/bash

experiments=$(jq -r 'keys[]' "data/checkpoints/best_models.json")
backbones=$(jq -r 'to_entries[0].value | keys[]' "data/checkpoints/best_models.json")

eval_types=("$@")
if [ ${#eval_types[@]} -eq 0 ]; then
    eval_types=("categorylearning" "rewardlearning" "levelso1o" "thingso1o")
fi

for eval_type in "${eval_types[@]}"; do
    for experiment in $experiments; do
        experiment=$(echo $experiment | tr -d "[]")
        for backbone in $backbones; do
            uv run bin/eval_"${eval_type}".py "$experiment" "$backbone"
        done
    done
done
