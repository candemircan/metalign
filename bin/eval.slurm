#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000
#SBATCH --job-name=eval_%x
#SBATCH --output=logs/eval_%x_%j.out

EVAL_TYPE=$1
EXPERIMENT=$2
BACKBONE=$3
PARTICIPANT=$4

# Run the evaluation
if [ "$EVAL_TYPE" = "brain" ]; then
    uv run bin/eval_"${EVAL_TYPE}".py "$EXPERIMENT" "$BACKBONE" "$PARTICIPANT" --n_jobs "$SLURM_CPUS_PER_TASK"
elif [ "$EVAL_TYPE" = "levelso1o" ] || [ "$EVAL_TYPE" = "thingso1o" ]; then
    DATASET=$(echo "$EVAL_TYPE" | sed 's/o1o$//')
    uv run bin/eval_o1o.py "$DATASET" "$EXPERIMENT" "$BACKBONE"
else
    uv run bin/eval_"${EVAL_TYPE}".py "$EXPERIMENT" "$BACKBONE"
fi
