#!/bin/bash

# no internet in the compute nodes, so gotta sync from the login node
RUN_DURATION_SECONDS=$((12 * 3600)) # 12 hours
SYNC_INTERVAL_SECONDS=60 # Sync every
SHARED_WANDB_BASE_DIR="wandb"

START_TIME=$(date +%s)
CURRENT_TIME=$(date +%s)

while [ $((CURRENT_TIME - START_TIME)) -lt "$RUN_DURATION_SECONDS" ]; do

    echo "Attempting to sync all W&B runs from: $SHARED_WANDB_BASE_DIR"
    wandb sync "$SHARED_WANDB_BASE_DIR"
    sleep "$SYNC_INTERVAL_SECONDS"
    CURRENT_TIME=$(date +%s)
done

wandb sync "$SHARED_WANDB_BASE_DIR"