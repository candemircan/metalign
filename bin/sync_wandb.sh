#!/bin/bash

# no internet in the compute nodes, so gotta sync from the login node
# run it in a tmux session
RUN_DURATION_SECONDS=$((24 * 3600)) # 12 hours
SYNC_INTERVAL_SECONDS=60 # Sync every
SHARED_WANDB_BASE_DIR="wandb"

START_TIME=$(date +%s)
CURRENT_TIME=$(date +%s)

while [ $((CURRENT_TIME - START_TIME)) -lt "$RUN_DURATION_SECONDS" ]; do

    uv run wandb sync "$SHARED_WANDB_BASE_DIR" --include-synced --include-offline --sync-all
    sleep "$SYNC_INTERVAL_SECONDS"
    CURRENT_TIME=$(date +%s)
done

uv run wandb sync "$SHARED_WANDB_BASE_DIR"