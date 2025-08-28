#!/bin/bash

# no internet in the compute nodes, so gotta sync from the login node
# run it in a tmux session
RUN_DURATION_SECONDS=$((12 * 3600)) # 12 hours
SYNC_INTERVAL_SECONDS=60 # Sync every
SHARED_WANDB_BASE_DIR="wandb"
DELETED_RUNS_FILE="$SHARED_WANDB_BASE_DIR/.deleted_runs"

# function to sync runs while skipping deleted ones
# some jobs just crash and leave behind offline runs that were deleted on wandb
# this messes up the sync process a bit, so we keep track of deleted runs in a file
sync_runs() {
    local wandb_dir="$1"
    
    # create deleted runs file if it doesn't exist
    touch "$DELETED_RUNS_FILE"
    
    for run_dir in "$wandb_dir"/offline-run-*; do
        if [ -d "$run_dir" ]; then
            # extract run ID from directory name (last part after the last dash)
            run_id=$(basename "$run_dir" | sed 's/.*-//')
            
            # skip if this run ID is in our deleted runs list
            if grep -q "^$run_id$" "$DELETED_RUNS_FILE" 2>/dev/null; then
                echo "Skipping deleted run: $run_id"
                continue
            fi
            
            echo "Syncing run: $run_id"
            sync_output=$(uv run wandb sync "$run_dir" 2>&1)
            
            # check if the run was deleted (409 error)
            if echo "$sync_output" | grep -q "was previously created and deleted"; then
                echo "Run $run_id was deleted, adding to skip list"
                echo "$run_id" >> "$DELETED_RUNS_FILE"
            elif echo "$sync_output" | grep -q "ERROR"; then
                echo "Error syncing run $run_id:"
                echo "$sync_output"
            else
                echo "Successfully synced run $run_id"
            fi
        fi
    done
}

START_TIME=$(date +%s)
CURRENT_TIME=$(date +%s)

while [ $((CURRENT_TIME - START_TIME)) -lt "$RUN_DURATION_SECONDS" ]; do
    sync_runs "$SHARED_WANDB_BASE_DIR"
    sleep "$SYNC_INTERVAL_SECONDS"
    CURRENT_TIME=$(date +%s)
done

# final sync
sync_runs "$SHARED_WANDB_BASE_DIR"