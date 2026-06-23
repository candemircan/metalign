#!/bin/bash
# Sync wandb offline runs to cloud every 2 minutes

echo "starting wandb sync daemon (every 2 minutes)"

while true; do
    uv run wandb sync --sync-all
    sleep 120
done
