#!/bin/bash

# Directory to clean
TARGET_DIR="/home/gupte.31/COLM/sae-compression/gemma2b/wandb"
# Sleep duration between runs (in seconds)
SLEEP_INTERVAL=900  # every hour

while true; do
    echo "[$(date)] Attempting to clean $TARGET_DIR"

    # Check if the directory exists
    if [ ! -d "$TARGET_DIR" ]; then
        echo "Directory does not exist: $TARGET_DIR"
        sleep "$SLEEP_INTERVAL"
        continue
    fi

    # Try to list open files in the directory (if any process is using it)
    if lsof +D "$TARGET_DIR" >/dev/null 2>&1; then
        echo "Directory is in use. Skipping cleanup."
    else
        # Attempt to remove contents inside the directory
        if rm -rf "$TARGET_DIR"/* "$TARGET_DIR"/.[!.]* "$TARGET_DIR"/..?* 2>/tmp/cleanup_error.log; then
            echo "Successfully cleaned directory: $TARGET_DIR"
        else
            if grep -q "Device or resource busy" /tmp/cleanup_error.log; then
                echo "Cleanup failed: Directory in use (resource busy)"
            else
                echo "Cleanup failed with other errors:"
                cat /tmp/cleanup_error.log
            fi
        fi
    fi

    echo "Sleeping for $SLEEP_INTERVAL seconds..."
    sleep "$SLEEP_INTERVAL"
done