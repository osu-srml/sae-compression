#!/bin/bash
set -e

# Get the device argument
DEVICE="$1"
PYTHON_SCRIPT="train.py"

# Iterate over layers and run only those assigned to the given device
for i in {0..11}; do
    if [ $((i % 4)) -eq "$DEVICE" ]; then
        echo "Running: python3 $PYTHON_SCRIPT --layer $i --device $DEVICE"
        python "$PYTHON_SCRIPT" --layer "$i" --device "$DEVICE"
    fi
done