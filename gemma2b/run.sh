#!/bin/bash
set -e

DEVICE_ID="$1"
PYTHON_SCRIPT="train.py"

# Devices to use
DEVICES=(1 3 5)

# List of layers to run
LAYERS=(0 1 2 3 5 6 7 9 11 13 14 15 17 18 19 21 22 23 25)

# Find this DEVICE_ID's index in DEVICES array
DEVICE_INDEX=-1
for idx in "${!DEVICES[@]}"; do
    if [ "${DEVICES[$idx]}" -eq "$DEVICE_ID" ]; then
        DEVICE_INDEX="$idx"
        break
    fi
done

# If DEVICE_ID is not in the list, exit
if [ "$DEVICE_INDEX" -eq -1 ]; then
    echo "Error: Device ID $DEVICE_ID is not in the list of allowed devices: ${DEVICES[*]}"
    exit 1
fi

# Loop through layers and only run those assigned to the current DEVICE_ID
for i in "${!LAYERS[@]}"; do
    LAYER=${LAYERS[$i]}
    if [ $((i % ${#DEVICES[@]})) -eq "$DEVICE_INDEX" ]; then
        echo "Running layer $LAYER on device $DEVICE_ID"
        python "$PYTHON_SCRIPT" --layer "$LAYER" --device "$DEVICE_ID"
    fi
done

wait
echo "Device $DEVICE_ID finished all assigned layers."