#!/bin/bash

# Launch TouchDesigner with the correct conda environment and optional .toe file
# Usage: ./launch_touchdesigner.sh [path_to_project.toe]

# Check for optional .toe file parameter
TOE_FILE=""
if [ $# -gt 0 ]; then
    TOE_FILE="$1"
    # Expand ~ to home directory if needed
    TOE_FILE="${TOE_FILE/#\~/$HOME}"
    
    if [ ! -f "$TOE_FILE" ]; then
        echo "Error: .toe file not found: $TOE_FILE"
        exit 1
    fi
    echo "Will load project: $TOE_FILE"
fi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env

# Set Python path to include the project directory
export PYTHONPATH="$HOME/work/code/yolo-video-detection-example:$PYTHONPATH"

# Find TouchDesigner executable
TD_APP="/Applications/TouchDesigner.app"
TD_BIN="$TD_APP/Contents/MacOS/TouchDesigner"

if [ ! -f "$TD_BIN" ]; then
    echo "TouchDesigner not found at $TD_BIN"
    echo "Please update the path in this script"
    exit 1
fi

echo "Launching TouchDesigner with conda environment..."
echo "Python: $(which python)"
echo "Environment: $CONDA_DEFAULT_ENV"

# Launch TouchDesigner with or without project file
if [ -n "$TOE_FILE" ]; then
    echo "Loading project: $TOE_FILE"
    "$TD_BIN" "$TOE_FILE" &
else
    echo "Starting TouchDesigner without project file"
    "$TD_BIN" &
fi

echo "TouchDesigner launched with yolo_env conda environment"