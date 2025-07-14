#!/bin/bash
# Start YOLO server - connects to TouchDesigner's shared memory

echo "Starting YOLO server (TouchDesigner memory mode)..."

# Check if model path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_yolo_model.pt>"
    echo "Example: $0 models/yolov8n.pt"
    exit 1
fi

MODEL_PATH=$1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env

# Connect to existing shared memory created by setup script
echo "Connecting to shared memory created by setup script..."

# Check if it's a pose model
if [[ $MODEL_PATH == *"pose"* ]]; then
    echo "Detected pose model - using 640x640 resolution"
    WIDTH=640
    HEIGHT=640
else
    echo "Using standard 1280x720 resolution"
    WIDTH=1280
    HEIGHT=720
fi

# Start server
echo -e "\nStarting YOLO detection server with model: $MODEL_PATH"
python processing.py \
    -p "$MODEL_PATH" \
    --shared_update_mem_name "yolo_states" \
    --shared_params_mem_name "params" \
    --shared_array_mem_name "image" \
    -iw $WIDTH \
    -ih $HEIGHT \
    -c 3 \
    --image_type float32