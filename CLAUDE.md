# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a TouchDesigner YOLO integration project that enables real-time object detection and pose estimation. It uses shared memory for high-performance communication between TouchDesigner and Python YOLO processing.

## Performance Philosophy

This codebase prioritizes **maximum real-time performance** for interactive installations:

### Core Design Principles

1. **Zero-Copy Architecture**: Shared memory between TouchDesigner and Python
2. **Direct Model Access**: Raw tensor outputs for maximum flexibility
3. **Custom Optimizations**: Cython NMS, in-place operations, minimal allocations
4. **TouchDesigner First**: All design decisions favor TD integration

### Performance Targets

- 60+ FPS for detection on modern GPUs
- 30+ FPS for pose estimation
- < 16ms latency end-to-end
- Support for multiple simultaneous models

### Key Features for Installations

- **Object Detection**: Bounding boxes with class labels
- **Pose Estimation**: 17-point COCO skeleton tracking
- **OpenPose Export**: ControlNet-ready JSON output (planned)
- **Multi-person Tracking**: Handles crowded scenes
- **Shared Memory IPC**: Zero-copy frame transfer

### When NOT to Use Ultralytics Directly

- We need raw model outputs (pose keypoints)
- TouchDesigner requires custom coordinate systems
- Shared memory integration is critical
- Custom NMS for TD compatibility

### Future Enhancements

- [ ] OpenPose JSON export for ControlNet
- [ ] ByteTrack integration for persistent IDs
- [ ] TensorRT optimization path
- [ ] Multi-camera synchronization

## Essential Commands

**IMPORTANT: Always use the conda environment `yolo_env` for this project!**

```bash
# Check if yolo_env exists, create if needed
conda env list | grep yolo_env || conda create -n yolo_env python=3.9 -y

# ALWAYS activate the environment before any Python operations
conda activate yolo_env
```

### Development Setup

```bash
# Install dependencies (choose one) - AFTER activating yolo_env
conda activate yolo_env
pip install -r requirements.gpu.txt  # For GPU
pip install -r requirements.cpu.txt  # For CPU
pip install -r requirements.dev.txt  # For development

# Build Cython extensions
pip install -e .  # Development mode
python -m cibuildwheel --platform <your_platform>  # Binary distribution
```

### Running the System

```bash
# Start YOLO server for TouchDesigner (scripts already handle conda activation)
./start_yolo_connect.sh models/yolov8n.pt  # Object detection
./start_yolo_connect.sh models/yolov8n-pose.pt  # Pose estimation

# Or manually with setup_all.py (activate conda first!)
conda activate yolo_env
python setup_all.py -m models/yolo11n-pose.pt -w 640 --height 640

# Launch TouchDesigner with environment
./launch_touchdesigner.sh [project.toe]

# Standalone video processing
conda activate yolo_env
python main.py -c models/yolov8n.pt -i input.mp4 -o output.mp4
```

### Testing and Quality

```bash
# ALWAYS activate conda environment first
conda activate yolo_env

# Run tests
pytest

# Linting
pylint yolo_models

# Auto-format code
autopep8 --in-place --recursive yolo_models

# Typecheck (when implemented)
npm run typecheck  # If available
ruff check .  # If available
```

## Architecture Overview

### Shared Memory Communication

The project uses multiprocessing shared memory for zero-copy data transfer:

- **yolo_states**: 3-byte synchronization flags
- **params**: Configuration parameters (thresholds, dimensions, draw flags)
- **image**: Frame buffer (float32, default 1280x720x3)
- **detection_data**: 16KB buffer for bounding boxes
- **pose_data**: 32KB buffer for keypoints (17 COCO keypoints per person)

### Processing Pipeline

1. TouchDesigner writes frame to shared memory
2. Python server detects new frame via state flags
3. YOLO model processes frame (detection or pose)
4. Results written back to shared memory
5. TouchDesigner reads processed frame and data

### Key Components

- **processing.py**: Main YOLO server that monitors shared memory
- **setup_all.py**: Automated setup and launcher
- **yolo_models/detection/detector.py**: PyTorch YOLO wrapper
- **yolo_models/processing/detection_processing.py**: Object detection processor
- **yolo_models/processing/pose_processing.py**: Pose estimation processor
- **td_top_resolution_config.py**: TouchDesigner Script TOP that writes required config file (could be checked and maybe refactored/handled differently)
- **td_top_yolo.py**: TouchDesigner Script TOP for video I/O, actual input to yolo processor on the backend, required for all others
- **td_chop_pose.py**: TouchDesigner Script CHOP for pose data output
- **td_chop_detection.py**: TouchDesigner Script CHOP for object detection data output (bounding boxes, labels, etc)

### Model Support

- **Detection models**: Standard YOLO (yolov8n.pt, yolov8x.pt, yolo11n.pt)
- **Pose models**: YOLO-Pose (yolov8n-pose.pt, yolo11n-pose.pt)
- Models auto-detected by "pose" in filename
- Supports PyTorch on CPU, CUDA, and Apple MPS

### Critical Implementation Details

- Pose models automatically use 640x640 resolution
- Detection models default to 1280x720
- Coordinate system: TouchDesigner uses vertical flip
- Custom Cython NMS implementation for TouchDesigner compatibility
- Keypoint smoothing (EMA) for stable pose tracking
- Frame duplicate detection to handle dropped frames

## Current Work Items

See `Yolo11_upgrade.md` for the ongoing YOLO11 migration:

- Upgrading ultralytics 8.0.114 → 8.3.165
- Adding yolo11n-pose.pt support
- Updating for API breaking changes

## Model Download Locations

Models should be placed in the `models/` directory. Download from:

- <https://github.com/ultralytics/ultralytics>
- YOLO11 models: yolo11n.pt, yolo11n-pose.pt, etc.

## GENERAL INSTRUCTIONS

- Avoid over zealously creating new files when the changes could just as well be applied to the existing file that we're currently working on.
- Make smart decisions as to when a change is much better suited as an edit instead of total rewrite under new file name from the ground up (where you'll usually forget half of the prior implementation while focusing on the new changes. I know applying / editing is hard but please don't be that guy who writes td_node_v4_final_fixed_thistimeforreal.tsx filenames)
