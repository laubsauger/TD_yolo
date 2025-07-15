# CLAUDE.md

AI assistant instructions for the TD_yolo project.

## Project Overview

TouchDesigner YOLO integration for real-time object detection and pose estimation using shared memory for zero-copy performance.

## Key Features

- Object detection (80+ classes) and pose estimation (17-point skeleton)
- Shared memory IPC between TouchDesigner and Python
- Support for YOLO11 models
- Cross-platform (Windows, macOS, Linux)

## Environment Setup

**Use Python 3.11** (3.13 has compatibility issues with dependencies)

```bash
# Setup virtual environment
./setup.sh         # macOS/Linux
setup.bat          # Windows
python3.11 setup_env.py  # Direct

# Activate
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

## Essential Commands

```bash
# All-in-one (recommended)
python setup_all.py -m models/yolo11n-pose.pt

# Manual steps
python start_yolo_server.py models/yolo11n-pose.pt
python launch_touchdesigner.py YoloDetection.toe

# Custom resolution
python setup_all.py -m models/yolo11n-pose.pt -w 1920 -h 1080

# Stop all
python setup_all.py --stop
```

## Architecture

### Shared Memory Layout

- `yolo_states`: 3-byte sync flags
- `params`: ShareableList with config
- `image`: Frame buffer (float32, 1280x720x3 default)
- `detection_data`: 16KB for bounding boxes  
- `pose_data`: 32KB for keypoints

### Processing Flow

1. TouchDesigner writes frame to shared memory
2. Python monitors `yolo_states` for new frames
3. YOLO processes frame (detection or pose)
4. Results written back to shared memory
5. TouchDesigner reads and visualizes

### Key Files

- `setup_all.py` - Creates shared memory, launches everything
- `processing.py` - Main YOLO server
- `yolo_models/processing/base_processing.py` - Shared memory handler
- `nodes_TD/td_top_yolo.py` - TouchDesigner video I/O
- `nodes_TD/td_chop_pose.py` - Pose data output

## Development Guidelines

### Code Style

- No comments unless requested
- Follow existing patterns
- Use type hints
- Keep functions focused

### Performance

- Minimize allocations in hot paths
- Use in-place operations
- Batch numpy operations
- Profile before optimizing

### Testing

```bash
pytest tests/
pylint yolo_models
```

## Common Issues

### NumPy Version

- Must use NumPy < 2.0 (OpenCV compatibility)
- Fixed in requirements.base.txt

### Python 3.13

- PyAV has Cython compatibility issues
- Use Python 3.11 or 3.12

### Shared Memory

- Server requires existing shared memory
- Use `setup_all.py` to create it first

## Model Support

**Detection**: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
**Pose**: yolo11n-pose.pt, yolo11s-pose.pt, etc.

Pose models auto-switch to 640x640 resolution.

## TouchDesigner Integration

Scripts in `nodes_TD/`:

- `td_top_yolo.py` - Main video I/O
- `td_chop_detection.py` - Detection data
- `td_chop_pose.py` - Pose keypoints
- `td_chop_fps_stats.py` - Performance metrics
- `td_chop_openpose.py` - Open Pose keypoints
- `td_dat_openpose.py` - Open Pose DAT Renderer

## Future Work

- [ ] ByteTrack for persistent IDs
- [ ] TensorRT optimization
- [ ] Multi-camera support
