# Pose Detection Implementation Status

## What's Implemented:

1. **Pose Processing Class** (`pose_processing.py`)
   - Automatically detects pose models by filename
   - Extracts keypoint data from raw model output
   - Writes pose data to shared memory buffer
   - Draws skeleton connections on video

2. **Detection Data Format**:
   - Buffer: `pose_data` (32KB)
   - Format per person: 56 float32 values
     - Bounding box: x1, y1, x2, y2
     - Score: confidence
     - 17 COCO keypoints × 3 values (x, y, confidence)

3. **TouchDesigner Scripts**:
   - `td_complete.py` - Creates all shared memory including pose buffer
   - `td_chop_pose.py` - Reads pose data and outputs as CHOP channels
   - Added flip vertical/horizontal parameters to handle coordinate systems

4. **Automatic Model Detection**:
   - If model filename contains "pose", uses pose processor
   - Otherwise uses standard object detection

## Current Limitations:

1. **Keypoint Extraction**: Currently using a workaround by storing raw model output and matching detections. This works but may not be 100% accurate for all cases.

2. **Coordinate System**: TouchDesigner often has inverted Y-axis. Use the "Flip Vertical" parameter in td_complete.py to correct this.

## Usage:

1. Launch TouchDesigner with environment:
   ```bash
   ./launch_touchdesigner.sh
   ```

2. Start YOLO server with pose model:
   ```bash
   ./start_yolo_connect.sh models/yolov8n-pose.pt
   ```

3. In TouchDesigner:
   - Add Script TOP with `td_complete.py` for video
   - Add Script CHOP with `td_chop_pose.py` for keypoints
   - Toggle "Flip Vertical" if bounding boxes appear mirrored

## Testing:

Run these to verify pose data:
```bash
python test_pose_load.py      # Test model loading
python test_pose_chop.py      # Test pose data reading
```

## Keypoint Indices:
0. Nose
1. Left Eye
2. Right Eye
3. Left Ear
4. Right Ear
5. Left Shoulder
6. Right Shoulder
7. Left Elbow
8. Right Elbow
9. Left Wrist
10. Right Wrist
11. Left Hip
12. Right Hip
13. Left Knee
14. Right Knee
15. Left Ankle
16. Right Ankle