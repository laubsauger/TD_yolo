# YOLO Pose Model Information

## Overview
The YOLOv8 pose models (like yolov8n-pose.pt) are designed for human pose estimation, not general object detection.

## Key Differences from Object Detection:
1. **Primary output**: Human keypoints (17 points for COCO pose format)
2. **Classes**: Usually only detects "person" class (class 0)
3. **Additional data**: Each detection includes keypoint coordinates and confidence scores

## Keypoints (COCO format):
1. Nose
2. Left Eye
3. Right Eye
4. Left Ear
5. Right Ear
6. Left Shoulder
7. Right Shoulder
8. Left Elbow
9. Right Elbow
10. Left Wrist
11. Right Wrist
12. Left Hip
13. Right Hip
14. Left Knee
15. Right Knee
16. Left Ankle
17. Right Ankle

## Current Limitations:
- The current detection processing code is designed for object detection
- It doesn't handle or output keypoint data
- Only bounding boxes are processed and displayed

## To properly use pose models:
1. Need a separate pose processing handler
2. Modify the detection data format to include keypoints
3. Update TouchDesigner scripts to handle keypoint data

## Workaround:
The current code will still show bounding boxes around detected persons, but won't display keypoints.