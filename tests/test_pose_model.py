#!/usr/bin/env python3
"""
Test what the pose model outputs
"""
from ultralytics import YOLO
import numpy as np

# Load pose model
model = YOLO('models/yolov8n-pose.pt')

# Create dummy image
img = np.zeros((640, 640, 3), dtype=np.uint8)

# Run inference
results = model(img)

if results:
    result = results[0]
    print("Model task:", model.task)
    print("\nResult attributes:")
    for attr in dir(result):
        if not attr.startswith('_'):
            print(f"  {attr}")
    
    if hasattr(result, 'keypoints'):
        print("\nKeypoints detected!")
        print("Keypoints shape:", result.keypoints.data.shape if result.keypoints.data is not None else "None")
    
    if hasattr(result, 'boxes'):
        print("\nBoxes:", result.boxes)
        if result.boxes is not None:
            print("Boxes shape:", result.boxes.data.shape)
            if hasattr(result.boxes, 'cls'):
                print("Classes:", result.boxes.cls)