#!/usr/bin/env python3
"""
Simple test to check pose model loading
"""
import torch
from yolo_models.detection import PyTorchYoloDetector
import numpy as np

# Test loading pose model
print("Loading pose model...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

detector = PyTorchYoloDetector("models/yolov8n-pose.pt", device)

# Create dummy image
img = np.zeros((640, 640, 3), dtype=np.uint8)

# Run prediction
print("Running prediction...")
result = detector.predict(img)

print(f"Detections: {len(result.scores)}")
if hasattr(detector, '_last_raw_output') and detector._last_raw_output is not None:
    print(f"Raw output shape: {detector._last_raw_output.shape}")
    if len(detector._last_raw_output.shape) == 3:
        print(f"Output dimensions: {detector._last_raw_output.shape[-1]} (expecting 56 for pose)")
else:
    print("No raw output saved")

print("Done!")