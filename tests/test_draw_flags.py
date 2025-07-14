#!/usr/bin/env python3
"""Test script to check DrawInfo flags"""

# Test the DrawInfo flags to understand the values
print("DrawInfo flag values:")
print(f"  DRAW_TEXT = 1")
print(f"  DRAW_BBOX = 2") 
print(f"  DRAW_CONF = 4")
print(f"  DRAW_SKELETON = 8")
print(f"  OVERLAY_ONLY = 16")
print(f"  TRANSPARENT_BG = 32")

print("\nCommon combinations:")
print(f"  All drawing on (default): 1+2+4+8 = {1+2+4+8}")
print(f"  Just bbox and skeleton: 2+8 = {2+8}")
print(f"  Everything with overlay: 1+2+4+8+16 = {1+2+4+8+16}")

# Test what value would enable skeleton
test_value = 15  # This should be the default
print(f"\nTesting value {test_value}:")
print(f"  Text enabled: {bool(test_value & 1)}")
print(f"  BBox enabled: {bool(test_value & 2)}")
print(f"  Conf enabled: {bool(test_value & 4)}")
print(f"  Skeleton enabled: {bool(test_value & 8)}")
print(f"  Overlay enabled: {bool(test_value & 16)}")
print(f"  Transparent enabled: {bool(test_value & 32)}")