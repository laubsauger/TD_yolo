#!/usr/bin/env python3
"""Debug what the YOLO server sees"""
from multiprocessing import shared_memory
import numpy as np
import time

print("Monitoring what YOLO server receives...")

shm_states = shared_memory.SharedMemory(name="yolo_states")
sl_params = shared_memory.ShareableList(name="params")
shm_image = shared_memory.SharedMemory(name="image")

# Get dimensions from shared params
width = int(sl_params[4]) if sl_params[4] else 1280
height = int(sl_params[5]) if sl_params[5] else 720
print(f"Using dimensions: {width}x{height}")

shared_array = np.ndarray((height, width, 3), dtype=np.float32, buffer=shm_image.buf)

last_state = None

while True:
    try:
        server_state = shm_states.buf[0]
        
        # When server receives new frame
        if server_state == 49 and last_state != 49:  # ASCII '1'
            width = int(sl_params[4])
            height = int(sl_params[5])
            
            # Check the image data
            image_slice = shared_array[:height, :width]
            print(f"\nServer received: {width}x{height}")
            print(f"  Data range: [{image_slice.min():.3f}, {image_slice.max():.3f}]")
            print(f"  Mean: {image_slice.mean():.3f}")
            print(f"  First pixel: {image_slice[0,0]}")
            
            # Check if it's all black
            if image_slice.max() < 0.01:
                print("  WARNING: Image is all black!")
        
        last_state = server_state
        time.sleep(0.1)
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")
        break

shm_states.close()
sl_params.shm.close()
shm_image.close()