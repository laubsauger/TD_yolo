#!/usr/bin/env python3
"""
Test script to verify detection data in shared memory
Run this while YOLO server is processing to see detection data
"""
import struct
import time
from multiprocessing import shared_memory

def main():
    print("Connecting to detection data memory...")

    try:
        # Connect to detection data
        shm_detection = shared_memory.SharedMemory(name="detection_data", create=False)
        print(f"-> Connected to detection memory (size: {shm_detection.size})")

        print("\nReading detection data (press Ctrl+C to stop)...")
        while True:
            # Read number of detections
            num_detections = struct.unpack('i', shm_detection.buf[0:4])[0]

            if num_detections > 0 and num_detections < 100:  # Sanity check
                print(f"\n--- {num_detections} detections ---")

                # Read each detection
                buffer_pos = 4
                for i in range(num_detections):
                    if buffer_pos + 24 > shm_detection.size:
                        break

                    # Unpack detection data
                    x1, y1, x2, y2, score, class_id = struct.unpack('6f', 
                        shm_detection.buf[buffer_pos:buffer_pos+24])

                    print(f"  Detection {i+1}:")
                    print(f"    BBox: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                    print(f"    Score: {score:.3f}")
                    print(f"    Class ID: {int(class_id)}")

                    buffer_pos += 24

            time.sleep(0.5)  # Check twice per second

    except FileNotFoundError:
        print("⚠️  Detection memory not found. Make sure YOLO server is running.")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if 'shm_detection' in locals():
            shm_detection.close()

if __name__ == "__main__":
    main()
