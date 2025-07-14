#!/usr/bin/env python3
"""
Test script to verify pose data in shared memory
Run this while YOLO server is processing with a pose model
"""
import struct
import time
from multiprocessing import shared_memory

# COCO Pose keypoint names
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear", 
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

def main():
    print("Connecting to pose data memory...")

    try:
        # Connect to pose data
        shm_pose = shared_memory.SharedMemory(name="pose_data", create=False)
        print(f"-> Connected to pose memory (size: {shm_pose.size})")

        print("\nReading pose data (press Ctrl+C to stop)...")
        while True:
            # Read number of persons
            num_persons = struct.unpack('i', shm_pose.buf[0:4])[0]

            if num_persons > 0 and num_persons < 10:  # Sanity check
                print(f"\n--- {num_persons} person(s) detected ---")

                # Read each person
                buffer_pos = 4
                for i in range(num_persons):
                    if buffer_pos + 224 > shm_pose.size:  # 56 floats = 224 bytes
                        break

                    # Unpack person data: bbox(4) + score(1) + keypoints(17*3) = 56 floats
                    person_data = struct.unpack('56f', shm_pose.buf[buffer_pos:buffer_pos+224])

                    x1, y1, x2, y2 = person_data[0:4]
                    score = person_data[4]

                    print(f"\n  Person {i+1}:")
                    print(f"    BBox: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                    print(f"    Score: {score:.3f}")
                    print(f"    Keypoints:")

                    # Extract keypoints (17 keypoints, 3 values each)
                    keypoints = person_data[5:]  # Skip bbox and score
                    for j, kpt_name in enumerate(KEYPOINT_NAMES):
                        x = keypoints[j*3]
                        y = keypoints[j*3 + 1]
                        conf = keypoints[j*3 + 2]
                        if conf > 0.5:  # Only show confident keypoints
                            print(f"      {kpt_name}: ({x:.1f}, {y:.1f}) conf={conf:.2f}")

                    buffer_pos += 224

            time.sleep(0.5)  # Check twice per second

    except FileNotFoundError:
        print("⚠️  Pose memory not found. Make sure:")
        print("   1. TouchDesigner is running with td_complete.py")
        print("   2. YOLO server is running with a pose model")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if 'shm_pose' in locals():
            shm_pose.close()

if __name__ == "__main__":
    main()
