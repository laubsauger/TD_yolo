#!/usr/bin/env python3
"""
Test script to verify CHOP can access shared memory created by setup script
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Simulate the CHOP setup function
from multiprocessing import shared_memory

def test_chop_memory_access():
    """Test what the CHOP script sees when trying to access shared memory"""
    print("🔍 Testing CHOP memory access...")
    
    # Try to connect to shared memory segments like the CHOP does
    segments_to_test = [
        ("yolo_states", "SharedMemory"),
        ("params", "ShareableList"), 
        ("image", "SharedMemory"),
        ("detection_data", "SharedMemory"),
        ("pose_data", "SharedMemory")
    ]
    
    found_segments = []
    missing_segments = []
    
    for name, memory_type in segments_to_test:
        try:
            if memory_type == "ShareableList":
                test_mem = shared_memory.ShareableList(name=name)
                print(f"   ✅ {name} (ShareableList) found")
                test_mem.shm.close()
                found_segments.append(name)
            else:
                test_mem = shared_memory.SharedMemory(name=name)
                print(f"   ✅ {name} (SharedMemory) found - size: {test_mem.size}")
                test_mem.close()
                found_segments.append(name)
        except FileNotFoundError:
            print(f"   ❌ {name} not found")
            missing_segments.append(name)
        except Exception as e:
            print(f"   ⚠️  {name} error: {e}")
            missing_segments.append(name)
    
    print(f"\n📊 Results:")
    print(f"   Found: {found_segments}")
    print(f"   Missing: {missing_segments}")
    
    if "pose_data" in found_segments:
        print("\n✅ SUCCESS: pose_data memory is accessible to CHOP")
        return True
    else:
        print("\n❌ FAILURE: pose_data memory not accessible to CHOP")
        return False

if __name__ == "__main__":
    test_chop_memory_access()