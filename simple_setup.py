#!/usr/bin/env python3
"""
Simple setup that just starts the YOLO server
TouchDesigner will handle shared memory creation
"""
import subprocess
import time
import os

def kill_yolo_processes():
    """Kill YOLO processes"""
    patterns = ["processing.py", "debug_server.py", "start_yolo_connect.sh"]
    
    for pattern in patterns:
        try:
            result = subprocess.run(f"pgrep -f '{pattern}'", shell=True, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    print(f"Killing {pattern} process (PID: {pid})")
                    subprocess.run(f"kill {pid}", shell=True)
                    time.sleep(0.1)
        except Exception as e:
            print(f"Note: Could not kill {pattern}: {e}")

def main():
    print("🎬 Simple YOLO Setup")
    print("=" * 30)
    
    # Kill existing processes
    print("\n📋 Cleaning up YOLO processes...")
    kill_yolo_processes()
    time.sleep(2)
    
    # Find model
    model_files = ["models/yolov8n-pose.pt", "models/yolo11n-pose.pt"]
    model_to_use = None
    for model in model_files:
        if os.path.exists(model):
            model_to_use = model
            break
    
    if not model_to_use:
        print("❌ No pose model found!")
        print("Please download: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n-pose.pt")
        return
    
    # Start YOLO server
    print(f"\n📋 Starting YOLO server with {model_to_use}...")
    subprocess.Popen(f"./start_yolo_connect.sh {model_to_use}", shell=True)
    time.sleep(3)
    
    # Start debug server
    print("📋 Starting debug server...")
    subprocess.Popen("python debug_server.py", shell=True)
    time.sleep(1)
    
    print("\n✨ Setup complete!")
    print("🎯 Now in TouchDesigner:")
    print("   1. Click Setup on Script TOP")
    print("   2. Click Setup on Script CHOP")
    print("   3. Done!")
    
    print("\n🛑 Press Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        kill_yolo_processes()

if __name__ == "__main__":
    main()