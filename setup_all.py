#!/usr/bin/env python3
"""
Automated setup script for YOLO TouchDesigner integration
This script automates the tedious manual setup process
"""
import subprocess
import time
import os
import argparse
from multiprocessing import shared_memory


def kill_specific_processes():
    """Kill only specific YOLO-related processes"""
    killed = []

    # Look for specific process patterns
    patterns = ["processing.py", "debug_server.py", "start_yolo_connect.sh"]

    for pattern in patterns:
        try:
            # Use pgrep to find processes, then pkill to kill them
            result = subprocess.run(
                f"pgrep -f '{pattern}'", shell=True, capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    print(f"Killing {pattern} process (PID: {pid})")
                    subprocess.run(f"kill {pid}", shell=True)
                    killed.append(pattern)
                    time.sleep(0.1)
        except Exception as e:
            print(f"   Note: Could not kill {pattern}: {e}")

    return killed


def create_shared_memory(width=1280, height=720, fps_limit=60):
    """Create shared memory segments that TouchDesigner normally creates"""
    # Cap FPS limit at 60
    if fps_limit > 60:
        print(f"   ⚠️  FPS limit capped at 60 (requested: {fps_limit})")
        fps_limit = 60
    
    print("\n📋 Creating shared memory segments...")
    print(f"   📐 Using resolution: {width}x{height}")
    print(f"   🎯 FPS limit: {fps_limit}")

    # Clean up any existing shared memory first
    segments = [
        ("yolo_states", 16384),  # macOS minimum size - we'll use first 3 bytes for states
        ("params", None),  # ShareableList - handle separately
        ("image", width * height * 3 * 4),  # float32 image buffer
        ("detection_data", 16384),
        ("pose_data", 32768),
        ("fps_stats", 16384),  # macOS minimum size - we'll use first 32 bytes for 4 floats
    ]

    created_segments = []

    for name, size in segments:
        try:
            if name == "params":
                # Clean up existing ShareableList first
                try:
                    sl_existing = shared_memory.ShareableList(name=name)
                    sl_existing.shm.close()
                    sl_existing.shm.unlink()
                    print(f"   🧹 Cleaned existing ShareableList: {name}")
                except:
                    pass
                
                # Create ShareableList for parameters
                params = [
                    0.5,  # IOU_THRESH
                    0.5,  # SCORE_THRESH
                    0,  # TOP_K
                    1.0,  # ETA
                    width,  # IMAGE_WIDTH
                    height,  # IMAGE_HEIGHT
                    3,  # IMAGE_CHANNELS
                    "image",  # SHARED_ARRAY_MEM_NAME
                    "yolo_states",  # SHARD_STATE_MEM_NAME
                    "float32",  # IMAGE_DTYPE
                    15,  # DRAW_INFO (all flags enabled: 1+2+4+8)
                    0.1,  # POSE_THRESHOLD (lowered for better detection)
                    fps_limit,  # FPS_LIMIT (0 = unlimited)
                ]

                sl = shared_memory.ShareableList(sequence=params, name=name)
                print(f"   ✅ Created ShareableList: {name}")
                created_segments.append((name, sl, "list"))
            else:
                # Create regular shared memory
                try:
                    # Try to clean up existing first
                    shm = shared_memory.SharedMemory(name=name)
                    shm.close()
                    shm.unlink()
                except:
                    pass

                shm = shared_memory.SharedMemory(create=True, size=size, name=name)

                # Initialize states buffer
                if name == "yolo_states":
                    shm.buf[0] = 48  # '0' - server state
                    shm.buf[1] = 48  # '0' - client state
                    shm.buf[2] = 48  # '0' - server alive

                print(f"   ✅ Created shared memory: {name} (size: {shm.size})")
                created_segments.append((name, shm, "memory"))

        except Exception as e:
            print(f"   ⚠️  Could not create {name}: {e}")

    return created_segments


def run_command(cmd, description, wait=True, shell=True):
    """Run a command with description"""
    print(f"\n🔄 {description}")
    print(f"   Command: {cmd}")

    try:
        if wait:
            result = subprocess.run(
                cmd, shell=shell, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(f"   ✅ Success: {description}")
                if result.stdout.strip():
                    print(f"   Output: {result.stdout.strip()}")
            else:
                print(f"   ❌ Failed: {description}")
                print(f"   Error: {result.stderr.strip()}")
            return result.returncode == 0
        else:
            subprocess.Popen(cmd, shell=shell)
            print(f"   🚀 Started: {description}")
            return True
    except subprocess.TimeoutExpired:
        print(f"   ⏰ Timeout: {description}")
        return False
    except Exception as e:
        print(f"   💥 Exception: {e}")
        return False


def check_file_exists(filepath):
    """Check if file exists"""
    exists = os.path.exists(filepath)
    print(f"   {'✅' if exists else '❌'} File: {filepath}")
    return exists


def main(quiet_mode=False, td_project=None, model_path=None, width=1280, height=720, fps_limit=60):
    # Validate TouchDesigner project if provided
    if td_project and not td_project.endswith(".toe"):
        print(f"❌ Invalid project file: {td_project}")
        print("Usage: python setup_all.py [path/to/project.toe] [--quiet]")
        return

    print("🎬 YOLO TouchDesigner Automation Setup")
    print("=" * 50)
    if td_project:
        print(f"🎯 Will launch project: {td_project}")

    # Step 1: Kill existing processes
    print("\n📋 Step 1: Cleaning up existing YOLO processes")
    killed = kill_specific_processes()
    if killed:
        print(f"   Killed processes: {', '.join(set(killed))}")
        time.sleep(2)  # Give processes time to die
    else:
        print("   No YOLO processes to kill")

    # Step 2: Create shared memory (before TouchDesigner setup)
    create_shared_memory(width=width, height=height, fps_limit=fps_limit)

    # Step 4: Make scripts executable
    print("\n📋 Step 3: Making scripts executable")
    run_command("chmod +x ./start_yolo_connect.sh", "Make start script executable")

    # Step 5: Start YOLO server with default model
    print("\n📋 Step 4: Starting YOLO server")

    # Use specified model or search for one
    if model_path:
        if os.path.exists(model_path):
            model_to_use = model_path
        else:
            print(f"   ❌ Specified model not found: {model_path}")
            return False
    else:
        # Try to find a model file
        model_files = ["yolo11n-pose.pt", "models/yolo11n-pose.pt", "models/yolo11x-pose.pt", "models/yolov8n-pose.pt", "models/yolo8n.pt"]
        model_to_use = None
        for model in model_files:
            if os.path.exists(model):
                model_to_use = model
                break

    if model_to_use:
        # Build command with optional quiet logging
        if quiet_mode:
            cmd = f"python processing.py -p {model_to_use} --log_config yolo_models/log_set/log_settings_quiet.yaml --shared_update_mem_name yolo_states --shared_params_mem_name params --shared_array_mem_name image -iw {width} -ih {height}"
            print("   🔇 Using quiet logging configuration")
            print(f"   📐 Using resolution: {width}x{height}")
        else:
            # Need to pass dimensions to the shell script too
            cmd = f"python processing.py -p {model_to_use} --shared_update_mem_name yolo_states --shared_params_mem_name params --shared_array_mem_name image -iw {width} -ih {height}"

        success = run_command(cmd, "Start YOLO server", wait=False)
        print(f"   Using model: {model_to_use}")
    else:
        print("   ❌ No pose model found! Please download a YOLO pose model:")
        print(
            "   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n-pose.pt"
        )
        return False

    if success:
        print("   ⏳ Waiting 5 seconds for server to initialize...")
        time.sleep(5)

    # Step 6: Start debug server
    print("\n📋 Step 5: Starting debug server")
    run_command("python debug_server.py", "Start debug server", wait=False)
    time.sleep(2)

    # Step 7: TouchDesigner launch (optional)
    print("\n📋 Step 6: TouchDesigner")

    if td_project:
        expanded_path = os.path.expanduser(td_project)
        if os.path.exists(expanded_path):
            print(f"   🚀 Launching TouchDesigner with {td_project}...")
            run_command(f"./launch_touchdesigner.sh '{td_project}'", "Launch TouchDesigner", wait=False)
            time.sleep(5)

            print("\n📋 Step 8: Final Setup")
            print("   ⚡ In TouchDesigner (should be opening now):")
            print("   1. Click 'Setup' on Script TOP")
            print("   2. Click 'Setup' on Script CHOP") 
            print("   3. DONE! Pose detection should work!")
        else:
            print(f"   ❌ Project not found: {expanded_path}")
            td_project = None

    if not td_project:
        print("   📋 Manual TouchDesigner setup:")
        print("   1. Open TouchDesigner manually")
        print("   2. Open your YOLO project")  
        print("   3. Click 'Setup' on Script TOP")
        print("   4. Click 'Setup' on Script CHOP")
        print("   5. DONE!")
        print(f"\n   💡 Next time use: python setup_all.py path/to/your/project.toe")
        print(f"   🔇 For less console output: python setup_all.py --quiet")

    print("\n✨ Automation complete!")
    print("🎯 Your YOLO pose detection should now be running")
    print("\n\n🛑 To stop everything, run:")
    print("   python setup_all.py --stop")
    print("   or use Ctrl+C in this terminal")

    # Keep script running so user can see output
    try:
        print("\n⏳ Press Ctrl+C to stop all YOLO processes...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping YOLO processes...")
        kill_specific_processes()
        print("✅ YOLO processes stopped")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Setup and launch YOLO detection system"
    )
    parser.add_argument(
        "toe_path",
        nargs="?",
        default=None,
        help="Path to TouchDesigner .toe file to open",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Use quiet logging configuration (less console output)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to YOLO model file (e.g., models/yolov8n-pose.pt)",
    )
    parser.add_argument(
        "--width", "-w",
        type=int,
        default=1280,
        help="Video width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Video height (default: 720)",
    )
    parser.add_argument(
        "--fps-limit",
        type=int,
        default=60,
        help="FPS limit (max 60, default: 60)",
    )
    parser.add_argument("--stop", action="store_true", help="Stop all YOLO processes")

    args = parser.parse_args()

    if args.stop:
        print("🛑 Stopping YOLO processes...")
        killed = kill_specific_processes()
        if killed:
            print(f"✅ Stopped: {', '.join(set(killed))}")
        else:
            print("✅ No YOLO processes were running")
    else:
        main(quiet_mode=args.quiet, td_project=args.toe_path, model_path=args.model, width=args.width, height=args.height, fps_limit=args.fps_limit)
