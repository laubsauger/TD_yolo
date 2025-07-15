#!/usr/bin/env python3
"""
Cross-platform YOLO server launcher.
Replaces start_yolo_connect.sh with Python implementation.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.env_utils import (
    ensure_venv,
    get_venv_paths,
    run_in_venv,
    check_venv_exists,
    is_in_virtualenv,
    get_current_env_info
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start YOLO server for TouchDesigner integration"
    )
    
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to YOLO model file (e.g., models/yolo11n-pose.pt)'
    )
    
    parser.add_argument(
        '--venv',
        type=str,
        default='venv',
        help='Path to virtual environment (default: venv)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=None,
        help='Image width (auto-detected based on model type if not specified)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=None,
        help='Image height (auto-detected based on model type if not specified)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Use quiet logging configuration'
    )
    
    parser.add_argument(
        '--no-venv-check',
        action='store_true',
        help='Skip virtual environment checks'
    )
    
    return parser.parse_args()


def detect_model_type(model_path):
    """Detect if model is for pose estimation based on filename."""
    model_name = Path(model_path).name.lower()
    return 'pose' in model_name


def validate_model_path(model_path):
    """Validate that the model file exists."""
    path = Path(model_path)
    if not path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        print("\nPlease provide a valid model path.")
        print("Example: python start_yolo_server.py models/yolo11n-pose.pt")
        return False
    return True


def get_resolution(model_path, width, height):
    """Determine resolution based on model type or user input."""
    if width is not None and height is not None:
        # User specified both dimensions
        return width, height
    
    # Auto-detect based on model type
    if detect_model_type(model_path):
        print("📐 Detected pose model - using 640x640 resolution")
        return 640, 640
    else:
        print("📐 Using standard 1280x720 resolution for detection model")
        return 1280, 720


def check_environment(venv_path, skip_check=False):
    """Check and prepare the environment."""
    if skip_check:
        return True
    
    # Check if we're in the correct virtual environment
    current_env = get_current_env_info()
    venv_path = Path(venv_path).resolve()
    
    if not is_in_virtualenv():
        print("⚠️  Not in a virtual environment!")
        print(f"   Expected venv at: {venv_path}")
        
        if not check_venv_exists(venv_path):
            print("\n❌ Virtual environment not found!")
            print("   Please run: python setup_env.py")
            return False
        
        print("\n   To activate the environment:")
        paths = get_venv_paths(venv_path)
        if sys.platform == 'win32':
            print(f"   Command Prompt: {paths['activate']}")
            print(f"   PowerShell: {paths['activate_ps']}")
        else:
            print(f"   source {paths['activate']}")
        return False
    
    # Check if we're in the wrong environment
    if current_env:
        env_path = Path(current_env['path']).resolve()
        if env_path != venv_path:
            print(f"⚠️  WARNING: In different environment: {current_env['path']}")
            print(f"   Expected: {venv_path}")
    
    return True


def build_command(model_path, width, height, quiet=False):
    """Build the command to run the YOLO server."""
    cmd = [
        'python', 'processing.py',
        '-p', model_path,
        '--shared_update_mem_name', 'yolo_states',
        '--shared_params_mem_name', 'params',
        '--shared_array_mem_name', 'image',
        '-iw', str(width),
        '-ih', str(height),
        '-c', '3',
        '--image_type', 'float32'
    ]
    
    if quiet:
        cmd.extend(['--log_config', 'yolo_models/log_set/log_settings_quiet.yaml'])
    
    return cmd


def main():
    """Main function."""
    args = parse_arguments()
    
    print("🚀 Starting YOLO Server")
    print("=" * 50)
    
    # Validate model path
    if not validate_model_path(args.model_path):
        return 1
    
    # Get resolution
    width, height = get_resolution(args.model_path, args.width, args.height)
    
    # Check environment
    venv_path = Path(args.venv)
    if not args.no_venv_check and not check_environment(venv_path, args.no_venv_check):
        return 1
    
    # Build command
    cmd = build_command(args.model_path, width, height, args.quiet)
    
    print(f"\n📋 Configuration:")
    print(f"   Model: {args.model_path}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Logging: {'Quiet' if args.quiet else 'Normal'}")
    
    print("\n🔄 Connecting to TouchDesigner shared memory...")
    print("   - yolo_states: Synchronization flags")
    print("   - params: Configuration parameters")
    print("   - image: Frame buffer")
    print("   - detection_data: Bounding boxes")
    print("   - pose_data: Keypoints")
    
    print("\n▶️  Starting YOLO detection server...")
    print("   Press Ctrl+C to stop\n")
    
    try:
        # If we're in a venv, just run directly
        if is_in_virtualenv() or args.no_venv_check:
            subprocess.run(cmd)
        else:
            # Run in the specified venv
            result = run_in_venv(venv_path, cmd)
            return result.returncode
    except KeyboardInterrupt:
        print("\n\n🛑 YOLO server stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())