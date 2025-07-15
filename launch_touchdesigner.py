#!/usr/bin/env python3
"""
Cross-platform TouchDesigner launcher with virtual environment support.
Replaces launch_touchdesigner.sh with Python implementation.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.env_utils import (
    find_touchdesigner,
    setup_touchdesigner_env,
    check_venv_exists,
    get_venv_paths,
    is_in_virtualenv,
    get_current_env_info,
    get_platform_info
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch TouchDesigner with Python virtual environment"
    )
    
    parser.add_argument(
        'toe_file',
        nargs='?',
        default=None,
        help='Path to TouchDesigner .toe project file (optional)'
    )
    
    parser.add_argument(
        '--td-path',
        type=str,
        default=None,
        help='Path to TouchDesigner executable (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--venv',
        type=str,
        default='venv',
        help='Path to virtual environment (default: venv)'
    )
    
    parser.add_argument(
        '--no-venv',
        action='store_true',
        help='Launch without setting up virtual environment'
    )
    
    return parser.parse_args()


def validate_toe_file(toe_path):
    """Validate TouchDesigner project file if provided."""
    if not toe_path:
        return None
    
    # Expand user path
    toe_path = Path(toe_path).expanduser()
    
    if not toe_path.exists():
        print(f"❌ Error: Project file not found: {toe_path}")
        return None
    
    if not toe_path.suffix.lower() == '.toe':
        print(f"⚠️  Warning: File doesn't have .toe extension: {toe_path}")
    
    return toe_path.resolve()


def find_td_executable(td_path=None):
    """Find TouchDesigner executable."""
    if td_path:
        # User specified path
        td_path = Path(td_path)
        if not td_path.exists():
            print(f"❌ Error: TouchDesigner not found at: {td_path}")
            return None
        return td_path
    
    # Auto-detect
    print("🔍 Searching for TouchDesigner...")
    td_exe = find_touchdesigner()
    
    if not td_exe:
        platform_info = get_platform_info()
        print(f"❌ TouchDesigner not found!")
        print(f"\nPlease install TouchDesigner or specify the path with --td-path")
        
        if platform_info['system'] == 'Darwin':
            print("\nOn macOS, TouchDesigner is usually at:")
            print("  /Applications/TouchDesigner.app/Contents/MacOS/TouchDesigner")
        elif platform_info['system'] == 'Windows':
            print("\nOn Windows, TouchDesigner is usually at:")
            print("  C:\\Program Files\\Derivative\\TouchDesigner\\bin\\TouchDesigner.exe")
        elif platform_info['system'] == 'Linux':
            print("\nOn Linux, TouchDesigner is usually at:")
            print("  /opt/touchdesigner/bin/touchdesigner")
        
        return None
    
    print(f"✓ Found TouchDesigner at: {td_exe}")
    return td_exe


def setup_environment(venv_path):
    """Set up environment variables for TouchDesigner."""
    venv_path = Path(venv_path).resolve()
    
    # Check if venv exists
    if not check_venv_exists(venv_path):
        print(f"❌ Virtual environment not found at: {venv_path}")
        print("   Please run: python setup_env.py")
        return None
    
    # Get environment variables
    env_vars = setup_touchdesigner_env(venv_path)
    
    # Merge with current environment
    env = os.environ.copy()
    env.update(env_vars)
    
    # Add project directory to PYTHONPATH
    project_dir = Path(__file__).parent.resolve()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = str(project_dir)
    
    return env


def check_current_environment(venv_path):
    """Check and warn about current environment."""
    current_env = get_current_env_info()
    venv_path = Path(venv_path).resolve()
    
    if current_env:
        env_path = Path(current_env['path']).resolve()
        if current_env['type'] == 'conda':
            print(f"⚠️  WARNING: Currently in conda environment: {current_env.get('name', 'unknown')}")
            print("   TouchDesigner might not use the correct Python environment.")
        elif env_path != venv_path:
            print(f"⚠️  WARNING: Currently in different venv: {current_env['path']}")
            print(f"   Expected: {venv_path}")


def launch_touchdesigner(td_exe, toe_file=None, env=None):
    """Launch TouchDesigner with optional project file."""
    cmd = [str(td_exe)]
    
    if toe_file:
        cmd.append(str(toe_file))
    
    print("\n🚀 Launching TouchDesigner...")
    if toe_file:
        print(f"   Project: {toe_file}")
    else:
        print("   No project file specified")
    
    if env and 'VIRTUAL_ENV' in env:
        print(f"   Virtual Environment: {env['VIRTUAL_ENV']}")
        print(f"   Python Path: {env.get('PYTHONPATH', 'Not set')}")
    
    print("\n   TouchDesigner is starting...")
    print("   This window will remain open while TD is running.")
    print("   Press Ctrl+C to stop TouchDesigner.\n")
    
    try:
        # Launch TouchDesigner
        process = subprocess.Popen(cmd, env=env)
        
        # Wait for it to complete
        process.wait()
        
        return process.returncode
    except KeyboardInterrupt:
        print("\n\n🛑 TouchDesigner stopped by user")
        process.terminate()
        return 0
    except Exception as e:
        print(f"\n❌ Error launching TouchDesigner: {e}")
        return 1


def main():
    """Main function."""
    args = parse_arguments()
    
    print("🎬 TouchDesigner Launcher")
    print("=" * 50)
    
    # Validate project file if provided
    toe_file = validate_toe_file(args.toe_file)
    
    # Find TouchDesigner executable
    td_exe = find_td_executable(args.td_path)
    if not td_exe:
        return 1
    
    # Set up environment unless --no-venv specified
    env = None
    if not args.no_venv:
        venv_path = Path(args.venv)
        
        # Check current environment
        check_current_environment(venv_path)
        
        # Set up TouchDesigner environment
        env = setup_environment(venv_path)
        if not env:
            return 1
        
        print(f"\n✓ Virtual environment configured: {venv_path}")
    else:
        print("\n⚠️  Launching without virtual environment setup")
    
    # Launch TouchDesigner
    return launch_touchdesigner(td_exe, toe_file, env)


if __name__ == "__main__":
    sys.exit(main())