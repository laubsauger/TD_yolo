#!/usr/bin/env python3
"""
Cross-platform virtual environment setup script for TD_yolo project.
Creates and configures a venv without polluting global Python namespace.

NOTE: If you get a Rye error, run this script directly with python3:
    python3 setup_env.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.env_utils import (
    check_python_version,
    is_in_virtualenv,
    get_current_env_info,
    create_venv,
    check_venv_exists,
    get_venv_paths,
    detect_gpu_availability,
    run_in_venv,
    validate_requirements_files,
    get_platform_info
)

# Global flag for non-interactive mode
NON_INTERACTIVE = False


def print_header():
    """Print setup header."""
    print("=" * 60)
    print("TD_yolo Virtual Environment Setup")
    print("=" * 60)
    
    # Show platform info
    info = get_platform_info()
    print(f"Platform: {info['system']} ({info['machine']})")
    print(f"Python: {info['python_version']} ({info['python_implementation']})")
    print()


def check_prerequisites():
    """Check all prerequisites before setup."""
    print("Checking prerequisites...")
    
    # Check Python version
    try:
        check_python_version(3, 9)
        print("✓ Python version: OK")
    except Exception as e:
        print(f"✗ Python version: {e}")
        return False
    
    # Check for existing virtual environment
    current_env = get_current_env_info()
    if current_env:
        print(f"\n⚠️  WARNING: Currently in {current_env['type']} environment!")
        print(f"   Path: {current_env['path']}")
        if current_env['type'] == 'conda':
            print(f"   Name: {current_env.get('name', 'unknown')}")
        print("\n   It's recommended to deactivate this environment first.")
        
        if NON_INTERACTIVE:
            print("\n   Continuing anyway (non-interactive mode)")
        else:
            response = input("\n   Continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("\nSetup cancelled. Please deactivate your environment and try again.")
                return False
    
    # Check requirements files
    print("\nChecking requirements files...")
    req_files = validate_requirements_files()
    
    if not req_files['requirements.base.txt']:
        print("✗ Missing requirements.base.txt")
        return False
    
    if not (req_files['requirements.cpu.txt'] or req_files['requirements.gpu.txt']):
        print("✗ Missing both requirements.cpu.txt and requirements.gpu.txt")
        return False
    
    print("✓ Requirements files: OK")
    
    return True


def detect_best_requirements():
    """Detect which requirements file to use based on GPU availability."""
    print("\nDetecting hardware capabilities...")
    
    gpu_info = detect_gpu_availability()
    
    if gpu_info['cuda']:
        print("✓ NVIDIA GPU detected (CUDA available)")
        return 'requirements.gpu.txt'
    elif gpu_info['mps']:
        print("✓ Apple Silicon GPU detected (MPS available)")
        # For now, use CPU requirements for MPS
        # TODO: Add specific MPS requirements if needed
        return 'requirements.cpu.txt'
    else:
        print("ℹ No GPU detected, using CPU requirements")
        return 'requirements.cpu.txt'


def setup_virtual_environment(venv_dir: Path):
    """Create and set up the virtual environment."""
    print(f"\nSetting up virtual environment at: {venv_dir}")
    
    # Check if venv already exists
    if check_venv_exists(venv_dir):
        print("ℹ Virtual environment already exists")
        if NON_INTERACTIVE:
            print("   Using existing environment (non-interactive mode)")
            return True
        else:
            response = input("   Recreate it? (y/N): ").strip().lower()
            if response == 'y':
                print("   Removing existing environment...")
                import shutil
                shutil.rmtree(venv_dir)
            else:
                print("   Using existing environment")
                return True
    
    # Create new venv
    print("Creating virtual environment...")
    try:
        create_venv(venv_dir)  # pip is included by default
        print("✓ Virtual environment created")
        return True
    except Exception as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return False


def upgrade_pip(venv_dir: Path):
    """Upgrade pip in the virtual environment."""
    print("\nUpgrading pip...")
    
    try:
        result = run_in_venv(
            venv_dir,
            ['python', '-m', 'pip', 'install', '--upgrade', 'pip'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ pip upgraded successfully")
            return True
        else:
            print(f"✗ Failed to upgrade pip: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error upgrading pip: {e}")
        return False


def install_requirements(venv_dir: Path, requirements_file: str):
    """Install requirements in the virtual environment."""
    print(f"\nInstalling requirements from {requirements_file}...")
    
    if not Path(requirements_file).exists():
        print(f"✗ Requirements file not found: {requirements_file}")
        return False
    
    try:
        # Install requirements
        result = run_in_venv(
            venv_dir,
            ['python', '-m', 'pip', 'install', '-r', requirements_file],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ Requirements installed successfully")
            
            # Install package in development mode
            print("\nInstalling project in development mode...")
            result = run_in_venv(
                venv_dir,
                ['python', '-m', 'pip', 'install', '-e', '.'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✓ Project installed in development mode")
                return True
            else:
                print(f"✗ Failed to install project: {result.stderr}")
                return False
        else:
            print(f"✗ Failed to install requirements: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error installing requirements: {e}")
        return False


def show_activation_instructions(venv_dir: Path):
    """Show instructions for activating the environment."""
    paths = get_venv_paths(venv_dir)
    platform_info = get_platform_info()
    
    print("\n" + "=" * 60)
    print("✓ Setup Complete!")
    print("=" * 60)
    
    print("\nTo activate the virtual environment:")
    
    if platform_info['system'] == 'Windows':
        print(f"\n  Command Prompt:")
        print(f"    {paths['activate']}")
        print(f"\n  PowerShell:")
        print(f"    {paths['activate_ps']}")
    else:
        print(f"    source {paths['activate']}")
    
    print("\nNext steps:")
    print("1. Activate the virtual environment (see above)")
    print("2. Start the YOLO server:")
    print("   python start_yolo_server.py models/yolo11n-pose.pt")
    print("3. Launch TouchDesigner:")
    print("   python launch_touchdesigner.py [project.toe]")
    
    print("\nOr use the all-in-one setup:")
    print("   python setup_all.py -m models/yolo11n-pose.pt")


def main():
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup virtual environment for TD_yolo')
    parser.add_argument('--yes', '-y', action='store_true', 
                       help='Answer yes to all prompts (non-interactive mode)')
    args = parser.parse_args()
    
    # Default venv directory
    venv_dir = Path('venv')
    
    # Print header
    print_header()
    
    # Store non-interactive mode
    global NON_INTERACTIVE
    NON_INTERACTIVE = args.yes
    
    # Check prerequisites
    if not check_prerequisites():
        return 1
    
    # Setup virtual environment
    if not setup_virtual_environment(venv_dir):
        return 1
    
    # Upgrade pip
    if not upgrade_pip(venv_dir):
        return 1
    
    # Detect and install requirements
    requirements_file = detect_best_requirements()
    if not install_requirements(venv_dir, requirements_file):
        return 1
    
    # Show activation instructions
    show_activation_instructions(venv_dir)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)