#!/usr/bin/env python3
"""
Centralized environment utilities for cross-platform venv management.
Provides reusable functions for environment detection, validation, and setup.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Optional, Tuple, Dict


class EnvironmentError(Exception):
    """Custom exception for environment-related errors"""
    pass


def get_platform_info() -> Dict[str, str]:
    """Get detailed platform information."""
    return {
        'system': platform.system(),  # Windows, Darwin, Linux
        'machine': platform.machine(),  # x86_64, arm64, etc.
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),  # CPython, PyPy, etc.
    }


def check_python_version(required_major: int = 3, required_minor: int = 9) -> None:
    """
    Check if Python version meets requirements.
    
    Args:
        required_major: Required major version (default: 3)
        required_minor: Required minor version (default: 9)
        
    Raises:
        EnvironmentError: If Python version is too old
    """
    if sys.version_info < (required_major, required_minor):
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        required = f"{required_major}.{required_minor}"
        raise EnvironmentError(
            f"Python {required}+ required, but found {current}\n"
            f"Please install Python {required} or newer."
        )


def is_in_virtualenv() -> bool:
    """Check if we're currently inside a virtual environment."""
    # Check for venv
    if hasattr(sys, 'real_prefix'):
        return True
    
    # Check for virtualenv/venv (Python 3.3+)
    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        return True
    
    # Check for conda (even though we're moving away from it)
    if os.environ.get('CONDA_DEFAULT_ENV'):
        return True
    
    # Check for common venv indicators
    if 'VIRTUAL_ENV' in os.environ:
        return True
    
    return False


def get_current_env_info() -> Optional[Dict[str, str]]:
    """
    Get information about the current virtual environment if any.
    
    Returns:
        Dict with env info or None if not in a virtual environment
    """
    if not is_in_virtualenv():
        return None
    
    info = {
        'type': 'unknown',
        'path': sys.prefix,
        'python': sys.executable,
    }
    
    # Detect environment type
    if os.environ.get('VIRTUAL_ENV'):
        info['type'] = 'venv'
        info['path'] = os.environ['VIRTUAL_ENV']
    elif os.environ.get('CONDA_DEFAULT_ENV'):
        info['type'] = 'conda'
        info['name'] = os.environ['CONDA_DEFAULT_ENV']
        info['path'] = os.environ.get('CONDA_PREFIX', sys.prefix)
    elif hasattr(sys, 'real_prefix'):
        info['type'] = 'virtualenv'
    
    return info


def get_venv_paths(venv_dir: Path) -> Dict[str, Path]:
    """
    Get platform-specific paths within a virtual environment.
    
    Args:
        venv_dir: Path to the virtual environment directory
        
    Returns:
        Dict with paths to python, pip, and scripts directory
    """
    venv_dir = Path(venv_dir)
    
    if platform.system() == 'Windows':
        return {
            'python': venv_dir / 'Scripts' / 'python.exe',
            'pip': venv_dir / 'Scripts' / 'pip.exe',
            'scripts': venv_dir / 'Scripts',
            'activate': venv_dir / 'Scripts' / 'activate.bat',
            'activate_ps': venv_dir / 'Scripts' / 'Activate.ps1',
        }
    else:
        return {
            'python': venv_dir / 'bin' / 'python',
            'pip': venv_dir / 'bin' / 'pip',
            'scripts': venv_dir / 'bin',
            'activate': venv_dir / 'bin' / 'activate',
        }


def check_venv_exists(venv_dir: Path) -> bool:
    """
    Check if a valid virtual environment exists at the given path.
    
    Args:
        venv_dir: Path to check for virtual environment
        
    Returns:
        True if valid venv exists, False otherwise
    """
    venv_dir = Path(venv_dir)
    if not venv_dir.exists():
        return False
    
    paths = get_venv_paths(venv_dir)
    
    # Check for essential files
    return (
        paths['python'].exists() and
        paths['pip'].exists() and
        (paths.get('activate') or paths.get('activate_ps')).exists()
    )


def create_venv(venv_dir: Path, with_pip: bool = True, system_site_packages: bool = False) -> None:
    """
    Create a new virtual environment.
    
    Args:
        venv_dir: Directory where venv should be created
        with_pip: Include pip in the environment (default: True)
        system_site_packages: Give venv access to system packages (default: False)
        
    Raises:
        EnvironmentError: If venv creation fails
    """
    venv_dir = Path(venv_dir)
    
    # Build command
    cmd = [sys.executable, '-m', 'venv']
    
    # Note: pip is included by default in venv, no need for --with-pip
    # The --without-pip flag exists if you want to exclude it
    if not with_pip:
        cmd.append('--without-pip')
    
    if system_site_packages:
        cmd.append('--system-site-packages')
    
    cmd.append(str(venv_dir))
    
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise EnvironmentError(f"Failed to create virtual environment: {e}")


def run_in_venv(venv_dir: Path, command: list, **kwargs) -> subprocess.CompletedProcess:
    """
    Run a command using the Python from a virtual environment.
    
    Args:
        venv_dir: Path to the virtual environment
        command: Command list to run
        **kwargs: Additional arguments for subprocess.run
        
    Returns:
        CompletedProcess instance
        
    Raises:
        EnvironmentError: If venv doesn't exist or command fails
    """
    if not check_venv_exists(venv_dir):
        raise EnvironmentError(f"Virtual environment not found at: {venv_dir}")
    
    paths = get_venv_paths(venv_dir)
    python_path = paths['python']
    
    # Ensure we use the venv's Python
    if command[0] in ['python', 'python3']:
        command[0] = str(python_path)
    
    # Set up environment
    env = os.environ.copy()
    env['VIRTUAL_ENV'] = str(venv_dir)
    env['PATH'] = f"{paths['scripts']}{os.pathsep}{env.get('PATH', '')}"
    
    # Remove any conda variables
    for key in list(env.keys()):
        if key.startswith('CONDA_'):
            del env[key]
    
    return subprocess.run(command, env=env, **kwargs)


def detect_gpu_availability() -> Dict[str, bool]:
    """
    Detect available GPU backends.
    
    Returns:
        Dict with 'cuda' and 'mps' availability
    """
    availability = {
        'cuda': False,
        'mps': False,
    }
    
    # Check for NVIDIA GPU
    if platform.system() == 'Windows':
        try:
            subprocess.check_output('nvidia-smi', shell=True, stderr=subprocess.DEVNULL)
            availability['cuda'] = True
        except:
            pass
    else:
        # Unix-like systems
        if os.path.exists('/dev/nvidia0') or os.path.exists('/dev/nvidiactl'):
            availability['cuda'] = True
        
        # Check nvidia-smi as backup
        try:
            subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL)
            availability['cuda'] = True
        except:
            pass
    
    # Check for Apple Silicon GPU (MPS)
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        availability['mps'] = True
    
    return availability


def get_touchdesigner_paths() -> Dict[str, list]:
    """
    Get platform-specific TouchDesigner installation paths.
    
    Returns:
        Dict mapping platform to list of possible TD paths
    """
    return {
        'Windows': [
            r'C:\Program Files\Derivative\TouchDesigner',
            r'C:\Program Files (x86)\Derivative\TouchDesigner',
            # Add version-specific paths if needed
        ],
        'Darwin': [  # macOS
            '/Applications/TouchDesigner.app/Contents/MacOS/TouchDesigner',
            '/Applications/TouchDesigner099.app/Contents/MacOS/TouchDesigner',
            # Add more versions as needed
        ],
        'Linux': [
            '/opt/touchdesigner/bin/touchdesigner',
            '/usr/local/bin/touchdesigner',
            # Add more common paths
        ]
    }


def find_touchdesigner() -> Optional[Path]:
    """
    Find TouchDesigner executable on the system.
    
    Returns:
        Path to TouchDesigner executable or None if not found
    """
    system = platform.system()
    possible_paths = get_touchdesigner_paths().get(system, [])
    
    for path_str in possible_paths:
        path = Path(path_str)
        if path.exists():
            return path
    
    # Try to find in PATH
    if system == 'Windows':
        executables = ['TouchDesigner.exe', 'TouchDesigner099.exe']
    else:
        executables = ['TouchDesigner', 'touchdesigner']
    
    for exe in executables:
        try:
            result = subprocess.run(
                ['which' if system != 'Windows' else 'where', exe],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return Path(result.stdout.strip().split('\n')[0])
        except:
            pass
    
    return None


def setup_touchdesigner_env(venv_dir: Path) -> Dict[str, str]:
    """
    Set up environment variables for TouchDesigner to use venv.
    
    Args:
        venv_dir: Path to virtual environment
        
    Returns:
        Dict of environment variables to set
    """
    venv_dir = Path(venv_dir)
    paths = get_venv_paths(venv_dir)
    
    env = {
        'PYTHONPATH': str(Path.cwd()),  # Project directory
        'VIRTUAL_ENV': str(venv_dir),
        'PATH': f"{paths['scripts']}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    
    # Platform-specific settings
    if platform.system() == 'Windows':
        # Windows-specific Python DLL paths if needed
        pass
    elif platform.system() == 'Darwin':
        # macOS-specific settings
        # TouchDesigner on macOS might need DYLD_LIBRARY_PATH
        pass
    
    return env


def validate_requirements_files() -> Dict[str, bool]:
    """
    Check which requirements files exist in the project.
    
    Returns:
        Dict mapping requirement file names to existence boolean
    """
    files = {
        'requirements.txt': Path('requirements.txt'),
        'requirements.base.txt': Path('requirements.base.txt'),
        'requirements.cpu.txt': Path('requirements.cpu.txt'),
        'requirements.gpu.txt': Path('requirements.gpu.txt'),
        'requirements.dev.txt': Path('requirements.dev.txt'),
    }
    
    return {name: path.exists() for name, path in files.items()}


# Convenience function for scripts
def ensure_venv(venv_dir: Path = Path('venv'), required_python: Tuple[int, int] = (3, 9)) -> Path:
    """
    Ensure a virtual environment exists and meets requirements.
    
    Args:
        venv_dir: Path to virtual environment (default: 'venv')
        required_python: Required Python version tuple (default: (3, 9))
        
    Returns:
        Path to the virtual environment
        
    Raises:
        EnvironmentError: If requirements not met
    """
    # Check Python version
    check_python_version(*required_python)
    
    # Check if we're in a different virtual environment
    current_env = get_current_env_info()
    if current_env and not str(venv_dir).endswith(os.path.basename(current_env['path'])):
        print(f"WARNING: Currently in {current_env['type']} environment: {current_env.get('name', current_env['path'])}")
        print("This might cause conflicts. Consider deactivating it first.")
    
    # Create venv if it doesn't exist
    if not check_venv_exists(venv_dir):
        print(f"Creating virtual environment at: {venv_dir}")
        create_venv(venv_dir)
    
    return venv_dir