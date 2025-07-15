# Virtual Environment Setup Research

## Is Using Python Scripts for venv Setup Common Practice?

Yes, using Python scripts to set up virtual environments is a common and recommended practice, especially for cross-platform projects. Here are several examples and considerations:

## Common Approaches in the Wild

### 1. **Poetry** (Most Popular Python Dependency Manager)
- Uses `poetry install` which is a Python script
- Manages venv creation internally
- Cross-platform by design

### 2. **Pipenv**
- Uses `pipenv install` command
- Python script that manages virtualenv creation
- Handles environment isolation automatically

### 3. **Many Popular Projects Use Setup Scripts**
- **Django**: Has management commands for setup
- **Flask**: Often includes setup.py or make_venv.py scripts
- **Scientific packages**: Commonly include environment setup scripts

## Best Practices for Safe venv Setup Scripts

### 1. **Use subprocess and sys.executable**
```python
import subprocess
import sys
import os

def create_venv(venv_path):
    # Use the current Python interpreter to ensure compatibility
    subprocess.check_call([sys.executable, '-m', 'venv', venv_path])
```

### 2. **Avoid Global Namespace Pollution**
```python
# setup_env.py
def main():
    """All logic inside functions to avoid global pollution"""
    # Setup code here
    pass

if __name__ == "__main__":
    main()
    # Script exits cleanly without leaving variables in global space
```

### 3. **Check for Existing Environments**
```python
import os
import sys

def check_venv_exists(venv_path):
    """Check if venv already exists to avoid overwriting"""
    if os.path.exists(venv_path):
        activate_script = os.path.join(
            venv_path, 
            'Scripts' if sys.platform == 'win32' else 'bin',
            'activate'
        )
        return os.path.exists(activate_script)
    return False
```

### 4. **Handle Platform Differences**
```python
def get_venv_python(venv_path):
    """Get the Python executable path in the venv"""
    if sys.platform == 'win32':
        return os.path.join(venv_path, 'Scripts', 'python.exe')
    else:
        return os.path.join(venv_path, 'bin', 'python')
```

## Potential Issues and Solutions

### 1. **System Python Pollution**
**Issue**: Running setup script might install packages in system Python
**Solution**: 
- Always create venv first
- Use explicit paths to venv's pip
- Never use `pip install` without activation

### 2. **Permission Issues**
**Issue**: May need admin rights on some systems
**Solution**:
- Create venv in user-writable location (project directory)
- Avoid system directories
- Check permissions before operations

### 3. **Python Version Mismatches**
**Issue**: User might have wrong Python version
**Solution**:
```python
import sys

def check_python_version():
    if sys.version_info < (3, 9):
        print(f"Error: Python 3.9+ required, found {sys.version}")
        sys.exit(1)
```

### 4. **Existing Virtual Environment Conflicts**
**Issue**: User might already be in a venv/conda environment
**Solution**:
```python
def warn_existing_env():
    if hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    ):
        print("WARNING: Already in a virtual environment!")
        print("Consider deactivating it first.")
        return True
    
    # Check for conda
    if os.environ.get('CONDA_DEFAULT_ENV'):
        print("WARNING: Conda environment detected!")
        print(f"Current env: {os.environ.get('CONDA_DEFAULT_ENV')}")
        return True
    
    return False
```

## Recommended Setup Script Structure

```python
#!/usr/bin/env python3
"""
Safe cross-platform virtual environment setup script.
This script creates and configures a venv without polluting global Python.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def main():
    """Main setup function - all logic contained here"""
    # 1. Check Python version
    check_python_version()
    
    # 2. Warn about existing environments
    if warn_existing_env():
        if not confirm("Continue anyway?"):
            return
    
    # 3. Create venv
    venv_path = Path("venv")  # or ".venv" for hidden
    create_venv(venv_path)
    
    # 4. Install requirements
    install_requirements(venv_path)
    
    # 5. Show activation instructions
    show_activation_instructions(venv_path)


def check_python_version():
    """Ensure we have the right Python version"""
    required = (3, 9)
    if sys.version_info < required:
        print(f"Error: Python {required[0]}.{required[1]}+ required")
        print(f"Found: Python {sys.version}")
        sys.exit(1)


def create_venv(venv_path):
    """Create virtual environment"""
    print(f"Creating virtual environment in '{venv_path}'...")
    subprocess.check_call([sys.executable, '-m', 'venv', str(venv_path)])
    print("✓ Virtual environment created")


def get_venv_executable(venv_path, command):
    """Get path to executable in venv"""
    if platform.system() == 'Windows':
        return venv_path / 'Scripts' / f'{command}.exe'
    return venv_path / 'bin' / command


def install_requirements(venv_path):
    """Install requirements using venv's pip"""
    pip_path = get_venv_executable(venv_path, 'pip')
    python_path = get_venv_executable(venv_path, 'python')
    
    # Upgrade pip first
    print("Upgrading pip...")
    subprocess.check_call([str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip'])
    
    # Detect GPU availability
    requirements_file = detect_requirements_file()
    
    print(f"Installing requirements from {requirements_file}...")
    subprocess.check_call([str(pip_path), 'install', '-r', requirements_file])
    print("✓ Requirements installed")


def detect_requirements_file():
    """Detect which requirements file to use"""
    try:
        import torch
        if torch.cuda.is_available():
            return 'requirements.gpu.txt'
    except ImportError:
        pass
    
    # Check for NVIDIA GPU on system
    if platform.system() == 'Windows':
        # Windows-specific GPU detection
        try:
            subprocess.check_output('nvidia-smi', shell=True)
            return 'requirements.gpu.txt'
        except:
            pass
    else:
        # Unix-like systems
        if os.path.exists('/dev/nvidia0') or os.path.exists('/dev/nvidiactl'):
            return 'requirements.gpu.txt'
    
    return 'requirements.cpu.txt'


def show_activation_instructions(venv_path):
    """Show how to activate the venv"""
    print("\n" + "="*50)
    print("✓ Setup complete!")
    print("="*50)
    print("\nTo activate the virtual environment:")
    
    if platform.system() == 'Windows':
        print(f"  Command Prompt: {venv_path}\\Scripts\\activate.bat")
        print(f"  PowerShell: {venv_path}\\Scripts\\Activate.ps1")
    else:
        print(f"  source {venv_path}/bin/activate")
    
    print("\nTo start the YOLO server:")
    print("  python start_yolo_server.py models/yolo11n-pose.pt")
    
    print("\nTo launch TouchDesigner:")
    print("  python launch_touchdesigner.py [project.toe]")


def warn_existing_env():
    """Check and warn about existing virtual environments"""
    # Implementation as shown above
    pass


def confirm(message):
    """Get yes/no confirmation from user"""
    while True:
        response = input(f"{message} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False


if __name__ == "__main__":
    # All code runs inside main() - no global pollution
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during setup: {e}")
        sys.exit(1)
```

## Conclusion

Using a Python script for venv setup is:
1. **Common practice** in modern Python projects
2. **Safer than shell scripts** for cross-platform compatibility  
3. **Recommended** when done properly with contained functions
4. **Used by major tools** like Poetry, Pipenv, and many others

The key is to:
- Keep all logic in functions (no global code execution)
- Use `if __name__ == "__main__":` guard
- Handle platform differences explicitly
- Check for existing environments
- Use subprocess with explicit paths
- Exit cleanly without side effects