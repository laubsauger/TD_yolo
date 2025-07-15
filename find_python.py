#!/usr/bin/env python3
"""
Helper script to find a suitable Python 3 interpreter.
This handles various cross-platform scenarios including Rye.
"""

import sys
import subprocess
import os
from pathlib import Path


def find_python():
    """Find a suitable Python 3 interpreter."""
    
    # First, check if we're already running Python 3.9+
    if sys.version_info >= (3, 9):
        return sys.executable
    
    # List of Python commands to try
    candidates = []
    
    # Platform-specific candidates
    if sys.platform == 'win32':
        candidates.extend([
            'python',  # Windows usually has python.exe
            'python3',
            'py -3',   # Python Launcher for Windows
        ])
        
        # Check common Windows Python locations
        for version in ['313', '312', '311', '310', '39']:
            candidates.extend([
                f'C:\\Python{version}\\python.exe',
                f'C:\\Program Files\\Python{version}\\python.exe',
                f'C:\\Users\\{os.environ.get("USERNAME", "")}\\AppData\\Local\\Programs\\Python\\Python{version}\\python.exe',
            ])
    else:
        # Unix-like systems (macOS, Linux)
        candidates.extend([
            'python3',
            'python',
            '/usr/bin/python3',
            '/usr/local/bin/python3',
            '/opt/homebrew/bin/python3',  # macOS ARM
            '/usr/local/opt/python/bin/python3',  # macOS Intel
        ])
    
    # Try each candidate
    for cmd in candidates:
        try:
            # Handle special case for 'py -3' on Windows
            if cmd == 'py -3':
                result = subprocess.run(
                    ['py', '-3', '--version'],
                    capture_output=True,
                    text=True
                )
            else:
                result = subprocess.run(
                    [cmd, '--version'],
                    capture_output=True,
                    text=True,
                    shell=False
                )
            
            if result.returncode == 0:
                # Parse version
                version_str = result.stdout.strip()
                if 'Python 3.' in version_str:
                    # Extract version number
                    version_parts = version_str.split()[1].split('.')
                    major = int(version_parts[0])
                    minor = int(version_parts[1])
                    
                    if major == 3 and minor >= 9:
                        # Found suitable Python!
                        if cmd == 'py -3':
                            return 'py -3'
                        return cmd
        except:
            continue
    
    return None


def main():
    python_cmd = find_python()
    
    if python_cmd:
        print(f"Found Python: {python_cmd}")
        # Get full path if possible
        try:
            if python_cmd == 'py -3':
                result = subprocess.run(['py', '-3', '-c', 'import sys; print(sys.executable)'], 
                                      capture_output=True, text=True)
            else:
                result = subprocess.run([python_cmd, '-c', 'import sys; print(sys.executable)'], 
                                      capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Full path: {result.stdout.strip()}")
        except:
            pass
        
        return 0
    else:
        print("ERROR: No suitable Python 3.9+ interpreter found!")
        print("\nPlease install Python 3.9 or later from https://python.org")
        print("\nSearched for:")
        print("  - python")
        print("  - python3")
        if sys.platform == 'win32':
            print("  - py -3 (Python Launcher)")
        print("  - Various system paths")
        return 1


if __name__ == "__main__":
    sys.exit(main())