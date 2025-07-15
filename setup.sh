#!/bin/bash
# Unix setup script for virtual environment
# This handles cross-platform Python discovery

echo "TD_yolo Virtual Environment Setup"
echo "================================="

# Function to check Python version
check_python() {
    local cmd=$1
    if command -v $cmd &> /dev/null; then
        version=$($cmd --version 2>&1 | grep -oE 'Python 3\.[0-9]+' | head -1)
        if [[ -n $version ]]; then
            minor=$(echo $version | cut -d'.' -f2)
            if [[ $minor -ge 9 ]]; then
                # Warn about Python 3.13
                if [[ $minor -ge 13 ]]; then
                    echo "Warning: $version has compatibility issues with some dependencies."
                    echo "Continuing search for Python 3.9-3.12..."
                    return 1
                fi
                echo "Found $version via $cmd"
                $cmd setup_env.py
                return 0
            fi
        fi
    fi
    return 1
}

# Try different Python commands - prioritize stable versions
if check_python python3.11; then
    exit 0
elif check_python python3.12; then
    exit 0
elif check_python python3.10; then
    exit 0
elif check_python python3.9; then
    exit 0
elif check_python python3; then
    exit 0
elif check_python python; then
    exit 0
elif check_python /usr/bin/python3; then
    exit 0
elif check_python /usr/local/bin/python3; then
    exit 0
elif check_python /opt/homebrew/bin/python3; then
    exit 0
else
    echo "ERROR: Python 3.9 or later not found!"
    echo ""
    echo "Please install Python 3.9+ using:"
    echo "  macOS: brew install python@3.11"
    echo "  Ubuntu/Debian: sudo apt install python3.11"
    echo "  Other: Visit https://python.org"
    exit 1
fi