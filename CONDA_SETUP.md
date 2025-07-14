# Conda Environment Setup Guide

This guide will help you set up the conda environment for the YOLO Video Detection project.

## Prerequisites

1. Install Miniconda or Anaconda from [conda.io](https://conda.io)
2. Ensure conda is added to your PATH

## Environment Setup

### Option 1: CPU Environment (Default)

```bash
# Create the environment from the yml file
conda env create -f environment.yml

# Activate the environment
conda activate yolo-video-detection

# Install the project in development mode
pip install -e .
```

### Option 2: GPU Environment (NVIDIA GPU with CUDA)

```bash
# Create the GPU environment
conda env create -f environment-gpu.yml

# Activate the environment
conda activate yolo-video-detection-gpu

# Install the project in development mode
pip install -e .
```

### Option 3: Manual Setup

```bash
# Create a new conda environment
conda create -n yolo-video-detection python=3.9

# Activate the environment
conda activate yolo-video-detection

# Install base requirements
pip install -r requirements.base.txt

# For CPU-only PyTorch
pip install -r requirements.torch.cpu.txt

# OR for GPU PyTorch
pip install -r requirements.torch.gpu.txt

# Install development dependencies
pip install -r requirements.dev.txt

# Install the project in development mode
pip install -e .
```

## VSCode/Cursor Integration

1. Open the project in VSCode/Cursor
2. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
3. Type "Python: Select Interpreter"
4. Choose the conda environment you created (e.g., `yolo-video-detection`)

The `.vscode/settings.json` file has been configured to:
- Automatically detect conda environments
- Set up proper Python paths
- Configure linting and formatting
- Enable IntelliSense for the project structure

## Verifying the Setup

```bash
# Check that the environment is active
conda info --envs

# Verify Python version
python --version

# Test PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Test CUDA availability (GPU only)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run tests
pytest tests/
```

## Troubleshooting

### Issue: VSCode doesn't recognize the conda environment

1. Restart VSCode after creating the environment
2. Make sure conda is in your system PATH
3. Try manually setting the interpreter path in `.vscode/settings.json`:
   ```json
   "python.defaultInterpreterPath": "~/miniconda3/envs/yolo-video-detection/bin/python"
   ```

### Issue: Import errors in VSCode but code runs fine

1. Check that PYTHONPATH is set correctly in the terminal
2. Reload the VSCode window (`Cmd+R` or `Ctrl+R`)
3. Clear Python/Pylance cache: `Cmd+Shift+P` > "Python: Clear Cache and Reload Window"

### Issue: Cython compilation errors

```bash
# Ensure Cython and numpy are installed first
pip install cython numpy

# Then reinstall the project
pip install -e . --force-reinstall
```

## Environment Management

```bash
# List all environments
conda env list

# Update environment from yml file
conda env update -f environment.yml

# Export current environment
conda env export > environment-export.yml

# Remove environment
conda deactivate
conda env remove -n yolo-video-detection
```