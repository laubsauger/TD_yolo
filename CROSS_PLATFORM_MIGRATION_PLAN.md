# Cross-Platform Migration Plan: Conda to Venv

## Executive Summary

This document outlines the migration strategy from conda to venv for the TD_yolo project to achieve cross-platform compatibility (Windows, macOS, Linux). The migration involves replacing shell scripts with Python-based launchers and ensuring TouchDesigner can access the virtual environment dependencies.

## Research Findings

### Shared Memory Compatibility

After researching Python's `multiprocessing.shared_memory` module, **the current implementation should work on Windows without modification**:

- The module abstracts platform differences (Windows uses memory-mapped files, Unix uses POSIX shared memory)
- Named segments work the same across platforms
- The macOS 16KB minimum size requirement is already handled in the code
- ShareableList works identically on all platforms

**Key considerations for Windows testing:**
- Process permissions may differ
- Resource cleanup behavior should be verified
- Performance characteristics may vary but API remains the same

### Virtual Environment Setup Best Practices

Using Python scripts for venv setup is **common practice** and recommended:

- Major tools like Poetry and Pipenv use Python scripts for environment management
- Safer than shell scripts for cross-platform compatibility
- Allows proper error handling and platform detection

**Key safety measures:**
- Keep all logic in functions (avoid global namespace pollution)
- Use `if __name__ == "__main__":` guard
- Check for existing environments before creation
- Use subprocess with explicit paths to venv executables

## Current State Analysis

### Environment Management
- **Current**: Uses conda environments (`yolo_env`, `yolo-video-detection`, `yolo-video-detection-gpu`)
- **Issue**: Environment name inconsistency between scripts and yml files
- **TouchDesigner Dependency**: TD requires access to Python packages (ultralytics, etc.)

### Platform-Specific Issues
1. **Shell Scripts**: Unix/macOS only (.sh files)
   - `start_yolo_connect.sh`: Launches YOLO server
   - `launch_touchdesigner.sh`: Launches TD with conda environment
2. **Hardcoded Paths**:
   - `~/miniconda3/etc/profile.d/conda.sh`
   - `/Applications/TouchDesigner.app` (macOS only)
3. **No Windows Support**: Missing .bat/.ps1/.cmd equivalents

### Key Dependencies
- Python 3.9
- PyTorch (CPU/GPU variants)
- Ultralytics YOLO
- OpenCV
- Cython (for performance extensions)
- Shared memory for TD communication

## Migration Strategy

### Phase 1: Environment Migration (conda → venv)

#### 1.1 Create venv Setup Scripts
- **setup_env.py**: Cross-platform environment setup
  - Detect OS and Python version
  - Create venv in project directory
  - Install appropriate requirements based on GPU availability
  - Handle torch index URLs for CPU/GPU

#### 1.2 Requirements Restructuring
- Keep existing requirements files structure
- Add `requirements.txt` as main entry point
- Ensure pip can resolve all dependencies without conda

#### 1.3 TouchDesigner Integration
- **Challenge**: TD needs to find and use venv packages
- **Solution**: 
  - Set PYTHONPATH environment variable
  - Create TD launcher that activates venv
  - Consider embedded Python approach

### Phase 2: Cross-Platform Launchers

#### 2.1 Replace Shell Scripts with Python
- **start_yolo_server.py**: Replace `start_yolo_connect.sh`
  - Cross-platform process management
  - Automatic venv activation
  - Model detection and resolution handling
  
- **launch_touchdesigner.py**: Replace `launch_touchdesigner.sh`
  - OS-specific TD path detection
  - Environment variable setup
  - Project file handling

#### 2.2 Unified Setup Script Enhancement
- Enhance `setup_all.py` to:
  - Work with venv instead of conda
  - Provide cross-platform process management
  - Handle Windows-specific shared memory quirks

### Phase 3: Platform-Specific Handling

#### 3.1 TouchDesigner Paths
Handled in `utils/env_utils.py` with the `find_touchdesigner()` function that:
- Searches common installation paths per platform
- Falls back to PATH search
- Returns None if not found (user must specify path)

#### 3.2 Shared Memory Considerations
- Windows: May need different shared memory implementation
- Test multiprocessing.shared_memory compatibility
- Document any platform-specific limitations

### Phase 4: Build and Distribution

#### 4.1 Development Workflow
```bash
# Universal commands (all platforms)
python setup_env.py          # Create and setup venv
python start_yolo_server.py models/yolo11n-pose.pt
python launch_touchdesigner.py project.toe
```

#### 4.2 Binary Distribution
- Use cibuildwheel for Cython extensions
- Consider PyInstaller for standalone executables
- Package venv activation into distributions

## Implementation Todos

### High Priority
- [x] Create `setup_env.py` for cross-platform venv setup
- [x] Create centralized `utils/env_utils.py` for environment management
- [x] Create `start_yolo_server.py` to replace shell script
- [x] Create `launch_touchdesigner.py` to replace shell script
- [ ] Test venv with TouchDesigner on all platforms
- [ ] Update `setup_all.py` for venv compatibility

### Medium Priority
- [x] Update README.md with cross-platform instructions
- [x] Add platform detection to Python scripts (in env_utils.py)
- [ ] Test shared memory on Windows
- [ ] Update CLAUDE.md with new commands
- [ ] Create migration guide for existing users
- [ ] Remove conda-related files after testing

### Low Priority
- [ ] Add PowerShell/batch convenience scripts
- [ ] Consider GUI launcher for non-technical users
- [ ] Package as standalone executables
- [ ] Add to Windows PATH automatically

## Testing Plan

### Environment Testing
1. Create venv on each platform
2. Install all dependencies
3. Verify torch GPU detection
4. Test Cython compilation

### Integration Testing
1. Launch YOLO server with venv
2. Connect TouchDesigner to venv Python
3. Verify shared memory communication
4. Test pose and detection models

### Platform-Specific Testing
- **Windows**: Test with Windows 10/11, various TD versions
- **macOS**: Test on Intel and Apple Silicon
- **Linux**: Test on Ubuntu 22.04+

## Migration Steps for Users

### For Existing Users
1. Backup current setup
2. Run migration script: `python migrate_to_venv.py`
3. Update launch commands to use new Python scripts
4. Remove conda environments (optional)

### For New Users
1. Clone repository
2. Run: `python setup_env.py`
3. Run: `python setup_all.py -m models/yolo11n-pose.pt`
4. Open TouchDesigner project

## Migration Approach

Full commitment to venv:
1. No conda fallback or `--use-conda` flags
2. Clean removal of all conda-related files after migration
3. Single, clear path forward with venv only
4. Centralized environment utilities for consistency

## Success Criteria

- [x] All platforms can run setup without shell scripts
- [ ] TouchDesigner successfully uses venv packages (needs testing)
- [ ] Performance remains unchanged (needs benchmarking)
- [ ] Installation time reduced by 30% (venv is faster than conda)
- [x] No platform-specific code in main scripts (all handled in env_utils)
- [x] Documentation covers all platforms equally

## Completed Items

### Week 1 Progress
- ✅ Created centralized environment utilities (`utils/env_utils.py`)
- ✅ Implemented cross-platform venv setup script (`setup_env.py`)
- ✅ Replaced shell scripts with Python equivalents:
  - `start_yolo_server.py` replaces `start_yolo_connect.sh`
  - `launch_touchdesigner.py` replaces `launch_touchdesigner.sh`
- ✅ Updated README with cross-platform instructions
- ✅ Removed conda fallback strategy - full commitment to venv

### Remaining Work
- Week 2: Update `setup_all.py` and test on all platforms
- Week 3: Full platform testing (Windows, macOS, Linux)
- Week 4: Migration guide and cleanup of conda files

## Notes

### TouchDesigner Python Integration
- TD uses system Python or embedded Python
- Must ensure TD Python version matches venv Python
- May need to set TD preferences for Python executable

### Shared Memory on Windows
- Windows may have different size limits
- Named shared memory might behave differently
- Test thoroughly with real-time performance

### GPU Support
- CUDA paths differ across platforms
- Consider using torch's automatic CUDA detection
- ROCm support for AMD GPUs (future consideration)