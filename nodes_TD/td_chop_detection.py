"""
TouchDesigner CHOP for detection data - Resolution aware + Optimized version
Combines v2 resolution awareness with performance optimizations
"""
import numpy as np
from multiprocessing import shared_memory
from typing import Optional, Tuple
import json

# Constants
BYTES_PER_DETECTION = 24  # 6 floats * 4 bytes
FLOATS_PER_DETECTION = 6
MAX_DETECTIONS = 100
CHANNEL_NAMES = ['x1', 'y1', 'x2', 'y2', 'score', 'class_id']

# Global state
_detection_memory: Optional[shared_memory.SharedMemory] = None
_initialized = False
_configured_resolution: Optional[Tuple[int, int]] = None
_channels_initialized = False
_last_num_detections = -1
_detection_cache: Optional[np.ndarray] = None
_cache_version = -1

# Resolution info file
RESOLUTION_INFO_FILE = "/tmp/yolo_resolution.json"


def load_resolution_info() -> Optional[Tuple[int, int]]:
    """Load resolution info from main node"""
    try:
        with open(RESOLUTION_INFO_FILE, 'r') as f:
            data = json.load(f)
            return (data['width'], data['height'])
    except:
        return None


def onSetupParameters(scriptOp):
    """Setup parameters"""
    global _detection_memory, _initialized, _configured_resolution

    print("\n=== Detection Data CHOP Setup (Optimized) ===")
    
    # Try to load resolution from main node
    resolution = load_resolution_info()
    if resolution:
        _configured_resolution = resolution
        print(f"[OK] Loaded resolution from main node: {resolution[0]}x{resolution[1]}")
    else:
        print("[WARNING] No resolution info found - please setup main YOLO node first")
    
    # Setup parameters
    page = scriptOp.appendCustomPage("Detection Data")
    
    # Add info display
    p = page.appendStr("Info", label="Resolution")
    p.val = f"{resolution[0]}x{resolution[1]}" if resolution else "Not configured"
    p.readOnly = True
    
    # Setup button
    p = page.appendPulse("Connect", label="Connect to Detection Data")
    
    return


def onPulse(par):
    """Handle button clicks"""
    if par.name == "Connect":
        connect_to_detection_data()


def connect_to_detection_data():
    """Connect to detection shared memory"""
    global _detection_memory, _initialized, _configured_resolution
    
    # Load resolution if not already loaded
    if _configured_resolution is None:
        _configured_resolution = load_resolution_info()
        if _configured_resolution is None:
            print("[ERROR] No resolution configured - please setup main YOLO node first")
            return
    
    print(f"\n[INFO] Connecting to detection data (resolution: {_configured_resolution[0]}x{_configured_resolution[1]})")
    
    try:
        _detection_memory = shared_memory.SharedMemory(name="detection_data")
        print(f"[OK] Connected to detection memory (size: {_detection_memory.size} bytes)")
        _initialized = True
        
        # Update info display safely
        try:
            scriptOp = op(me)
            if hasattr(scriptOp.par, 'Info'):
                scriptOp.par.Info.val = f"{_configured_resolution[0]}x{_configured_resolution[1]} - Connected"
        except:
            pass
        
    except FileNotFoundError:
        print("[ERROR] Detection memory not found - is YOLO server running?")
        _initialized = False
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        _initialized = False


def setup_channels(scriptOp, num_detections: int):
    """Efficiently setup or update channels"""
    global _channels_initialized, _last_num_detections
    
    # Only update if needed
    if not _channels_initialized or scriptOp.numSamples != max(1, num_detections):
        scriptOp.numSamples = max(1, num_detections)
        
        # Only recreate channels if not initialized
        if not _channels_initialized:
            scriptOp.clear()
            for name in CHANNEL_NAMES:
                scriptOp.appendChan(name)
            _channels_initialized = True
        
        _last_num_detections = num_detections


def onCook(scriptOp):
    """Optimized cook with caching and bulk operations"""
    global _detection_memory, _initialized, _detection_cache, _cache_version
    
    if not _initialized or _detection_memory is None:
        scriptOp.clear()
        return
    
    try:
        # Read header efficiently
        header = np.frombuffer(_detection_memory.buf[:8], dtype=np.int32)
        num_detections = header[0]
        version = header[1] if len(header) > 1 else 0
        
        # Early validation
        if num_detections < 0 or num_detections > MAX_DETECTIONS:
            # Invalid data - output zeros
            setup_channels(scriptOp, 1)
            for ch in scriptOp.chans:
                ch[0] = 0.0
            return
        
        # Setup channels efficiently
        setup_channels(scriptOp, num_detections)
        
        if num_detections == 0:
            # No detections - single zero sample
            for ch in scriptOp.chans:
                ch[0] = 0.0
            return
        
        # Check cache validity
        if version != _cache_version or _detection_cache is None or len(_detection_cache) != num_detections:
            # Read all detection data at once
            data_size = num_detections * BYTES_PER_DETECTION
            if data_size + 8 <= _detection_memory.size:
                _detection_cache = np.frombuffer(
                    _detection_memory.buf[8:8+data_size],
                    dtype=np.float32
                ).reshape(num_detections, FLOATS_PER_DETECTION)
                _cache_version = version
        
        # Bulk assign to channels using numpy
        if _detection_cache is not None and len(_detection_cache) >= num_detections:
            for ch_idx in range(FLOATS_PER_DETECTION):
                # Assign entire column at once
                scriptOp.chans[ch_idx][:num_detections] = _detection_cache[:num_detections, ch_idx]
        
    except Exception as e:
        print(f"Error in optimized detection reader: {e}")
        # Fallback to safe output
        setup_channels(scriptOp, 1)
        for ch in scriptOp.chans:
            ch[0] = 0.0
    
    return


def onDestroy():
    """Cleanup"""
    global _detection_memory, _channels_initialized
    if _detection_memory is not None:
        _detection_memory.close()
        _detection_memory = None
    _channels_initialized = False