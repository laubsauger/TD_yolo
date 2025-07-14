"""
TouchDesigner CHOP for FPS statistics - Optimized simplified version
"""
import numpy as np
from multiprocessing import shared_memory
from typing import Optional
from collections import deque
import time

# Constants
FPS_STATS_SIZE = 16384  # macOS minimum shared memory size
CHANNEL_NAMES = ['server_fps', 'server_process_ms', 'server_frames', 'td_fps']

# Global state
_fps_memory: Optional[shared_memory.SharedMemory] = None
_initialized = False
_channels_setup = False
_td_frame_times = deque(maxlen=100)
_last_td_frame_time: Optional[float] = None
_frame_count = 0
_stats_cache: Optional[np.ndarray] = None
_cache_frame = -1

# Pre-allocate arrays for efficiency
_zero_output = np.zeros(4, dtype=np.float32)


def onSetupParameters(scriptOp):
    """Setup parameters"""
    print("\n=== FPS Stats CHOP Setup (Optimized) ===")
    
    # Set timeslice to cook every frame
    scriptOp.par.timeslice = True
    
    # Create minimal parameters
    page = scriptOp.appendCustomPage("FPS")
    page.appendPulse("Connect", label="Connect to FPS Stats")
    
    print("[OK] Ready to connect")
    return


def onPulse(par):
    """Handle button clicks"""
    if par.name == "Connect":
        connect_to_fps_stats()


def setup_channels_once(scriptOp):
    """Setup channels only once"""
    global _channels_setup
    
    if not _channels_setup:
        scriptOp.clear()
        scriptOp.numChans = 4
        scriptOp.numSamples = 1
        
        # Set channel names if possible
        try:
            for i, name in enumerate(CHANNEL_NAMES):
                scriptOp[i].name = name
        except:
            pass
        
        _channels_setup = True


def connect_to_fps_stats():
    """Connect to FPS shared memory"""
    global _fps_memory, _initialized, _channels_setup
    
    print("\n[INFO] Connecting to FPS stats")
    
    try:
        _fps_memory = shared_memory.SharedMemory(name="fps_stats")
        
        # Verify size
        if _fps_memory.size < 16:
            print(f"[ERROR] FPS stats memory too small: {_fps_memory.size} bytes")
            _fps_memory.close()
            _fps_memory = None
            _initialized = False
            return
        
        print(f"[OK] Connected to FPS stats memory (size: {_fps_memory.size} bytes)")
        _initialized = True
        
        # Setup CHOP channels once
        scriptOp = op(me)
        if hasattr(scriptOp, 'numChans'):
            setup_channels_once(scriptOp)
            
            # Force channel names to override any input inheritance
            for i, name in enumerate(CHANNEL_NAMES):
                if i < scriptOp.numChans:
                    try:
                        scriptOp[i].name = name
                    except:
                        pass
        
    except FileNotFoundError:
        print("[ERROR] FPS stats memory not found - is YOLO server running?")
        _initialized = False
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        _initialized = False


def onCook(scriptOp):
    """Optimized cook with caching"""
    global _fps_memory, _initialized, _td_frame_times, _last_td_frame_time, _frame_count
    global _stats_cache, _cache_frame, _channels_setup
    
    # Ensure channel names are set correctly
    if _channels_setup:
        try:
            for i, name in enumerate(CHANNEL_NAMES):
                if i < scriptOp.numChans and scriptOp[i].name != name:
                    scriptOp[i].name = name
        except:
            pass
    
    if not _initialized or _fps_memory is None:
        # Output zeros efficiently - avoid numChans during cooking
        try:
            for i in range(4):
                scriptOp[i][0] = 0.0
        except:
            scriptOp.clear()
        return
    
    _frame_count += 1
    
    try:
        # Track TD frame timing efficiently
        current_time = time.time()
        if _last_td_frame_time is not None:
            _td_frame_times.append(current_time - _last_td_frame_time)
        _last_td_frame_time = current_time
        
        # Calculate TD FPS only when needed
        td_fps = 0.0
        if len(_td_frame_times) > 1:
            avg_frame_time = sum(_td_frame_times) / len(_td_frame_times)
            td_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        # Read stats with caching (only read every 5 frames)
        if _stats_cache is None or _frame_count - _cache_frame >= 5:
            _stats_cache = np.frombuffer(_fps_memory.buf[:16], dtype=np.float32).copy()
            _cache_frame = _frame_count
        
        # Bulk update CHOP channels - avoid numChans during cooking
        try:
            # Use cached stats
            scriptOp[0][0] = _stats_cache[0] if len(_stats_cache) > 0 else 0.0  # server_fps
            scriptOp[1][0] = _stats_cache[1] if len(_stats_cache) > 1 else 0.0  # server_process_ms
            scriptOp[2][0] = _stats_cache[2] if len(_stats_cache) > 2 else 0.0  # server_frames
            scriptOp[3][0] = td_fps  # td_fps
        except:
            pass  # Ignore if channels not ready
        
        # Write TD FPS back to shared memory
        if len(_stats_cache) >= 4:
            _stats_cache[3] = td_fps
            # Write back to shared memory
            np.frombuffer(_fps_memory.buf[:16], dtype=np.float32)[:] = _stats_cache
        
        # Reduced console output - every 600 frames instead of 300
        if _frame_count % 600 == 0:
            print(f"\n[FPS] Server: {_stats_cache[0]:.1f} | Process: {_stats_cache[1]:.1f}ms | TD: {td_fps:.1f}")
        
    except Exception as e:
        if _frame_count % 600 == 0:
            print(f"[ERROR] Reading FPS stats: {e}")
        # Output zeros on error - avoid numChans during cooking
        try:
            for i in range(4):
                scriptOp[i][0] = 0.0
        except:
            pass


def onDestroy():
    """Cleanup"""
    global _fps_memory, _channels_setup
    if _fps_memory is not None:
        _fps_memory.close()
        _fps_memory = None
    _channels_setup = False
    print("[INFO] Cleaned up FPS connection")