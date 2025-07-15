"""
TouchDesigner CHOP for FPS statistics - Optimized simplified version
"""
import numpy as np
from multiprocessing import shared_memory
from typing import Optional
from collections import deque
import time
from td_logging import TDLogger, get_logger

# Constants
FPS_STATS_SIZE = 16384  # macOS minimum shared memory size
CHANNEL_NAMES = ['server_fps', 'server_process_ms', 'server_frames', 'td_fps']

# Global state
_logger: Optional[TDLogger] = None
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
    global _logger
    
    # Initialize logger
    _logger = get_logger(parent(), TDLogger.LEVEL_INFO)
    
    _logger.log("\n=== FPS Stats CHOP Setup (Optimized) ===")
    
    # Set timeslice to cook every frame
    scriptOp.par.timeslice = True
    
    # Create minimal parameters
    page = scriptOp.appendCustomPage("FPS")
    page.appendPulse("Connect", label="Connect to FPS Stats")
    
    # Auto-connect toggle
    p = page.appendToggle("Autoconnect", label="Auto Connect")
    p.default = True
    p.val = True
    p.help = "Automatically connect on startup"
    
    # Logging level control
    p = page.appendMenu('Loglevel', label='Log Level')
    p.menuNames = ['Off', 'Error', 'Warning', 'Info', 'Debug']
    p.menuLabels = ['Off', 'Error', 'Warning', 'Info', 'Debug']
    p.default = 'Info'
    p.val = 'Info'
    p.help = 'Control logging verbosity'
    
    # Try to auto-connect immediately
    if hasattr(scriptOp.par, 'Autoconnect') and scriptOp.par.Autoconnect.eval():
        try:
            import threading
            # Delay connection slightly to ensure shared memory is ready
            threading.Timer(0.5, connect_to_fps_stats).start()
        except:
            pass
    
    _logger.info("Ready to connect")
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
    
    _logger.info("Connecting to FPS stats")
    
    try:
        _fps_memory = shared_memory.SharedMemory(name="fps_stats")
        
        # Verify size
        if _fps_memory.size < 16:
            _logger.error(f"FPS stats memory too small: {_fps_memory.size} bytes")
            _fps_memory.close()
            _fps_memory = None
            _initialized = False
            return
        
        _logger.info(f"Connected to FPS stats memory (size: {_fps_memory.size} bytes)")
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
        _logger.error("FPS stats memory not found - is YOLO server running?")
        _initialized = False
    except Exception as e:
        _logger.error(f"Connection failed: {e}")
        _initialized = False


def onCook(scriptOp):
    """Optimized cook with caching"""
    global _fps_memory, _initialized, _td_frame_times, _last_td_frame_time, _frame_count
    global _stats_cache, _cache_frame, _channels_setup, _logger
    
    # Initialize logger if needed
    if _logger is None:
        _logger = get_logger(parent(), TDLogger.LEVEL_INFO)
    
    # Update logger level based on parameter
    try:
        if hasattr(scriptOp.par, 'Loglevel'):
            level_str = scriptOp.par.Loglevel.eval()
            level_map = {
                'Off': TDLogger.LEVEL_OFF,
                'Error': TDLogger.LEVEL_ERROR,
                'Warning': TDLogger.LEVEL_WARNING,
                'Info': TDLogger.LEVEL_INFO,
                'Debug': TDLogger.LEVEL_DEBUG
            }
            _logger.set_level(level_map.get(level_str, TDLogger.LEVEL_INFO))
    except:
        pass
    
    # Ensure channel names are set correctly
    if _channels_setup:
        try:
            for i, name in enumerate(CHANNEL_NAMES):
                if i < scriptOp.numChans and scriptOp[i].name != name:
                    scriptOp[i].name = name
        except:
            pass
    
    if not _initialized or _fps_memory is None:
        # Try to auto-connect on first few cooks if not connected and auto-connect is enabled
        if _frame_count < 10 and _frame_count % 3 == 0:
            if hasattr(scriptOp.par, 'Autoconnect') and scriptOp.par.Autoconnect.eval():
                connect_to_fps_stats()
        
        # Output zeros efficiently - avoid numChans during cooking
        try:
            for i in range(4):
                scriptOp[i][0] = 0.0
        except:
            scriptOp.clear()
            
        if _frame_count % 120 == 0 and not _initialized:
            _logger.warning("FPS stats not connected - click Connect button or check YOLO server")
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
            
            # Debug log the raw values
            if _frame_count % 60 == 0:
                _logger.debug(f"Raw FPS stats: {_stats_cache}")
                _logger.debug(f"Server FPS: {_stats_cache[0]:.2f}, Process MS: {_stats_cache[1]:.2f}, Frames: {_stats_cache[2]:.0f}, TD FPS: {_stats_cache[3]:.2f}")
        
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
            _logger.info(f"FPS - Server: {_stats_cache[0]:.1f} | Process: {_stats_cache[1]:.1f}ms | TD: {td_fps:.1f} | Frames: {_stats_cache[2]:.0f}")
        
    except Exception as e:
        if _frame_count % 600 == 0:
            _logger.error(f"Reading FPS stats: {e}")
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
    if _logger:
        _logger.info("Cleaned up FPS connection")