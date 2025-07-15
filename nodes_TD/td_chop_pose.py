"""
TouchDesigner CHOP for pose keypoint data - Resolution aware + Optimized version
Combines v2 resolution awareness with performance optimizations
"""
import numpy as np
from multiprocessing import shared_memory
from typing import Optional, List, Tuple
import json
import struct

# Constants
BYTES_PER_PERSON = 224  # 56 floats * 4 bytes
FLOATS_PER_PERSON = 56
MAX_PERSONS = 10
CACHE_TTL = 5  # Cache validity in frames

# COCO Pose keypoint names
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear", 
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

# Mapping for swapped keypoints when video is flipped
KEYPOINT_SWAP_MAP = {
    "left_eye": "right_eye",
    "right_eye": "left_eye",
    "left_ear": "right_ear",
    "right_ear": "left_ear",
    "left_shoulder": "right_shoulder",
    "right_shoulder": "left_shoulder",
    "left_elbow": "right_elbow",
    "right_elbow": "left_elbow",
    "left_wrist": "right_wrist",
    "right_wrist": "left_wrist",
    "left_hip": "right_hip",
    "right_hip": "left_hip",
    "left_knee": "right_knee",
    "right_knee": "left_knee",
    "left_ankle": "right_ankle",
    "right_ankle": "left_ankle",
    "nose": "nose"  # nose stays the same
}

# Pre-computed channel names for different modes
CHANNEL_NAMES_SINGLE = (
    ['x1', 'y1', 'x2', 'y2', 'score'] +
    [f'{kpt}_{axis}' for kpt in KEYPOINT_NAMES for axis in ['x', 'y', 'conf']]
)

# Channel names with swapped left/right
CHANNEL_NAMES_SINGLE_FLIPPED = (
    ['x1', 'y1', 'x2', 'y2', 'score'] +
    [f'{KEYPOINT_SWAP_MAP.get(kpt, kpt)}_{axis}' for kpt in KEYPOINT_NAMES for axis in ['x', 'y', 'conf']]
)

def _generate_multi_person_channels(max_persons: int) -> List[str]:
    """Pre-generate channel names for multi-person mode"""
    channels = []
    for p in range(max_persons):
        prefix = f'person{p}_'
        channels.extend([
            f'{prefix}x1', f'{prefix}y1', f'{prefix}x2', f'{prefix}y2', f'{prefix}score'
        ])
        channels.extend([
            f'{prefix}{kpt}_{axis}' 
            for kpt in KEYPOINT_NAMES 
            for axis in ['x', 'y', 'conf']
        ])
    return channels

# Pre-computed channel names
CHANNEL_NAMES_MULTI = _generate_multi_person_channels(5)

# Global state
_pose_memory: Optional[shared_memory.SharedMemory] = None
_initialized = False
_configured_resolution: Optional[Tuple[int, int]] = None
_params_cache: Optional['PoseParamsCache'] = None
_pose_data_cache: Optional['PoseDataCache'] = None
_last_frame = -1
_single_channels_setup = False
_last_num_persons = -1

# Resolution info file
RESOLUTION_INFO_FILE = "/tmp/yolo_resolution.json"


class PoseParamsCache:
    """Cache for CHOP parameters"""
    def __init__(self):
        self.person_index = 0
        self.all_persons = False
        self.separate_channels = False
        self.swap_leftright = False  # For handling flipped video
        self.debug = False
        self._last_update = -1
        self._update_interval = 30
    
    def update_if_needed(self, scriptOp, frame):
        """Update parameters periodically"""
        if frame - self._last_update >= self._update_interval:
            # Safely access parameters
            if hasattr(scriptOp.par, 'Personindex'):
                self.person_index = scriptOp.par.Personindex.eval()
            if hasattr(scriptOp.par, 'Allpersons'):
                self.all_persons = scriptOp.par.Allpersons.eval()
            if hasattr(scriptOp.par, 'Separatechannels'):
                self.separate_channels = scriptOp.par.Separatechannels.eval()
            if hasattr(scriptOp.par, 'Swapleftright'):
                self.swap_leftright = scriptOp.par.Swapleftright.eval()
            if hasattr(scriptOp.par, 'Debug'):
                self.debug = scriptOp.par.Debug.eval()
            self._last_update = frame


class PoseDataCache:
    """Efficient pose data caching with version tracking"""
    def __init__(self):
        self.version = -1
        self.num_persons = 0
        self.data_buffer = None
        self._last_frame = -1
    
    def clear(self):
        """Clear the cache and release references"""
        self.version = -1
        self.num_persons = 0
        self.data_buffer = None
        self._last_frame = -1
    
    def check_and_update(self, pose_memory: shared_memory.SharedMemory, frame: int) -> bool:
        """Check if data changed and update cache. Returns True if data is new."""
        # Read header (only num_persons - 4 bytes)
        # Use struct.unpack to avoid numpy view issues
        num_persons = struct.unpack('i', pose_memory.buf[:4])[0]
        
        # Validate
        if num_persons < 0 or num_persons > MAX_PERSONS:
            self.num_persons = 0  # Reset to 0 if invalid
            return False
        
        # Use frame as version since backend doesn't write version
        new_version = frame
        
        # Check if data changed
        if new_version == self.version and frame - self._last_frame < CACHE_TTL:
            return False
        
        # Update cache
        self.version = new_version
        self.num_persons = num_persons
        self._last_frame = frame
        
        if num_persons > 0:
            # Read all person data at once
            data_size = num_persons * BYTES_PER_PERSON
            if data_size + 4 <= pose_memory.size:
                # IMPORTANT: Make a copy to avoid holding references to shared memory
                temp_data = np.frombuffer(
                    pose_memory.buf[4:4+data_size],  # Start from offset 4, not 8
                    dtype=np.float32
                ).reshape(num_persons, FLOATS_PER_PERSON)
                # Create a copy that doesn't reference the shared memory
                self.data_buffer = temp_data.copy()
                
                # Validate the data - check for NaN or unreasonable values
                if np.any(np.isnan(self.data_buffer)) or np.any(np.isinf(self.data_buffer)):
                    print(f"[WARNING] Invalid pose data detected (NaN/Inf) at frame {frame}")
                    self.num_persons = 0
                    self.data_buffer = None
                    return False
        else:
            self.data_buffer = None
        
        return True


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
    global _pose_memory, _initialized, _configured_resolution, _params_cache, _pose_data_cache

    print("\n=== Pose Data CHOP Setup (Optimized) ===")
    
    # Enable automatic cooking
    
    # Initialize caches
    _params_cache = PoseParamsCache()
    _pose_data_cache = PoseDataCache()
    
    # Try to load resolution from main node
    resolution = load_resolution_info()
    if resolution:
        _configured_resolution = resolution
        if _logger:
            _logger.info(f"Loaded resolution from main node: {resolution[0]}x{resolution[1]}")
    else:
        if _logger:
            _logger.warning("No resolution info found - please setup main YOLO node first")
    
    # Setup parameters
    page = scriptOp.appendCustomPage("Pose Data")
    
    # Add info display
    p = page.appendStr("Info", label="Resolution")
    p.val = f"{resolution[0]}x{resolution[1]}" if resolution else "Not configured"
    p.readOnly = True
    
    # Setup buttons
    p = page.appendPulse("Connect", label="Connect to Pose Data")
    p = page.appendPulse("Reloadconfig", label="Reload Resolution Config")
    
    # Pose parameters
    params_config = [
        ("Personindex", "Int", {"label": "Person Index", "min": 0, "max": 10, "default": 0}),
        ("Allpersons", "Toggle", {"label": "Show All Persons", "default": False}),
        ("Separatechannels", "Toggle", {"label": "Separate Person Channels", "default": False}),
        ("Swapleftright", "Toggle", {"label": "Swap Left/Right (for flipped video)", "default": False}),
        ("Active", "Toggle", {"label": "Active", "default": True}),
        ("Debug", "Toggle", {"label": "Debug Output", "default": False})
    ]
    
    for name, ptype, props in params_config:
        if ptype == "Int":
            p = page.appendInt(name, label=props["label"])
            p.min = props["min"]
            p.max = props["max"]
        else:
            p = page.appendToggle(name, label=props["label"])
        p.default = props["default"]
        p.val = props["default"]
    
    # Logging level control
    p = page.appendMenu('Loglevel', label='Log Level')
    p.menuNames = ['Off', 'Error', 'Warning', 'Info', 'Debug']
    p.menuLabels = ['Off', 'Error', 'Warning', 'Info', 'Debug']
    p.default = 'Info'
    p.val = 'Info'
    p.help = 'Control logging verbosity'
    
    return


def onPulse(par):
    """Handle button clicks"""
    global _configured_resolution
    
    if par.name == "Connect":
        connect_to_pose_data()
    elif par.name == "Reloadconfig":
        resolution = load_resolution_info()
        if resolution:
            _configured_resolution = resolution
            if _logger:
                _logger.info(f"Reloaded resolution config: {resolution[0]}x{resolution[1]}")
            # Update display
            try:
                scriptOp = op(me)
                if hasattr(scriptOp.par, 'Info'):
                    scriptOp.par.Info.val = f"{resolution[0]}x{resolution[1]}"
            except:
                pass
        else:
            if _logger:
                _logger.warning("No resolution config found")


def connect_to_pose_data():
    """Connect to pose shared memory"""
    global _pose_memory, _initialized, _configured_resolution, _params_cache, _pose_data_cache
    
    # Clear cache FIRST to release any numpy views
    if _pose_data_cache is not None:
        _pose_data_cache.clear()
    
    # Clean up any existing connection after clearing cache
    if _pose_memory is not None:
        try:
            _pose_memory.close()
        except:
            pass
        _pose_memory = None
        _initialized = False
    
    # Load resolution if not already loaded
    if _configured_resolution is None:
        _configured_resolution = load_resolution_info()
        if _configured_resolution is None:
            _logger.error("No resolution configured - please setup main YOLO node first")
            return
    
    print(f"\n[INFO] Connecting to pose data (resolution: {_configured_resolution[0]}x{_configured_resolution[1]})")
    
    # Initialize caches if they don't exist
    if _params_cache is None:
        _params_cache = PoseParamsCache()
    if _pose_data_cache is None:
        _pose_data_cache = PoseDataCache()
    
    try:
        # Connect to existing shared memory
        _pose_memory = shared_memory.SharedMemory(name="pose_data")
        _logger.info(f"Connected to pose memory (size: {_pose_memory.size} bytes)")
        _initialized = True
        
        # Update info display safely
        try:
            scriptOp = op(me)
            if hasattr(scriptOp.par, 'Info'):
                scriptOp.par.Info.val = f"{_configured_resolution[0]}x{_configured_resolution[1]} - Connected"
        except:
            pass
        
    except FileNotFoundError:
        _logger.error("Pose memory not found - is YOLO server running?")
        _initialized = False
    except Exception as e:
        _logger.error(f"Error: {e}")
        _initialized = False


def setup_channels_single_person(scriptOp, swap_leftright=False):
    """Efficiently setup channels for single person mode"""
    global _single_channels_setup, _last_swap_state
    
    # Check if we need to rebuild channels due to swap state change
    current_swap_state = swap_leftright
    need_rebuild = False
    
    if '_last_swap_state' not in globals():
        globals()['_last_swap_state'] = None
        need_rebuild = True
    elif _last_swap_state != current_swap_state:
        need_rebuild = True
        _single_channels_setup = False
    
    if not _single_channels_setup or need_rebuild:
        scriptOp.clear()
        channel_names = CHANNEL_NAMES_SINGLE_FLIPPED if swap_leftright else CHANNEL_NAMES_SINGLE
        for name in channel_names:
            scriptOp.appendChan(name)
        _single_channels_setup = True
        _last_swap_state = current_swap_state


def setup_channels_all_persons(scriptOp, num_persons: int):
    """Setup channels for all persons mode"""
    global _last_num_persons
    
    if _last_num_persons != num_persons:
        scriptOp.clear()
        scriptOp.numSamples = max(1, num_persons)
        
        # Add channels
        for name in CHANNEL_NAMES_SINGLE:
            scriptOp.appendChan(name)
        
        _last_num_persons = num_persons


def setup_channels_separate(scriptOp, num_persons: int):
    """Setup separate channels for each person"""
    max_persons = min(num_persons, 5)
    expected_channels = max_persons * FLOATS_PER_PERSON
    
    if scriptOp.numChans != expected_channels:
        scriptOp.clear()
        
        # Use pre-computed channel names
        channels_to_add = CHANNEL_NAMES_MULTI[:expected_channels]
        for name in channels_to_add:
            scriptOp.appendChan(name)


def onCook(scriptOp):
    """Optimized cook with caching"""
    global _pose_memory, _initialized, _params_cache, _pose_data_cache, _last_frame, _logger
    
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
    
    # Check if active
    if hasattr(scriptOp.par, 'Active') and not scriptOp.par.Active.eval():
        return
    
    if not _initialized or _pose_memory is None:
        # Try to auto-connect
        connect_to_pose_data()
        if not _initialized:
            scriptOp.clear()
            return
    
    # Type guards
    if _params_cache is None or _pose_data_cache is None:
        _logger.error("Missing required caches")
        scriptOp.clear()
        return
    
    # Get current frame
    try:
        current_frame = abs(int(scriptOp.time.frame))
    except:
        current_frame = _last_frame + 1
    
    # Update caches
    _params_cache.update_if_needed(scriptOp, current_frame)
    
    # Check and update pose data
    if _pose_memory is not None:
        try:
            _pose_data_cache.check_and_update(_pose_memory, current_frame)
        except BufferError as e:
            # Handle buffer export error - reconnect on next frame
            _logger.warning(f"BufferError in pose data: {e}. Reconnecting...")
            _pose_memory = None
            _initialized = False
            scriptOp.clear()
            return
    
    if _pose_data_cache.num_persons == 0:
        scriptOp.clear()
        return
    
    # Get cached data
    num_persons = _pose_data_cache.num_persons
    pose_data = _pose_data_cache.data_buffer
    
    # Debug output
    if _params_cache.debug and current_frame % 30 == 0:
        print(f"[DEBUG] Pose CHOP: {num_persons} persons detected")
        if pose_data is not None and len(pose_data) > 0:
            # Check first person's nose position
            nose_x = pose_data[0, 5]  # nose x
            nose_y = pose_data[0, 6]  # nose y
            nose_conf = pose_data[0, 7]  # nose conf
            print(f"[DEBUG] First person nose: x={nose_x:.1f}, y={nose_y:.1f}, conf={nose_conf:.3f}")
            # Check bbox
            print(f"[DEBUG] First person bbox: ({pose_data[0, 0]:.1f}, {pose_data[0, 1]:.1f}) - ({pose_data[0, 2]:.1f}, {pose_data[0, 3]:.1f})")
    
    # Process based on mode
    if _params_cache.separate_channels:
        # Separate channels mode
        setup_channels_separate(scriptOp, num_persons)
        
        # Bulk assign data
        if pose_data is not None:
            max_persons = min(num_persons, 5)
            flat_data = pose_data[:max_persons].flatten()
            
            # Direct assignment to all channels at once
            chans = scriptOp.chans()
            for i, value in enumerate(flat_data):
                if i < len(chans):
                    chans[i][0] = value
    
    elif _params_cache.all_persons:
        # All persons mode
        setup_channels_all_persons(scriptOp, num_persons)
        
        if pose_data is not None:
            # Bulk assign using numpy operations
            if scriptOp.numSamples == num_persons:
                # Can assign to multiple samples
                chans = scriptOp.chans()
                for ch_idx in range(min(FLOATS_PER_PERSON, len(chans))):
                    chans[ch_idx][:num_persons] = pose_data[:, ch_idx]
            else:
                # Single sample mode
                chans = scriptOp.chans()
                for ch_idx in range(min(FLOATS_PER_PERSON, len(chans))):
                    chans[ch_idx][0] = pose_data[0, ch_idx]
    
    else:
        # Single person mode
        person_idx = min(_params_cache.person_index, num_persons - 1)
        setup_channels_single_person(scriptOp, _params_cache.swap_leftright)
        
        if pose_data is not None and person_idx < len(pose_data):
            # Direct assignment
            person_values = pose_data[person_idx]
            chans = scriptOp.chans()
            for ch_idx in range(min(FLOATS_PER_PERSON, len(chans))):
                chans[ch_idx][0] = person_values[ch_idx]
    
    _last_frame = current_frame
    return


def onDestroy():
    """Cleanup"""
    global _pose_memory, _single_channels_setup, _last_num_persons, _pose_data_cache, _initialized, _params_cache
    
    # Clear all cache references FIRST before closing shared memory
    if _pose_data_cache is not None:
        _pose_data_cache.clear()
        _pose_data_cache = None
    
    _params_cache = None
    
    # Clear state
    _initialized = False
    _single_channels_setup = False
    _last_num_persons = -1
    
    # Close shared memory connection LAST
    if _pose_memory is not None:
        try:
            _pose_memory.close()
        except:
            pass
        _pose_memory = None
    
    print("[INFO] Pose CHOP cleaned up")