"""
TouchDesigner YOLO Integration - Final Optimized Version
Combines resolution configuration with performance optimizations
"""
import numpy as np
import cv2
from multiprocessing import shared_memory
from typing import Optional, Tuple, Dict
import json
import os
import time
from collections import deque

# Configuration files
CONFIG_FILE = "/tmp/yolo_td_config.json"
RESOLUTION_INFO_FILE = "/tmp/yolo_resolution.json"

# Constants for drawing
DRAW_TEXT = 1
DRAW_BBOX = 2
DRAW_CONF = 4
DRAW_SKELETON = 8
OVERLAY_ONLY = 16
TRANSPARENT_BG = 32
DRAW_KEYPOINT_CONF = 64
MIRROR_LABELS = 128

# Flip constants
FLIP_NONE = -2
FLIP_BOTH = -1
FLIP_VERTICAL = 0
FLIP_HORIZONTAL = 1

# Global state
_shm_states: Optional[shared_memory.SharedMemory] = None
_sl_params: Optional[shared_memory.ShareableList] = None
_shm_image: Optional[shared_memory.SharedMemory] = None
_shm_detections: Optional[shared_memory.SharedMemory] = None
_shm_pose: Optional[shared_memory.SharedMemory] = None
_shm_fps: Optional[shared_memory.SharedMemory] = None
_initialized = False
_configured_resolution: Tuple[int, int] = (1280, 720)
_shared_array: Optional[np.ndarray] = None
_frame_count = 0
_last_output: Optional[np.ndarray] = None
_last_output_time: Optional[float] = None
_processing_frame = False

# Performance optimizations
_params_cache: Optional["ParamsCache"] = None
_array_views: Dict[str, np.ndarray] = {}
_flip_lookup = {
    (True, True): FLIP_BOTH,
    (True, False): FLIP_VERTICAL,
    (False, True): FLIP_HORIZONTAL,
    (False, False): FLIP_NONE,
}

# FPS tracking
_td_frame_times = deque(maxlen=100)
_last_td_frame_time: Optional[float] = None
_total_td_frames = 0


class ParamsCache:
    """Cache for TouchDesigner parameters to avoid repeated access"""
    
    def __init__(self):
        self.flip_v = True  # Default to True for webcam compatibility
        self.flip_h = True  # Default to True (works for standard webcam)
        self.mirror = False  # Mirror the final output
        self.mirror_labels = True  # Mirror the labels (True when flip_h is True)
        self.nms = 0.5
        self.score = 0.5
        self.maxk = 1
        self.pose_threshold = 0.1
        self.smoothing = 0.7
        self.draw_bbox = True
        self.draw_skeleton = True
        self.draw_text = True
        self.draw_score = True
        self.overlay_only = False  # Changed to False to prevent accumulation
        self.draw_keypoint_conf = False
        self.debug = False
        self._last_update = -1
        self._update_interval = 1  # Update every frame for responsiveness
    
    def update_if_needed(self, scriptOp, frame_count):
        """Update cache periodically"""
        if frame_count - self._last_update >= self._update_interval:
            self._update_all(scriptOp)
            self._last_update = frame_count
    
    def _update_all(self, scriptOp):
        """Batch update all parameters"""
        # Use hasattr to check parameter existence
        if hasattr(scriptOp.par, 'Flipv'):
            self.flip_v = scriptOp.par.Flipv.eval()
        if hasattr(scriptOp.par, 'Fliph'):
            self.flip_h = scriptOp.par.Fliph.eval()
        if hasattr(scriptOp.par, 'Mirror'):
            self.mirror = scriptOp.par.Mirror.eval()
        if hasattr(scriptOp.par, 'Mirrorlabels'):
            self.mirror_labels = scriptOp.par.Mirrorlabels.eval()
        if hasattr(scriptOp.par, 'Nms'):
            self.nms = scriptOp.par.Nms.eval()
        if hasattr(scriptOp.par, 'Score'):
            self.score = scriptOp.par.Score.eval()
        if hasattr(scriptOp.par, 'Maxk'):
            self.maxk = scriptOp.par.Maxk.eval()
        if hasattr(scriptOp.par, 'Posethreshold'):
            self.pose_threshold = scriptOp.par.Posethreshold.eval()
        if hasattr(scriptOp.par, 'Smoothing'):
            self.smoothing = scriptOp.par.Smoothing.eval()
        if hasattr(scriptOp.par, 'Drawbbox'):
            self.draw_bbox = scriptOp.par.Drawbbox.eval()
        if hasattr(scriptOp.par, 'Drawskeleton'):
            self.draw_skeleton = scriptOp.par.Drawskeleton.eval()
        if hasattr(scriptOp.par, 'Drawtext'):
            self.draw_text = scriptOp.par.Drawtext.eval()
        if hasattr(scriptOp.par, 'Drawscore'):
            self.draw_score = scriptOp.par.Drawscore.eval()
        if hasattr(scriptOp.par, 'Overlayonly'):
            self.overlay_only = scriptOp.par.Overlayonly.eval()
        if hasattr(scriptOp.par, 'Drawkeypointconf'):
            self.draw_keypoint_conf = scriptOp.par.Drawkeypointconf.eval()
        if hasattr(scriptOp.par, 'Debug'):
            self.debug = scriptOp.par.Debug.eval()
    
    def get_draw_info(self):
        """Compute draw info flags efficiently"""
        flags = 0
        if self.draw_text:
            flags |= DRAW_TEXT
        if self.draw_bbox:
            flags |= DRAW_BBOX
        if self.draw_score:
            flags |= DRAW_CONF
        if self.draw_skeleton:
            flags |= DRAW_SKELETON
        if self.overlay_only:
            flags |= OVERLAY_ONLY
        if self.draw_keypoint_conf:
            flags |= DRAW_KEYPOINT_CONF
        if self.mirror_labels:
            flags |= MIRROR_LABELS
        return flags


def load_config():
    """Load resolution configuration"""
    global _configured_resolution
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                _configured_resolution = (config['width'], config['height'])
                print(f"[OK] Loaded resolution config: {_configured_resolution[0]}x{_configured_resolution[1]}")
                return True
        except Exception as e:
            print(f"[WARNING] Error loading config: {e}")
    else:
        print(f"[WARNING] No config file found at {CONFIG_FILE}")
        print("   Run td_resolution_config.py first to set resolution!")
    
    return False


def save_resolution_info():
    """Save resolution for other nodes"""
    try:
        with open(RESOLUTION_INFO_FILE, 'w') as f:
            json.dump({'width': _configured_resolution[0], 'height': _configured_resolution[1]}, f)
    except:
        pass


def process_input_frame(raw_frame, params_cache):
    """Optimized input processing with minimal copies
    
    Returns (model_frame, original_frame):
    - model_frame: Frame in the orientation expected by YOLO model
    - original_frame: Original frame (WITHOUT mirror - mirror is applied after processing)
    
    The flip settings control how to transform the input for the model.
    Mirror is NOT applied here - it's applied after YOLO processing to preserve correct labels.
    """
    if raw_frame is None or raw_frame.size == 0:
        return None, None
    
    # Verify we have RGB (3 channels)
    if raw_frame.ndim == 2:
        # Grayscale to RGB
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2RGB)
    elif raw_frame.shape[2] == 3:
        # Already RGB - perfect
        frame = raw_frame
    else:
        print(f"[ERROR] Expected 3-channel RGB, got {raw_frame.shape[2]} channels")
        return None, None
    
    # Convert type if needed
    if frame.dtype == np.float32:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    # DO NOT apply mirror here - we'll do it after YOLO processing
    # This ensures labels are correct (left ear is labeled as left)
    
    # The flips transform input to model space
    # If your input needs to be flipped for the model to work correctly,
    # enable the flip settings
    flip_code = _flip_lookup.get((params_cache.flip_v, params_cache.flip_h), FLIP_NONE)
    if flip_code != FLIP_NONE:
        model_frame = cv2.flip(frame, flip_code)
    else:
        model_frame = frame
    
    # Return model frame for processing and original frame (unmirrored)
    return model_frame, frame


def onSetupParameters(scriptOp):
    """Setup with optimizations"""
    global _params_cache
    
    print("\n=== YOLO TouchDesigner Final Optimized ===")
    
    # Initialize params cache
    _params_cache = ParamsCache()
    
    # Load configuration
    if load_config():
        print(f"[INFO] Using resolution: {_configured_resolution[0]}x{_configured_resolution[1]}")
    else:
        print(f"[INFO] Using default: {_configured_resolution[0]}x{_configured_resolution[1]}")
    
    # Create controls
    page = scriptOp.appendCustomPage("YOLO")
    
    # Resolution info display
    p = page.appendStr('Resolutioninfo', label='Resolution')
    p.val = f'{_configured_resolution[0]}x{_configured_resolution[1]}'
    p.readOnly = True
    
    # Connection controls
    p = page.appendPulse("Connect", label="Connect to YOLO Server")
    p.help = f"Connect at {_configured_resolution[0]}x{_configured_resolution[1]}"
    
    p = page.appendPulse("Reload", label="Reload Config")
    p.help = "Reload resolution configuration"
    
    # Overlay mode at the top of display options
    p = page.appendToggle("Overlayonly", label="Overlay Only Mode")
    p.default = False
    p.val = False
    p.help = "Draw detections on black background (can cause accumulation)"
    
    # Input transformation parameters
    p = page.appendToggle("Flipv", label="Flip Vertical")
    p.default = True
    p.val = True
    p.help = "Flip vertically for YOLO (typically ON for webcam to fix coordinate system)"
    
    p = page.appendToggle("Fliph", label="Flip Horizontal")
    p.default = True
    p.val = True
    p.help = "Flip horizontally before YOLO (ON=webcam default, OFF=pre-flipped or normal video)"
    
    # Webcam-specific parameters (grouped together)
    p = page.appendStr("Webcamsection", label="--- Webcam Settings ---")
    p.val = "Defaults work for standard webcam"
    p.readOnly = True
    
    p = page.appendToggle("Mirror", label="Mirror Output")
    p.default = False
    p.val = False
    p.help = "Mirror the final output AFTER processing (webcam: right hand appears on right side, labels stay correct)"
    
    p = page.appendToggle("Mirrorlabels", label="Mirror Labels")
    p.default = True
    p.val = True
    p.help = "Swap left/right labels (ON when Flip Horizontal=ON for correct webcam labels)"
    
    # Detection parameters
    p = page.appendFloat("Nms", label="NMS/IOU Threshold")
    p.default = 0.5
    p.min = 0
    p.max = 1
    p.val = 0.5
    
    p = page.appendFloat("Score", label="Score Threshold")
    p.default = 0.5
    p.min = 0.1
    p.max = 1.0
    p.val = 0.5
    
    p = page.appendInt("Maxk", label="Max Detections")
    p.default = 1
    p.min = 0
    p.max = 20
    p.val = 1
    
    # Drawing options - all grouped together
    p = page.appendToggle("Drawbbox", label="Draw Bounding Boxes")
    p.default = True
    p.val = True
    
    p = page.appendToggle("Drawskeleton", label="Draw Pose Skeleton")
    p.default = True
    p.val = True
    
    p = page.appendToggle("Drawtext", label="Draw Labels")
    p.default = True
    p.val = True
    
    p = page.appendToggle("Drawscore", label="Draw Scores")
    p.default = True
    p.val = True
    
    p = page.appendToggle("Drawkeypointconf", label="Draw Keypoint Confidence")
    p.default = False
    p.val = False
    
    # Pose parameters
    p = page.appendFloat("Posethreshold", label="Pose Confidence")
    p.default = 0.1
    p.min = 0
    p.max = 1
    p.val = 0.1
    
    p = page.appendFloat("Smoothing", label="Keypoint Smoothing")
    p.default = 0.7
    p.min = 0
    p.max = 1
    p.val = 0.7
    
    p = page.appendInt("Frameskip", label="Frame Skip")
    p.default = 1
    p.min = 1
    p.max = 10
    p.val = 1
    p.help = "Process every Nth frame (1 = every frame)"
    
    # Debug mode at the end
    p = page.appendToggle("Debug", label="Debug Mode")
    p.default = False
    p.val = False
    p.help = "Enable debug logging"
    
    print("\n[OK] Click 'Connect to YOLO Server' to start")
    return


def onPulse(par):
    """Handle buttons"""
    if par.name == "Connect":
        initialize_connection()
    elif par.name == "Reload":
        if load_config():
            print(f"[OK] Reloaded configuration: {_configured_resolution[0]}x{_configured_resolution[1]}")
            # Update the resolution display
            try:
                scriptOp = op(me)
                if hasattr(scriptOp.par, 'Resolutioninfo'):
                    scriptOp.par.Resolutioninfo.val = f'{_configured_resolution[0]}x{_configured_resolution[1]}'
            except:
                pass
        else:
            print("[ERROR] Failed to reload configuration")


def initialize_connection():
    """Connect to YOLO server with optimizations"""
    global _shm_states, _sl_params, _shm_image, _shm_detections, _shm_pose, _shm_fps
    global _initialized, _shared_array, _array_views
    
    width, height = _configured_resolution
    print(f"\n[INFO] Connecting to YOLO server at {width}x{height}...")
    
    # Save for other nodes
    save_resolution_info()
    
    try:
        # Connect to shared memory
        _shm_states = shared_memory.SharedMemory(name="yolo_states")
        _sl_params = shared_memory.ShareableList(name="params")
        _shm_image = shared_memory.SharedMemory(name="image")
        _shm_detections = shared_memory.SharedMemory(name="detection_data")
        _shm_pose = shared_memory.SharedMemory(name="pose_data")
        
        # Try FPS
        try:
            _shm_fps = shared_memory.SharedMemory(name="fps_stats")
        except:
            _shm_fps = None
        
        # Check buffer size
        required = width * height * 3 * 4
        if _shm_image.size < required:
            print(f"[ERROR] Buffer too small! Need {required}, have {_shm_image.size}")
            print(f"   Restart server: python setup_all.py -w {width} -h {height}")
            cleanup()
            return
        
        # Create array view for configured resolution
        _shared_array = np.ndarray((height, width, 3), dtype=np.float32, buffer=_shm_image.buf)
        
        # Pre-allocate view for this resolution
        key = f"{width}x{height}"
        _array_views[key] = _shared_array
        
        print("[OK] Connected successfully!")
        _initialized = True
        
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        cleanup()


def cleanup():
    """Clean up connections"""
    global _shm_states, _sl_params, _shm_image, _shm_detections, _shm_pose, _shm_fps
    global _initialized, _shared_array, _array_views
    
    # Mark as not initialized first
    _initialized = False
    
    # Clear shared array reference first to avoid BufferError
    _shared_array = None
    _array_views.clear()
    
    # Give Python time to release references
    import gc
    gc.collect()
    
    # Close ShareableList first as it has its own shared memory
    if _sl_params:
        try:
            _sl_params.shm.close()
        except:
            pass
        _sl_params = None
    
    # Close shared memory segments
    for shm in [_shm_states, _shm_image, _shm_detections, _shm_pose, _shm_fps]:
        if shm:
            try:
                shm.close()
            except BufferError:
                # Ignore buffer errors during cleanup
                pass
            except:
                pass
    
    _shm_states = None
    _shm_image = None
    _shm_detections = None
    _shm_pose = None
    _shm_fps = None


def onCook(scriptOp):
    """Optimized cook function"""
    global _frame_count, _initialized, _shared_array, _shm_states, _sl_params, _shm_fps
    global _last_output, _last_output_time, _processing_frame, _td_frame_times, _last_td_frame_time, _total_td_frames
    global _params_cache
    
    width, height = _configured_resolution
    
    # Ensure params cache exists
    if _params_cache is None:
        _params_cache = ParamsCache()
    
    # Update params cache periodically
    _params_cache.update_if_needed(scriptOp, _frame_count)
    
    # Check overlay mode FIRST - this is critical
    is_overlay_mode = _params_cache.overlay_only
    
    # Track TD frame timing
    current_time = time.time()
    if _last_td_frame_time is not None:
        frame_time = current_time - _last_td_frame_time
        _td_frame_times.append(frame_time)
    _last_td_frame_time = current_time
    _total_td_frames += 1
    
    # Update TD FPS in shared memory if available
    if _shm_fps is not None and len(_td_frame_times) > 1:
        avg_frame_time = sum(_td_frame_times) / len(_td_frame_times)
        td_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        try:
            stats = np.frombuffer(_shm_fps.buf[:16], dtype=np.float32)
            if len(stats) >= 4:
                stats[3] = td_fps
        except:
            pass
    
    if not _initialized:
        # Black frame while not connected
        black = np.zeros((height, width, 3), dtype=np.uint8)
        scriptOp.copyNumpyArray(black)
        if _frame_count % 60 == 0:
            print(f"[INFO] Not connected ({width}x{height})")
        _frame_count += 1
        return
    
    # SAFETY CHECK: If in overlay mode and no last output, ensure we never show input
    if is_overlay_mode and _last_output is None:
        # Pre-initialize with black to ensure we never accidentally show input
        _last_output = np.zeros((height, width, 3), dtype=np.uint8)
        _last_output_time = time.time()
    
    # Frame rate limiting - DISABLED in overlay mode to prevent sync issues
    if is_overlay_mode:
        # In overlay mode, process every frame to maintain sync
        frame_skip = 1
        should_process_frame = True
    else:
        frame_skip = int(scriptOp.par.Frameskip.eval()) if hasattr(scriptOp.par, 'Frameskip') else 1
        # Ensure frame_skip is at least 1 to avoid division by zero
        frame_skip = max(1, frame_skip)
        # Determine if we should process this frame
        should_process_frame = (_frame_count % frame_skip == 0)
    
    # Check server status efficiently
    if _shm_states is None or not _initialized:
        return
    
    try:
        states_view = np.frombuffer(_shm_states.buf, dtype=np.uint8)
    except (ValueError, BufferError) as e:
        # Handle case where shared memory is no longer valid
        if _frame_count % 60 == 0:
            print(f"[WARNING] Shared memory access error: {e}")
        cleanup()
        return
    server_alive = states_view[2] if len(states_view) > 2 else 0
    
    if server_alive != 51:  # ASCII '3'
        if _frame_count % 60 == 0:
            print(f"Server not ready (alive={server_alive})")
        _frame_count += 1
        return
    
    server_state = states_view[0]
    client_state = states_view[1]
    
    # Process results if available
    new_result = None
    if client_state == 50 and _shared_array is not None:  # ASCII '2' - result ready
        # Read result from shared memory
        try:
            result = _shared_array.copy()
            
            # Validate data before using
            if result.shape == (height, width, 3) and np.isfinite(result).all():
                # Convert to uint8 efficiently
                if result.max() > 1.0:
                    result_uint8 = result.astype(np.uint8)
                else:
                    result_uint8 = (result * 255).astype(np.uint8)
                
                # The result has drawings in model space
                # We need to flip it back to match the original frame orientation
                if _params_cache:
                    flip_code = _flip_lookup.get((_params_cache.flip_v, _params_cache.flip_h), FLIP_NONE)
                    if flip_code != FLIP_NONE:
                        # Flip back to original orientation
                        new_result = cv2.flip(result_uint8, flip_code)
                    else:
                        new_result = result_uint8
                else:
                    new_result = result_uint8
                
                # Clear client state only if we successfully got the result
                states_view[1] = 48  # ASCII '0'
                _processing_frame = False
                
                if _params_cache and _params_cache.debug and _frame_count % 30 == 0:
                    print(f"[DEBUG] Received valid result at frame {_frame_count}")
            else:
                # Invalid data - don't update, keep last good frame
                if _params_cache and _params_cache.debug:
                    print(f"[WARNING] Invalid result data at frame {_frame_count}, keeping last good frame")
        except Exception as e:
            # Error reading - keep last good frame
            if _params_cache and _params_cache.debug:
                print(f"[WARNING] Error reading result at frame {_frame_count}: {e}")
    
    # Get input frame efficiently
    model_frame = None
    original_frame = None
    
    if scriptOp.inputs and len(scriptOp.inputs) > 0:
        raw_frame = scriptOp.inputs[0].numpyArray(delayed=True, writable=False)
        
        if raw_frame is not None and raw_frame.size > 0:
            # Convert RGBA to RGB if needed (TouchDesigner often outputs RGBA)
            if raw_frame.ndim == 3 and raw_frame.shape[2] == 4:
                raw_frame = raw_frame[:, :, :3]  # Drop alpha channel
            
            # Check if input matches configured resolution
            input_h, input_w = raw_frame.shape[:2]
            if (input_w, input_h) != _configured_resolution:
                if _frame_count % 60 == 0:
                    print(f"[WARNING] Input {input_w}x{input_h} != configured {width}x{height}, resizing...")
                raw_frame = cv2.resize(raw_frame, _configured_resolution)
            
            # Process input efficiently
            if _params_cache:
                model_frame, original_frame = process_input_frame(raw_frame, _params_cache)
            else:
                model_frame = original_frame = raw_frame
    
    # Send new frame if server is ready, we have input, and it's time to process
    if model_frame is not None and server_state == 48 and should_process_frame:
        # In overlay mode, only send new frame if we don't have a pending result
        if is_overlay_mode and _processing_frame:
            # Skip sending new frame - wait for current one to complete
            if _frame_count % 60 == 0:
                print(f"[INFO] Overlay mode: Waiting for backend to complete frame")
        else:
            # Log if we're dropping frames (backend still processing)
            if _processing_frame and _frame_count % 30 == 0:
                print(f"[WARNING] Backend still processing, may drop frames (frame {_frame_count})")
            
            # Update parameters efficiently
            if _sl_params and _params_cache:
                try:
                    # Batch update parameters - must match ParamsIndex order
                    # Only update the numeric parameters we can change
                    _sl_params[0] = float(_params_cache.nms)  # IOU_THRESH
                    _sl_params[1] = float(_params_cache.score)  # SCORE_THRESH
                    _sl_params[2] = int(_params_cache.maxk)  # TOP_K
                    # Skip 3 (ETA), 4 (WIDTH), 5 (HEIGHT), 6 (CHANNELS)
                    # Skip 7-9 (string parameters)
                    _sl_params[10] = _params_cache.get_draw_info()  # DRAW_INFO
                    _sl_params[11] = float(_params_cache.pose_threshold)  # POSE_THRESHOLD
                    # Skip 12 (FPS_LIMIT) - leave as configured
                except Exception as e:
                    if _frame_count % 60 == 0:
                        print(f"Error updating params: {e}")
            
            # Convert to float32 and copy to shared memory
            if model_frame.dtype == np.uint8:
                image_float = model_frame.astype(np.float32) / 255.0
            elif model_frame.dtype == np.float32:
                # Check if already normalized
                if model_frame.max() > 1.0:
                    image_float = model_frame / 255.0
                else:
                    image_float = model_frame
            else:
                image_float = model_frame.astype(np.float32) / 255.0
            
            # Debug shape
            if _params_cache and _params_cache.debug and _frame_count % 60 == 0:
                print(f"[DEBUG] image_float shape: {image_float.shape}, model_frame shape: {model_frame.shape}")
            
            # TouchDesigner should provide 3 channels - verify
            if image_float.shape[2] != 3:
                print(f"[ERROR] Expected 3 channels, got {image_float.shape[2]} at frame {_frame_count}")
                return
            
            if _shared_array is not None:
                _shared_array[:] = image_float
            
            # Signal new frame
            states_view[0] = 49  # ASCII '1'
            _processing_frame = True
            
            if _params_cache and _params_cache.debug and _frame_count % 30 == 0:
                print(f"[DEBUG] Sent frame {_frame_count} to backend")
    
    # Determine output - CRITICAL: Check overlay mode with is_overlay_mode variable
    if is_overlay_mode:
        # OVERLAY MODE: NEVER show input video, only detections on black
        # ALWAYS prefer last known good output over anything else
        
        if new_result is not None:
            # New detection result available - update our cache
            _last_output = new_result.copy()  # Make a copy to ensure it persists
            _last_output_time = time.time()
            output_frame = new_result
            if _params_cache.debug and _frame_count % 60 == 0:
                print(f"[DEBUG] Overlay mode: showing new result at frame {_frame_count}")
        elif _last_output is not None:
            # Keep showing last detection result (no timeout in overlay mode)
            # This is the KEY - we ALWAYS show last good output rather than going black
            output_frame = _last_output
            if _params_cache.debug and _frame_count % 120 == 0:
                age = time.time() - _last_output_time if _last_output_time else 0
                print(f"[DEBUG] Overlay mode: reusing last output (age={age:.1f}s)")
        else:
            # No detections yet - show black only as absolute last resort
            output_frame = np.zeros((height, width, 3), dtype=np.uint8)
            if _params_cache.debug:
                print(f"[DEBUG] Overlay mode: no results yet, showing black at frame {_frame_count}")
    else:
        # NORMAL MODE: Can show input video
        if new_result is not None:
            # New result available
            _last_output = new_result
            _last_output_time = time.time()
            output_frame = new_result
        elif _last_output is not None and _last_output_time is not None:
            # Check if last output is too old
            if time.time() - _last_output_time < 5.0:
                # Use last good result if not too old
                output_frame = _last_output
            elif original_frame is not None:
                # Fall back to input
                output_frame = original_frame
            else:
                # Black frame as last resort
                output_frame = np.zeros((height, width, 3), dtype=np.uint8)
        elif original_frame is not None:
            # No processed frames yet - show input
            output_frame = original_frame
        else:
            # Black frame as last resort
            output_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # FINAL SAFETY CHECK: Ensure we have a valid output frame
    if 'output_frame' not in locals() or output_frame is None:
        print(f"[ERROR] No output frame defined at frame {_frame_count}! Using black.")
        output_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # OVERLAY MODE FINAL CHECK: Never output input video
    if is_overlay_mode:
        # Extra safety: ensure we're not accidentally outputting the input
        if original_frame is not None and output_frame is not None:
            # Check if output looks like input (simple heuristic - similar mean values)
            try:
                # Ensure arrays are in the right format for mean calculation
                if isinstance(output_frame, np.ndarray):
                    output_mean = float(output_frame.mean())
                else:
                    output_mean = float(np.asarray(output_frame).mean())
                
                if isinstance(original_frame, np.ndarray):
                    original_mean = float(original_frame.mean())
                else:
                    original_mean = float(np.asarray(original_frame).mean())
                
                if abs(output_mean - original_mean) < 5:
                    print(f"[ERROR] Overlay mode output too similar to input at {_frame_count}! Using last good or black.")
                    if _last_output is not None:
                        output_frame = _last_output
                    else:
                        output_frame = np.zeros((height, width, 3), dtype=np.uint8)
            except Exception:
                # If comparison fails, just continue
                pass
    
    # Apply mirror AFTER processing if requested (for natural webcam view)
    # This ensures labels are correct but display is mirrored
    if _params_cache and _params_cache.mirror and output_frame is not None:
        output_frame = cv2.flip(output_frame, 1)  # Horizontal flip
    
    # Output - ensure contiguous memory for TouchDesigner
    if not output_frame.flags['C_CONTIGUOUS']:
        output_frame = np.ascontiguousarray(output_frame)
    scriptOp.copyNumpyArray(output_frame)
    _frame_count += 1


def onDestroy():
    """Cleanup on destroy"""
    cleanup()
    print("Cleaned up")