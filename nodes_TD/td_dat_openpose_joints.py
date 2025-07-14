"""
TouchDesigner Script DAT for OpenPose joint output
Outputs pose data with specific joint names matching your format
"""
import numpy as np
from multiprocessing import shared_memory
import struct
import json
import os

# Configuration files
RESOLUTION_INFO_FILE = "/tmp/yolo_resolution.json"

# Global state
_shm_pose = None
_initialized = False
_frame_count = 0
_last_valid_joints = None  # Store last valid joint positions for smoothing
_configured_resolution = (640, 640)  # Default, will be loaded from config

# Your specific joint names in order
JOINT_NAMES = [
    "lAnkle",
    "lElbow", 
    "lHeadInner",
    "lHeadOuter",
    "lHip",
    "lKnee",
    "lShoulder",
    "lWrist",
    "nose",
    "rAnkle",
    "rElbow",
    "rHeadInner",
    "rHeadOuter",
    "rHip",
    "rKnee",
    "rShoulder",
    "rWrist",
    "upperChest"
]

def coco_to_custom_joints(coco_keypoints):
    """
    Convert 17 COCO keypoints to your custom joint format
    
    COCO format (17 keypoints):
    0: nose
    1: left_eye
    2: right_eye
    3: left_ear
    4: right_ear
    5: left_shoulder
    6: right_shoulder
    7: left_elbow
    8: right_elbow
    9: left_wrist
    10: right_wrist
    11: left_hip
    12: right_hip
    13: left_knee
    14: right_knee
    15: left_ankle
    16: right_ankle
    
    Returns dictionary with joint_name: (x, y) pairs
    """
    joints = {}
    
    # Direct mappings
    joints["nose"] = (coco_keypoints[0, 0], coco_keypoints[0, 1])
    
    # Left side
    joints["lAnkle"] = (coco_keypoints[15, 0], coco_keypoints[15, 1])
    joints["lElbow"] = (coco_keypoints[7, 0], coco_keypoints[7, 1])
    joints["lHeadInner"] = (coco_keypoints[1, 0], coco_keypoints[1, 1])  # left_eye
    joints["lHeadOuter"] = (coco_keypoints[3, 0], coco_keypoints[3, 1])  # left_ear
    joints["lHip"] = (coco_keypoints[11, 0], coco_keypoints[11, 1])
    joints["lKnee"] = (coco_keypoints[13, 0], coco_keypoints[13, 1])
    joints["lShoulder"] = (coco_keypoints[5, 0], coco_keypoints[5, 1])
    joints["lWrist"] = (coco_keypoints[9, 0], coco_keypoints[9, 1])
    
    # Right side
    joints["rAnkle"] = (coco_keypoints[16, 0], coco_keypoints[16, 1])
    joints["rElbow"] = (coco_keypoints[8, 0], coco_keypoints[8, 1])
    joints["rHeadInner"] = (coco_keypoints[2, 0], coco_keypoints[2, 1])  # right_eye
    joints["rHeadOuter"] = (coco_keypoints[4, 0], coco_keypoints[4, 1])  # right_ear
    joints["rHip"] = (coco_keypoints[12, 0], coco_keypoints[12, 1])
    joints["rKnee"] = (coco_keypoints[14, 0], coco_keypoints[14, 1])
    joints["rShoulder"] = (coco_keypoints[6, 0], coco_keypoints[6, 1])
    joints["rWrist"] = (coco_keypoints[10, 0], coco_keypoints[10, 1])
    
    # upperChest = average of shoulders
    if coco_keypoints[5, 2] > 0 and coco_keypoints[6, 2] > 0:
        joints["upperChest"] = (
            (coco_keypoints[5, 0] + coco_keypoints[6, 0]) / 2,
            (coco_keypoints[5, 1] + coco_keypoints[6, 1]) / 2
        )
    else:
        joints["upperChest"] = (0, 0)
    
    return joints


def load_resolution_info():
    """Load resolution from the shared config file"""
    global _configured_resolution
    try:
        if os.path.exists(RESOLUTION_INFO_FILE):
            with open(RESOLUTION_INFO_FILE, 'r') as f:
                data = json.load(f)
                _configured_resolution = (data['width'], data['height'])
                return _configured_resolution
    except Exception as e:
        print(f"Error loading resolution config: {e}")
    return _configured_resolution


def onSetupParameters(scriptOp):
    """Setup parameters"""
    # IMPORTANT: To make this DAT update every frame, you have 3 options:
    #
    # OPTION 1 (Recommended): Use Execute DAT
    # 1. Create an Execute DAT
    # 2. Turn on "Frame Start" in the Execute DAT
    # 3. In its onFrameStart callback, add: op('name_of_this_dat').cook(force=True)
    #
    # OPTION 2: Use Info CHOP
    # 1. Create an Info CHOP and set it to "Perform Monitors" 
    # 2. Connect its output to a Null CHOP
    # 3. Reference the Null CHOP in this DAT's parameters (creates dependency)
    #
    # OPTION 3: Use Timer CHOP
    # 1. Create a Timer CHOP set to your desired FPS
    # 2. Connect to a CHOP Execute DAT
    # 3. In onValueChange callback: op('name_of_this_dat').cook(force=True)
    
    page = scriptOp.appendCustomPage('Joint Output')
    
    # Load resolution configuration
    width, height = load_resolution_info()
    
    # Display resolution info
    p = page.appendStr('Resolutioninfo', label='Resolution')
    p.val = f'{width}x{height}'
    p.readOnly = True
    
    p = page.appendPulse('Connect', label='Connect to Pose Data')
    p = page.appendPulse('Reloadconfig', label='Reload Resolution Config')
    p = page.appendToggle('Active', label='Active')
    p.default = True
    p = page.appendInt('Personindex', label='Person Index')
    p.default = 0
    p.min = 0
    p.max = 10
    
    # Add coordinate transformation parameters
    p = page.appendToggle('Normalize', label='Normalize Coordinates')
    p.default = False
    p.help = 'Normalize coordinates to 0-1 range'
    
    p = page.appendToggle('Centercoords', label='Center Coordinates')
    p.default = False
    p.help = 'Transform coordinates to be centered (0,0 at center)'
    
    p = page.appendToggle('Tdcoords', label='TouchDesigner Coords')
    p.default = True
    p.val = True
    p.help = 'Use TouchDesigner coordinate system (pixel coords for OpenPose renderer)'
    
    p = page.appendFloat('Scale', label='Scale')
    p.default = 1.0
    p.min = 0.1
    p.max = 10.0
    p.help = 'Scale factor for coordinates'
    
    p = page.appendFloat('Offsetx', label='X Offset') 
    p.default = 0.0
    p.min = -2000
    p.max = 2000
    p.help = 'Additional X offset after transformation (e.g., -960 for 640x640, which is -1.5x width)'
    
    p = page.appendFloat('Offsety', label='Y Offset')
    p.default = 0.0
    p.min = -1000
    p.max = 1000
    p.help = 'Additional Y offset after transformation'
    
    p = page.appendToggle('Autocenter', label='Auto Center')
    p.default = True
    p.val = True
    p.help = 'Automatically calculate offset as -1.5x content width (works for standard OpenPose renderer)'
    
    # Add dummy parameter to create dependency for Option 2
    p = page.appendCHOP('Trigger', label='Trigger CHOP (Option 2)')
    p.help = 'Connect a constantly changing CHOP here to trigger updates'
    
    return


def onPulse(par):
    """Handle button presses"""
    global _shm_pose, _initialized
    
    if par.name == 'Connect':
        try:
            _shm_pose = shared_memory.SharedMemory(name="pose_data")
            _initialized = True
            print("Connected to pose data memory")
        except:
            print("Pose data memory not found - is the YOLO pose server running?")
            _initialized = False
    elif par.name == 'Reloadconfig':
        width, height = load_resolution_info()
        print(f"Reloaded resolution config: {width}x{height}")
        # Update the display
        if hasattr(op(me).par, 'Resolutioninfo'):
            op(me).par.Resolutioninfo.val = f'{width}x{height}'
    return


def onCook(scriptOp):
    """Main processing - outputs joint table"""
    global _shm_pose, _initialized, _frame_count, _last_valid_joints
    _frame_count += 1
    
    # Check if active
    try:
        if hasattr(scriptOp.par, 'Active') and scriptOp.par.Active is not None:
            if not scriptOp.par.Active.eval():
                scriptOp.clear()
                return
    except:
        pass
    
    if not _initialized:
        # Try to connect automatically
        try:
            _shm_pose = shared_memory.SharedMemory(name="pose_data")
            _initialized = True
        except:
            pass
    
    if not _initialized or _shm_pose is None:
        # Output table with zeros
        scriptOp.clear()
        scriptOp.appendRow(['Joint Name', 'X', 'Y'])
        for joint_name in JOINT_NAMES:
            scriptOp.appendRow([joint_name, '0', '0'])
        return
    
    # Clear existing data
    scriptOp.clear()
    
    # Add header
    scriptOp.appendRow(['Joint Name', 'X', 'Y'])
    
    # Read pose data
    try:
        # Read number of persons
        num_persons_bytes = _shm_pose.buf[:4]
        num_persons = struct.unpack('i', num_persons_bytes)[0]
        
        if num_persons <= 0:
            # Output zeros for all joints
            for joint_name in JOINT_NAMES:
                scriptOp.appendRow([joint_name, '0', '0'])
            return
        
        # Get person index
        person_idx = 0
        try:
            if hasattr(scriptOp.par, 'Personindex') and scriptOp.par.Personindex is not None:
                person_idx = int(scriptOp.par.Personindex.eval())
        except:
            person_idx = 0
        
        # Clamp to valid range
        person_idx = max(0, min(person_idx, num_persons - 1))
        
        # Read person data
        offset = 4 + person_idx * 56 * 4
        person_data = struct.unpack('56f', _shm_pose.buf[offset:offset + 56*4])
        
        # Extract COCO keypoints
        coco_keypoints = np.array(person_data[5:]).reshape(17, 3)
        
        # Validate keypoints - check if we have valid data
        valid_keypoints = 0
        for i in range(17):
            if coco_keypoints[i, 2] > 0.1:  # Check confidence
                # Also check if coordinates are reasonable
                if 0 <= coco_keypoints[i, 0] <= 2000 and 0 <= coco_keypoints[i, 1] <= 2000:
                    valid_keypoints += 1
        
        # If we have too few valid keypoints, use last valid data
        if valid_keypoints < 3:
            if _last_valid_joints is not None:
                # Use last valid joint positions
                for joint_name in JOINT_NAMES:
                    if joint_name in _last_valid_joints:
                        x, y = _last_valid_joints[joint_name]
                        scriptOp.appendRow([joint_name, f'{x:.4f}', f'{y:.4f}'])
                    else:
                        scriptOp.appendRow([joint_name, '0', '0'])
            else:
                # No previous data - output zeros
                for joint_name in JOINT_NAMES:
                    scriptOp.appendRow([joint_name, '0', '0'])
            return
        
        # Convert to custom joint format
        joints = coco_to_custom_joints(coco_keypoints)
        
        # Get resolution from config
        width, height = load_resolution_info()
        
        # Get transformation parameters
        normalize = False
        center_coords = True
        td_coords = False
        scale = 1.0
        offset_x = 0.0
        offset_y = 0.0
        auto_center = False
        
        try:
            if hasattr(scriptOp.par, 'Normalize') and scriptOp.par.Normalize is not None:
                normalize = scriptOp.par.Normalize.eval()
            if hasattr(scriptOp.par, 'Centercoords') and scriptOp.par.Centercoords is not None:
                center_coords = scriptOp.par.Centercoords.eval()
            if hasattr(scriptOp.par, 'Tdcoords') and scriptOp.par.Tdcoords is not None:
                td_coords = scriptOp.par.Tdcoords.eval()
            if hasattr(scriptOp.par, 'Scale') and scriptOp.par.Scale is not None:
                scale = scriptOp.par.Scale.eval()
            if hasattr(scriptOp.par, 'Offsetx') and scriptOp.par.Offsetx is not None:
                offset_x = scriptOp.par.Offsetx.eval()
            if hasattr(scriptOp.par, 'Offsety') and scriptOp.par.Offsety is not None:
                offset_y = scriptOp.par.Offsety.eval()
            if hasattr(scriptOp.par, 'Autocenter') and scriptOp.par.Autocenter is not None:
                auto_center = scriptOp.par.Autocenter.eval()
        except:
            pass
        
        # Auto-calculate offset if enabled
        if auto_center:
            # The OpenPose renderer seems to expect coordinates in a space that's 1.5x the content size
            # For 640x640 content, -960 offset works, which is exactly -1.5 * 640
            # This suggests the renderer has (0,0) at center of a 1.5x scaled coordinate space
            
            if td_coords:
                # For TD coords mode, offset by -1.5x the width
                # For 640x640: -1.5 * 640 = -960
                offset_x = -1.5 * width
                offset_y = 0  # Usually no Y offset needed
            elif center_coords:
                # For center coords mode, we already centered, so additional offset
                # might be different. Let's use -0.5x width as a starting point
                offset_x = -0.5 * width
                offset_y = 0
            
            if _frame_count % 60 == 0:
                print(f"[INFO] Auto-center: content={width}x{height}, calculated offset=({offset_x:.0f},{offset_y:.0f})")
        
        # Store transformed joints for smoothing
        output_joints = {}
        
        # Output joints in the specified order
        for joint_name in JOINT_NAMES:
            if joint_name in joints:
                x, y = joints[joint_name]
                
                # Debug: Check raw coordinate ranges on first joint
                if joint_name == "nose" and _frame_count % 30 == 0:
                    print(f"[DEBUG] Raw nose coords: x={x:.1f}, y={y:.1f} (resolution={width}x{height})")
                    # Also check a few other joints for debugging
                if joint_name in ["lShoulder", "rShoulder"] and _frame_count % 30 == 0:
                    print(f"[DEBUG] Raw {joint_name}: x={x:.1f}, y={y:.1f}")
                
                # Apply transformations
                # YOLO outputs coordinates in pixel space (0 to width/height)
                # TouchDesigner might expect different coordinate systems
                
                if td_coords:
                    # TouchDesigner/OpenPose coordinate system
                    # For 640x640 content, just use pixel coordinates with offset
                    # User reported -960 offset works for X, so apply offsets directly
                    x = x * scale + offset_x
                    y = y * scale + offset_y
                    
                    # Debug transformed coordinates
                    if joint_name == "nose" and _frame_count % 30 == 0:
                        print(f"[DEBUG] TD coords nose: raw=({joints[joint_name][0]:.1f},{joints[joint_name][1]:.1f}) final=({x:.1f},{y:.1f})")
                elif normalize:
                    # Normalize to 0-1 range
                    x = x / width
                    y = y / height
                elif center_coords:
                    # Center coordinates (0,0 at center instead of top-left)
                    # Transform to centered coordinates then apply offset
                    x_centered = (x - width/2) * scale
                    y_centered = (y - height/2) * scale
                    
                    x = x_centered + offset_x
                    y = y_centered + offset_y
                    
                    # Debug transformed coordinates
                    if joint_name == "nose" and _frame_count % 30 == 0:
                        print(f"[DEBUG] Centered nose: centered=({x_centered:.1f},{y_centered:.1f}) final=({x:.1f},{y:.1f})")
                else:
                    # Raw pixel coordinates with optional scale and offset
                    x = x * scale + offset_x
                    y = y * scale + offset_y
                
                output_joints[joint_name] = (x, y)
                scriptOp.appendRow([joint_name, f'{x:.4f}', f'{y:.4f}'])
            else:
                # Try to use last valid position for missing joints
                if _last_valid_joints and joint_name in _last_valid_joints:
                    x, y = _last_valid_joints[joint_name]
                    output_joints[joint_name] = (x, y)
                    scriptOp.appendRow([joint_name, f'{x:.4f}', f'{y:.4f}'])
                else:
                    scriptOp.appendRow([joint_name, '0', '0'])
        
        # Store valid joints for next frame
        _last_valid_joints = output_joints
        
    except Exception as e:
        scriptOp.clear()
        scriptOp.appendRow(['Joint Name', 'X', 'Y'])
        scriptOp.appendRow(['ERROR', str(e), ''])
    
    return


def onDestroy():
    """Cleanup when node is deleted"""
    global _shm_pose, _initialized, _last_frame_data
    
    # Mark as not initialized first
    _initialized = False
    
    # Clear any cached data
    _last_frame_data = None
    
    # Close shared memory
    if _shm_pose:
        try:
            _shm_pose.close()
        except:
            pass
        _shm_pose = None
    
    print("[INFO] OpenPose joints DAT cleaned up")


