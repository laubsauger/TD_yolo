"""
TouchDesigner CHOP for OpenPose-compatible output
Converts 17 COCO keypoints to 25 OpenPose keypoints with 3D coordinates

OpenPose format (25 keypoints):
0: Nose
1: Neck
2: RShoulder
3: RElbow
4: RWrist
5: LShoulder
6: LElbow
7: LWrist
8: MidHip
9: RHip
10: RKnee
11: RAnkle
12: LHip
13: LKnee
14: LAnkle
15: REye
16: LEye
17: REar
18: LEar
19: LBigToe
20: LSmallToe
21: LHeel
22: RBigToe
23: RSmallToe
24: RHeel

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
"""

import numpy as np
from multiprocessing import shared_memory
import struct
import json
import os

# Global state
_shm_pose = None
_last_frame_count = -1
_cache_data = {}
_initialized = False

# Read resolution from config
def get_resolution():
    config_file = "/tmp/yolo_resolution.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config['width'], config['height']
        except:
            pass
    return 1280, 720


def connect_to_pose_data():
    """Connect to pose data shared memory"""
    global _shm_pose, _initialized
    
    try:
        _shm_pose = shared_memory.SharedMemory(name="pose_data")
        _initialized = True
        print("Connected to pose data memory")
        return True
    except:
        print("Pose data memory not found - is the YOLO pose server running?")
        return False


def coco_to_openpose(coco_keypoints):
    """
    Convert 17 COCO keypoints to 25 OpenPose keypoints
    
    Args:
        coco_keypoints: numpy array of shape (17, 3) with x, y, confidence
        
    Returns:
        openpose_keypoints: numpy array of shape (25, 3) with x, y, z (z is 0 for 2D)
    """
    # Initialize OpenPose array with zeros
    openpose = np.zeros((25, 3))
    
    # Direct mappings from COCO to OpenPose
    # OpenPose 0: Nose = COCO 0
    openpose[0] = [coco_keypoints[0, 0], coco_keypoints[0, 1], 0]
    
    # OpenPose 1: Neck = average of shoulders (COCO 5, 6)
    neck_x = (coco_keypoints[5, 0] + coco_keypoints[6, 0]) / 2
    neck_y = (coco_keypoints[5, 1] + coco_keypoints[6, 1]) / 2
    neck_conf = min(coco_keypoints[5, 2], coco_keypoints[6, 2])
    openpose[1] = [neck_x, neck_y, 0] if neck_conf > 0 else [0, 0, 0]
    
    # OpenPose 2-7: Shoulders, Elbows, Wrists
    openpose[2] = [coco_keypoints[6, 0], coco_keypoints[6, 1], 0]  # RShoulder
    openpose[3] = [coco_keypoints[8, 0], coco_keypoints[8, 1], 0]  # RElbow
    openpose[4] = [coco_keypoints[10, 0], coco_keypoints[10, 1], 0]  # RWrist
    openpose[5] = [coco_keypoints[5, 0], coco_keypoints[5, 1], 0]  # LShoulder
    openpose[6] = [coco_keypoints[7, 0], coco_keypoints[7, 1], 0]  # LElbow
    openpose[7] = [coco_keypoints[9, 0], coco_keypoints[9, 1], 0]  # LWrist
    
    # OpenPose 8: MidHip = average of hips (COCO 11, 12)
    hip_x = (coco_keypoints[11, 0] + coco_keypoints[12, 0]) / 2
    hip_y = (coco_keypoints[11, 1] + coco_keypoints[12, 1]) / 2
    hip_conf = min(coco_keypoints[11, 2], coco_keypoints[12, 2])
    openpose[8] = [hip_x, hip_y, 0] if hip_conf > 0 else [0, 0, 0]
    
    # OpenPose 9-14: Hips, Knees, Ankles
    openpose[9] = [coco_keypoints[12, 0], coco_keypoints[12, 1], 0]  # RHip
    openpose[10] = [coco_keypoints[14, 0], coco_keypoints[14, 1], 0]  # RKnee
    openpose[11] = [coco_keypoints[16, 0], coco_keypoints[16, 1], 0]  # RAnkle
    openpose[12] = [coco_keypoints[11, 0], coco_keypoints[11, 1], 0]  # LHip
    openpose[13] = [coco_keypoints[13, 0], coco_keypoints[13, 1], 0]  # LKnee
    openpose[14] = [coco_keypoints[15, 0], coco_keypoints[15, 1], 0]  # LAnkle
    
    # OpenPose 15-18: Eyes and Ears
    openpose[15] = [coco_keypoints[2, 0], coco_keypoints[2, 1], 0]  # REye
    openpose[16] = [coco_keypoints[1, 0], coco_keypoints[1, 1], 0]  # LEye
    openpose[17] = [coco_keypoints[4, 0], coco_keypoints[4, 1], 0]  # REar
    openpose[18] = [coco_keypoints[3, 0], coco_keypoints[3, 1], 0]  # LEar
    
    # OpenPose 19-24: Foot keypoints (not in COCO, estimate from ankles)
    # For feet, we'll place them slightly below the ankles
    foot_offset = 20  # pixels below ankle
    
    # Left foot (19: BigToe, 20: SmallToe, 21: Heel)
    if coco_keypoints[15, 2] > 0:  # If left ankle visible
        ankle_x = coco_keypoints[15, 0]
        ankle_y = coco_keypoints[15, 1]
        openpose[19] = [ankle_x - 10, ankle_y + foot_offset, 0]  # LBigToe
        openpose[20] = [ankle_x + 10, ankle_y + foot_offset, 0]  # LSmallToe
        openpose[21] = [ankle_x, ankle_y + foot_offset - 5, 0]  # LHeel
    
    # Right foot (22: BigToe, 23: SmallToe, 24: Heel)
    if coco_keypoints[16, 2] > 0:  # If right ankle visible
        ankle_x = coco_keypoints[16, 0]
        ankle_y = coco_keypoints[16, 1]
        openpose[22] = [ankle_x + 10, ankle_y + foot_offset, 0]  # RBigToe
        openpose[23] = [ankle_x - 10, ankle_y + foot_offset, 0]  # RSmallToe
        openpose[24] = [ankle_x, ankle_y + foot_offset - 5, 0]  # RHeel
    
    return openpose


def onSetupParameters(scriptOp):
    """Setup parameters for the CHOP"""
    page = scriptOp.appendCustomPage('OpenPose')
    
    # Display resolution info
    width, height = get_resolution()
    p = page.appendStr('Resolutioninfo', label='Resolution')
    p.val = f'{width}x{height}'
    p.readOnly = True
    
    p = page.appendPulse('Connect', label='Connect to Pose Data')
    p.help = 'Connect to YOLO pose detection shared memory'
    
    p = page.appendPulse('Reloadconfig', label='Reload Resolution Config')
    p.help = 'Reload resolution configuration'
    
    p = page.appendToggle('Normalize', label='Normalize Coordinates')
    p.default = False
    p.help = 'Normalize coordinates to [0,1] range'
    
    p = page.appendToggle('Jsonformat', label='JSON Format Output')
    p.default = False
    p.help = 'Output as OpenPose JSON format string'
    
    p = page.appendToggle('Active', label='Active')
    p.default = True
    p.help = 'Enable/disable automatic updates'
    
    return


def onPulse(par):
    """Handle button presses"""
    if par.name == 'Connect':
        connect_to_pose_data()
    elif par.name == 'Reloadconfig':
        width, height = get_resolution()
        print(f"[OK] Reloaded resolution config: {width}x{height}")
        # Update display
        try:
            scriptOp = op(me)
            if hasattr(scriptOp.par, 'Resolutioninfo'):
                scriptOp.par.Resolutioninfo.val = f'{width}x{height}'
        except:
            pass
    return


def onCook(scriptOp):
    """Main processing function"""
    global _last_frame_count, _cache_data, _initialized
    
    # Check if active
    if hasattr(scriptOp.par, 'Active') and not scriptOp.par.Active.eval():
        return
    
    if not _initialized:
        # Try to connect automatically
        connect_to_pose_data()
    
    if not _initialized or _shm_pose is None:
        # Output empty channels if not connected
        scriptOp.clear()
        return
    
    # Always clear and recreate channels to avoid duplication
    scriptOp.clear()
    
    # Read pose data
    try:
        # Read number of persons (first 4 bytes)
        num_persons_bytes = _shm_pose.buf[:4]
        num_persons = struct.unpack('i', num_persons_bytes)[0]
        
        if num_persons <= 0:
            scriptOp.clear()
            return
        
        # Get resolution
        width, height = get_resolution()
        normalize = scriptOp.par.Normalize.eval() if hasattr(scriptOp.par, 'Normalize') else False
        json_format = scriptOp.par.Jsonformat.eval() if hasattr(scriptOp.par, 'Jsonformat') else False
        
        # For now, output first person only (can be extended for multiple)
        if num_persons > 0:
            # Each person: 4 (bbox) + 1 (score) + 51 (17 keypoints * 3) = 56 floats
            offset = 4  # Skip num_persons
            person_data = struct.unpack('56f', _shm_pose.buf[offset:offset + 56*4])
            
            # Extract COCO keypoints (skip bbox and score)
            coco_keypoints = np.array(person_data[5:]).reshape(17, 3)
            
            # Convert to OpenPose format
            openpose_keypoints = coco_to_openpose(coco_keypoints)
            
            if json_format:
                # Create OpenPose JSON format
                json_data = {
                    "version": 1.3,
                    "people": [{
                        "person_id": [-1],
                        "pose_keypoints_2d": openpose_keypoints.flatten().tolist(),
                        "face_keypoints_2d": [],
                        "hand_left_keypoints_2d": [],
                        "hand_right_keypoints_2d": [],
                        "pose_keypoints_3d": [],
                        "face_keypoints_3d": [],
                        "hand_left_keypoints_3d": [],
                        "hand_right_keypoints_3d": []
                    }]
                }
                # Output as single string channel
                json_str = json.dumps(json_data)
                scriptOp.appendChan('openpose_json').vals = [ord(c) for c in json_str[:1000]]  # Limit length
            else:
                # Output as individual channels
                keypoint_names = [
                    'Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
                    'LShoulder', 'LElbow', 'LWrist', 'MidHip', 'RHip',
                    'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
                    'REye', 'LEye', 'REar', 'LEar', 'LBigToe',
                    'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel'
                ]
                
                for i, name in enumerate(keypoint_names):
                    x, y, z = openpose_keypoints[i]
                    
                    if normalize and (x > 0 or y > 0):
                        x = x / width
                        y = y / height
                    
                    # Create channels
                    chan_x = scriptOp.appendChan(f'{name}_x')
                    chan_y = scriptOp.appendChan(f'{name}_y')
                    chan_z = scriptOp.appendChan(f'{name}_z')
                    
                    chan_x.vals = [x]
                    chan_y.vals = [y]
                    chan_z.vals = [z]
            
    except Exception as e:
        print(f"Error reading pose data: {e}")
        scriptOp.clear()
    
    return


def onDestroy():
    """Cleanup when node is deleted"""
    global _shm_pose, _initialized
    if _shm_pose:
        try:
            _shm_pose.close()
        except:
            pass
        _shm_pose = None
    _initialized = False