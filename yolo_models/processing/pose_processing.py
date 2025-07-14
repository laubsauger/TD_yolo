from typing import Optional, Dict, Tuple, Union
import struct
from multiprocessing import shared_memory
from multiprocessing.shared_memory import ShareableList

import numpy as np
import torch
import cv2

from .info import ParamsIndex, DrawInfo
from .base_processing import BaseProcessServer
from ..detection import PyTorchYoloDetector


# COCO Pose keypoint names
COCO_KEYPOINTS = [
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

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    # Head
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    # Torso
    (5, 6),   # shoulders
    (5, 11), (6, 12),  # shoulders to hips
    (11, 12),  # hips
    # Arms
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    # Legs
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]


class YoloPoseDetectorProcesing(BaseProcessServer):
    def __init__(self,
                 update_shared_mem_name: str,
                 params_shared_mem_name: str,
                 array_shared_mem_name: str,
                 path_to_model: str,
                 image_width: int,
                 image_height: int,
                 num_channels: int,
                 image_dtype,
                 device: Optional[str] = None,
                 half_precision: bool = False,
                 trace: bool = False,
                 class_to_color_json: Optional[str] = None):  # pylint: disable=unused-argument
        super().__init__(update_shared_mem_name,
                         params_shared_mem_name,
                         array_shared_mem_name,
                         image_width,
                         image_height,
                         num_channels,
                         image_dtype)

        # Check for Apple Silicon (MPS) first
        if device is None:
            # Check if MPS backend exists and is available
            if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available") and torch.backends.mps.is_available():  # type: ignore
                device_name = "mps"
            elif torch.cuda.is_available():
                device_name = "cuda"
            else:
                device_name = "cpu"
                self._logging.warning("GPU is not available. Using CPU. This will be slow.")
        else:
            device_name = device

        # MPS doesn't support half precision
        if device_name == "mps" and half_precision:
            self._logging.warning("Half precision not supported on MPS, using full precision")
            half_precision = False

        torch_device = torch.device(device_name)
        self._detector = PyTorchYoloDetector(path_to_model, torch_device, half_precision, trace)

        # Get model input size
        self._model_input_size = self._detector.max_image_size
        self._logging.info(f"🏃 Pose model loaded: {path_to_model} | 🖥️ Device: {device_name} | 📐 Input: {self._model_input_size}x{self._model_input_size}")

        # Create pose data buffer name
        self._pose_shared_mem_name = "pose_data"
        self._sh_mem_pose = None
        self._pose_buffer_size = 1024 * 32  # 32KB for pose data (more than detection)
        self._last_valid_pose_data = None  # Store last valid pose data

        # Color coding for different body parts
        self._keypoint_colors = {
            # Head - Blue shades
            'nose': (255, 200, 0),
            'left_eye': (255, 150, 0),
            'right_eye': (255, 150, 0),
            'left_ear': (255, 100, 0),
            'right_ear': (255, 100, 0),
            
            # Arms - Green shades (left) and Cyan shades (right)
            'left_shoulder': (0, 255, 100),
            'left_elbow': (0, 200, 100),
            'left_wrist': (0, 150, 100),
            'right_shoulder': (255, 255, 0),
            'right_elbow': (200, 200, 0),
            'right_wrist': (150, 150, 0),
            
            # Torso/Legs - Red shades (left) and Magenta shades (right)
            'left_hip': (100, 0, 255),
            'left_knee': (150, 0, 200),
            'left_ankle': (200, 0, 150),
            'right_hip': (255, 0, 255),
            'right_knee': (200, 0, 200),
            'right_ankle': (150, 0, 150),
        }

        self._skeleton_color = (255, 255, 255)  # White for skeleton
        self._bbox_color = (255, 0, 0)  # Red for bbox

        # Frame counter for reduced logging
        self._log_frame_count = 0
        self._log_frequency = 30  # Log every N frames

        # Last good frame storage
        self._last_good_image = None
        self._last_good_det_info = None
        self._last_good_keypoints = None

        # Smoothing for keypoints (exponential moving average)
        self._smoothed_keypoints = {}  # person_id -> smoothed keypoints
        self._smoothing_alpha = 0.3  # Lower = more smoothing, higher = more responsive

    def init_mem(self):
        """Override to add pose data buffer"""
        super().init_mem()

        # Create pose data buffer
        try:
            self._sh_mem_pose = shared_memory.SharedMemory(
                name=self._pose_shared_mem_name, create=False, size=self._pose_buffer_size)
        except:
            try:
                self._sh_mem_pose = shared_memory.SharedMemory(
                    name=self._pose_shared_mem_name, create=True, size=self._pose_buffer_size)
            except:
                self._logging.warning("Could not create pose memory - pose data won't be available")
                self._sh_mem_pose = None
        
        if self._sh_mem_pose:
            self._logging.info(f"Pose buffer connected: {self._pose_buffer_size // 1024}KB")

    def dispose(self):
        """Override to clean up pose buffer"""
        if self._sh_mem_pose is not None:
            self._sh_mem_pose.close()
            if self._created_shared_memory:
                self._sh_mem_pose.unlink()
        super().dispose()

    def process(self, image: np.ndarray, params: Union[list, ShareableList]):
        # Store original image for composite mode
        original_image = image.copy()

        # Initialize variables
        det_info = None
        keypoints_data = None
        is_duplicate = False

        # Check if this is a duplicate/dropped frame by comparing with last
        if self._last_good_image is not None:
            # Simple check: if the image is very similar to last, it might be a duplicate
            diff = np.abs(image.astype(float) - self._last_good_image.astype(float)).mean()
            if diff < 0.001:  # Very small difference
                # Detected duplicate frame, using last good data
                # Use last good data instead of reprocessing
                if self._last_good_det_info is not None:
                    det_info = self._last_good_det_info
                    keypoints_data = self._last_good_keypoints
                    is_duplicate = True
                else:
                    # No cached data available, return original
                    return image

        # Get dimensions 
        h, w = image.shape[:2]
        
        # Reduced frequency logging
        self._log_frame_count += 1
        
        # Only run detection if not a duplicate frame
        if not is_duplicate:
            # Get detection info first (bounding boxes)
            try:
                det_info = self._detector.predict(
                    image,
                    score_threshold=params[ParamsIndex.SCORE_THRESH],
                    nms_threshold=params[ParamsIndex.IOU_THRESH],
                    max_k=params[ParamsIndex.TOP_K],
                    eta=params[ParamsIndex.ETA])
            except Exception as e:
                self._logging.error(f"Detection failed: {e}")
                # Use last good data if available
                if self._last_good_det_info is not None:
                    det_info = self._last_good_det_info
                    keypoints_data = self._last_good_keypoints
                    is_duplicate = True
                else:
                    # No cached data, return original
                    return image

            # Store as last good image
            self._last_good_image = image.copy()
            
            if self._log_frame_count % self._log_frequency == 0:
                self._logging.info(f"🏃 Pose: {w}x{h} | 👥 {len(det_info.scores)} persons detected")

            # For keypoints, we need to get the raw model output before NMS
            # This is a bit of a hack but works with the current architecture
            keypoints_data = None

            try:
                # Get the last raw prediction from the detector
                if hasattr(self._detector, '_last_raw_output') and self._detector._last_raw_output is not None:
                    raw_output = self._detector._last_raw_output

                    # Check if this is pose model output
                    if isinstance(raw_output, torch.Tensor):
                        raw_np = raw_output.cpu().numpy() if raw_output.is_cuda or raw_output.device.type == 'mps' else raw_output.numpy()
                    else:
                        raw_np = raw_output

                    # YOLOv8 pose format can be:
                    # [batch, 56, num_predictions] OR [batch, num_predictions, 56]
                    # where 56 = 5 (x,y,w,h,conf) + 51 (17 keypoints * 3)
                    if len(raw_np.shape) == 3 and (raw_np.shape[1] == 56 or raw_np.shape[2] == 56):
                        # Transpose if needed
                        if raw_np.shape[1] == 56:
                            raw_np = raw_np.transpose(0, 2, 1)  # Now [batch, num_predictions, 56]
                        # We need to match post-NMS detections to raw predictions
                        # This is approximate - we match by bbox overlap
                        num_detections = len(det_info.scores)
                        keypoints_list = []

                        # Get raw bboxes and confidences
                        raw_bboxes = raw_np[0, :, :4]  # [8400, 4] in xywh format
                        raw_confs = raw_np[0, :, 4]    # [8400] confidence scores

                        # Filter by confidence first to reduce search space
                        # PERFORMANCE FIX: Use much higher threshold to dramatically reduce candidates
                        conf_mask = raw_confs > 0.9  # Keep high threshold for performance
                        valid_indices = np.where(conf_mask)[0]
                        
                        # PERFORMANCE FIX: Limit to top N candidates if still too many
                        if len(valid_indices) > 20:
                            # Get top 20 by confidence only
                            top_indices = np.argpartition(raw_confs[valid_indices], -20)[-20:]
                            valid_indices = valid_indices[top_indices]
                        
                        # If we have very few candidates, lower the threshold and try again
                        # This helps with edge cases without impacting normal performance
                        if len(valid_indices) < num_detections and len(valid_indices) < 3:
                            # Fallback: use lower threshold for edge cases
                            conf_mask_fallback = raw_confs > 0.7
                            valid_indices = np.where(conf_mask_fallback)[0]
                            if len(valid_indices) > 30:
                                # Still too many, take top 30
                                top_indices = np.argpartition(raw_confs[valid_indices], -30)[-30:]
                                valid_indices = valid_indices[top_indices]
                        
                        # PERFORMANCE FIX: Skip expensive matching if still too many candidates
                        if len(valid_indices) > 20:
                            self._logging.warning(f"Too many pose candidates ({len(valid_indices)}), using simple approach")
                            # Just use the first N detections with dummy keypoints
                            for i in range(min(num_detections, 10)):
                                # Create dummy keypoints (all zeros)
                                keypoints = np.zeros((17, 3))
                                keypoints_list.append(keypoints)
                            return keypoints_list

                        # Debug logging removed for less verbosity

                        if len(valid_indices) == 0:
                            self._logging.warning("No confident raw detections found")
                            keypoints_data = None
                        else:
                            for i in range(num_detections):
                                det_bbox = det_info.xyxy_boxes[i]
                                # Convert to center for comparison
                                det_cx = (det_bbox[0] + det_bbox[2]) / 2
                                det_cy = (det_bbox[1] + det_bbox[3]) / 2
                                det_w = det_bbox[2] - det_bbox[0]
                                det_h = det_bbox[3] - det_bbox[1]

                                # Find best matching raw detection
                                best_idx = -1
                                best_score = -1.0
                                best_distance = float('inf')
                                
                                # Check if detection is near viewport edge
                                edge_margin = 100  # pixels from edge
                                is_edge_detection = (
                                    det_bbox[0] < edge_margin or  # left edge
                                    det_bbox[1] < edge_margin or  # top edge
                                    det_bbox[2] > w - edge_margin or  # right edge
                                    det_bbox[3] > h - edge_margin  # bottom edge
                                )

                                for idx in valid_indices:
                                    # Raw bbox is in xywh format
                                    raw_cx = raw_bboxes[idx, 0]
                                    raw_cy = raw_bboxes[idx, 1]
                                    raw_w = raw_bboxes[idx, 2]
                                    raw_h = raw_bboxes[idx, 3]

                                    # YOLO raw outputs are in model coordinate space
                                    # Need to scale to image coordinates
                                    scale_x = w / self._model_input_size
                                    scale_y = h / self._model_input_size
                                    raw_cx = raw_cx * scale_x
                                    raw_cy = raw_cy * scale_y
                                    raw_w = raw_w * scale_x
                                    raw_h = raw_h * scale_y

                                    # Calculate center distance
                                    dx = abs(det_cx - raw_cx)
                                    dy = abs(det_cy - raw_cy)
                                    center_dist = (dx*dx + dy*dy) ** 0.5
                                    
                                    # For edge detections, use more lenient distance threshold
                                    max_dist_factor = 1.2 if is_edge_detection else 0.8
                                    if dx > (det_w + raw_w) * max_dist_factor or dy > (det_h + raw_h) * max_dist_factor:
                                        continue

                                    # Calculate IoU
                                    inter_w = min(det_cx + det_w/2, raw_cx + raw_w/2) - max(det_cx - det_w/2, raw_cx - raw_w/2)
                                    inter_h = min(det_cy + det_h/2, raw_cy + raw_h/2) - max(det_cy - det_h/2, raw_cy - raw_h/2)

                                    iou = 0.0
                                    if inter_w > 0 and inter_h > 0:
                                        inter_area = inter_w * inter_h
                                        union_area = det_w * det_h + raw_w * raw_h - inter_area
                                        iou = inter_area / union_area
                                    
                                    # For edge detections, combine IoU and distance scores
                                    if is_edge_detection:
                                        # Normalize distance (0 = perfect match, 1 = far away)
                                        max_possible_dist = ((w**2 + h**2) ** 0.5)
                                        norm_distance = min(center_dist / max_possible_dist, 1.0)
                                        
                                        # Weighted score: distance matters more for edge cases
                                        # IoU weight: 20%, Distance weight: 60%, Confidence weight: 20%
                                        conf_score = raw_confs[idx]
                                        combined_score = (
                                            iou * 0.2 + 
                                            (1.0 - norm_distance) * 0.6 + 
                                            conf_score * 0.2
                                        )
                                        
                                        if combined_score > best_score:
                                            best_score = combined_score
                                            best_idx = idx
                                            best_distance = center_dist
                                    else:
                                        # For non-edge detections, prioritize IoU
                                        if iou > best_score or (iou == best_score and center_dist < best_distance):
                                            best_score = iou
                                            best_idx = idx
                                            best_distance = center_dist

                                # For edge cases, accept lower quality matches
                                min_score_threshold = 0.15 if is_edge_detection else 0.3
                                
                                if best_idx >= 0 and best_score >= min_score_threshold:
                                    # Get keypoint data (last 51 values)
                                    kpt_data = raw_np[0, best_idx, 5:56]
                                    # Reshape to [17, 3] for 17 keypoints with x,y,conf
                                    kpts = kpt_data.reshape(17, 3)
                                    keypoints_list.append(kpts)
                                    pass  # Debug logging removed
                                else:
                                    # No matching keypoints found
                                    keypoints_list.append(np.zeros((17, 3)))

                        if keypoints_list:
                            keypoints_data = np.array(keypoints_list)
                            # Extracted keypoints successfully

                            # Apply smoothing to keypoints
                            keypoints_data = self._smooth_keypoints(keypoints_data, det_info)
                    else:
                        self._logging.warning(f"Raw output shape {raw_np.shape} doesn't match pose format")
            except Exception as e:
                self._logging.warning(f"Could not extract keypoints: {e}")
                import traceback
                traceback.print_exc()

            # Store last good detection data
            if det_info is not None and len(det_info.scores) > 0:
                self._last_good_det_info = det_info
                self._last_good_keypoints = keypoints_data

        # Write pose data to shared memory
        if self._sh_mem_pose is not None:
            # Build the new pose data first
            new_pose_data = None

            if keypoints_data is not None and det_info is not None:
                try:
                    # Format: [num_persons, person1_data, person2_data, ...]
                    # Each person: [bbox(4), score(1), keypoints(17*3)]
                    num_persons = min(len(det_info.scores), len(keypoints_data))

                    # Build complete buffer first
                    temp_buffer = bytearray()

                    # Write number of persons as int32
                    temp_buffer.extend(struct.pack('i', num_persons))

                    # Write each person's data
                    valid_persons = 0
                    for i in range(num_persons):
                        # Pack bbox, score, and keypoints
                        x1, y1, x2, y2 = det_info.xyxy_boxes[i]
                        score = det_info.scores[i]

                        # Calculate bbox from keypoints for more accurate bounds
                        kpts = keypoints_data[i]
                        visible_kpts = kpts[kpts[:, 2] > 0.3]  # Only use visible keypoints

                        if len(visible_kpts) > 0:
                            # Get bounding box from visible keypoints
                            kpt_x_min = float(visible_kpts[:, 0].min())
                            kpt_x_max = float(visible_kpts[:, 0].max())
                            kpt_y_min = float(visible_kpts[:, 1].min())
                            kpt_y_max = float(visible_kpts[:, 1].max())

                            # Add some padding
                            padding = 20
                            x1 = max(0, kpt_x_min - padding)
                            y1 = max(0, kpt_y_min - padding)
                            x2 = min(w - 1, kpt_x_max + padding)
                            y2 = min(h - 1, kpt_y_max + padding)

                            pass  # Debug logging removed
                        else:
                            # Fall back to YOLO bbox and scale if needed
                            if max(x2, y2) <= self._model_input_size:
                                # Coordinates are in model space, scale to image space
                                scale_x = w / self._model_input_size
                                scale_y = h / self._model_input_size
                                x1 = x1 * scale_x
                                y1 = y1 * scale_y
                                x2 = x2 * scale_x
                                y2 = y2 * scale_y

                        # Validate bbox values
                        if any(abs(v) > 10000 for v in [x1, y1, x2, y2]):
                            self._logging.warning(f"Skipping person {i} with invalid bbox")
                            continue

                        # Debug score values
                        if i == 0:  # Only log first person to avoid spam
                            self._logging.debug(f"Raw score from detector: {score}, type: {type(score)}")

                        # Flatten keypoints [17, 3] -> [51]
                        kpts_flat = keypoints_data[i].flatten()

                        # Fix score packing - ensure it stays in 0-1 range
                        normalized_score = min(max(float(score), 0.0), 1.0)

                        # Pack all data: 4 (bbox) + 1 (score) + 51 (keypoints) = 56 floats
                        person_data = struct.pack('56f',
                            float(x1), float(y1), float(x2), float(y2), normalized_score,
                            *[float(k) for k in kpts_flat])

                        temp_buffer.extend(person_data)
                        valid_persons += 1

                    # Update person count with actual valid persons
                    if valid_persons != num_persons:
                        temp_buffer[0:4] = struct.pack('i', valid_persons)

                    new_pose_data = bytes(temp_buffer)
                    self._last_valid_pose_data = new_pose_data

                except Exception as e:
                    self._logging.error(f"Error building pose data: {e}")
            elif det_info is not None:
                # No keypoints, just write bounding boxes
                try:
                    num_persons = len(det_info.scores)

                    # Build complete buffer first
                    temp_buffer = bytearray()

                    # Write number of persons as int32
                    temp_buffer.extend(struct.pack('i', num_persons))

                    # Write each person's data with zero keypoints
                    valid_persons = 0
                    for i in range(num_persons):
                        x1, y1, x2, y2 = det_info.xyxy_boxes[i]
                        score = det_info.scores[i]

                        # Scale bbox coordinates from model space to image space if needed
                        if max(x2, y2) <= self._model_input_size:
                            # Coordinates are in model space, scale to image space
                            scale_x = w / self._model_input_size
                            scale_y = h / self._model_input_size
                            x1 = x1 * scale_x
                            y1 = y1 * scale_y
                            x2 = x2 * scale_x
                            y2 = y2 * scale_y

                        # Validate bbox values
                        if any(abs(v) > 10000 for v in [x1, y1, x2, y2]):
                            self._logging.warning(f"Skipping person {i} with invalid bbox")
                            continue

                        # Fix score packing - ensure it stays in 0-1 range
                        normalized_score = min(max(float(score), 0.0), 1.0)

                        # Pack bbox, score, and zero keypoints
                        person_data = struct.pack('56f',
                            float(x1), float(y1), float(x2), float(y2), normalized_score,
                            *[0.0] * 51)  # 51 zeros for keypoints

                        temp_buffer.extend(person_data)
                        valid_persons += 1

                    # Update person count with actual valid persons
                    if valid_persons != num_persons:
                        temp_buffer[0:4] = struct.pack('i', valid_persons)

                    new_pose_data = bytes(temp_buffer)
                    self._last_valid_pose_data = new_pose_data

                except Exception as e:
                    self._logging.error(f"Error building bbox-only pose data: {e}")

            # Always write something to the buffer - either new data or last valid
            if new_pose_data and len(new_pose_data) <= self._sh_mem_pose.size:
                self._sh_mem_pose.buf[0:len(new_pose_data)] = new_pose_data
                if self._log_frame_count % self._log_frequency == 0:
                    self._logging.debug(f"Wrote new pose data: {len(new_pose_data)} bytes")
            elif self._last_valid_pose_data and len(self._last_valid_pose_data) <= self._sh_mem_pose.size:
                # No new valid data - keep last valid data in buffer
                self._sh_mem_pose.buf[0:len(self._last_valid_pose_data)] = self._last_valid_pose_data
                if self._log_frame_count % (self._log_frequency * 3) == 0:  # Even less frequent for this
                    self._logging.debug("Keeping last valid pose data in buffer")
            else:
                # No data at all - write zero persons
                self._sh_mem_pose.buf[0:4] = struct.pack('i', 0)
                if self._log_frame_count % self._log_frequency == 0:
                    self._logging.debug("No pose data available - wrote 0 persons")

        # Draw visualization
        draw_info = DrawInfo(params[ParamsIndex.DRAW_INFO])

        # Debug draw info
        if self._log_frame_count % self._log_frequency == 0:
            self._logging.info(f"DrawInfo value: {params[ParamsIndex.DRAW_INFO]} -> {draw_info} (Text:{bool(draw_info & DrawInfo.DRAW_TEXT)}, BBox:{bool(draw_info & DrawInfo.DRAW_BBOX)}, Skeleton:{bool(draw_info & DrawInfo.DRAW_SKELETON)})")

        # Check drawing mode flags
        overlay_only = (draw_info & DrawInfo.OVERLAY_ONLY) != 0
        transparent_bg = (draw_info & DrawInfo.TRANSPARENT_BG) != 0

        if overlay_only:
            # Create black background for overlay-only mode
            black_background = np.zeros_like(original_image)
            image[:] = black_background
            draw_target = image

            # If transparent background requested, we'll create an alpha mask
            if transparent_bg:
                # Create a separate mask to track where we draw
                alpha_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                alpha_mask = None
        else:
            # Composite mode - draw on original image
            image[:] = original_image[:]
            draw_target = image
            alpha_mask = None

        # Draw bounding boxes
        if det_info is not None:
            for i, (score, xyxy) in enumerate(zip(det_info.scores, det_info.xyxy_boxes)):
                if (draw_info & DrawInfo.DRAW_BBOX) != 0:
                    # Make bounding boxes much thicker and more visible
                    bbox_thickness = 4
                    bbox_color = (255, 0, 0)  # Bright red for visibility

                    # For drawing, calculate bbox from keypoints if available for better accuracy
                    x1, y1, x2, y2 = xyxy

                    if keypoints_data is not None and i < len(keypoints_data):
                        kpts = keypoints_data[i]
                        visible_kpts = kpts[kpts[:, 2] > 0.3]  # Only use visible keypoints

                        if len(visible_kpts) > 0:
                            # Get bounding box from visible keypoints
                            kpt_x_min = float(visible_kpts[:, 0].min())
                            kpt_x_max = float(visible_kpts[:, 0].max())
                            kpt_y_min = float(visible_kpts[:, 1].min())
                            kpt_y_max = float(visible_kpts[:, 1].max())

                            # Add some padding
                            padding = 20
                            x1 = int(max(0, kpt_x_min - padding))
                            y1 = int(max(0, kpt_y_min - padding))
                            x2 = int(min(w - 1, kpt_x_max + padding))
                            y2 = int(min(h - 1, kpt_y_max + padding))
                        else:
                            # No visible keypoints, use original bbox
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    else:
                        # No keypoint data, scale bbox if needed
                        if max(x2, y2) <= self._model_input_size:
                            # Coordinates are in model space, scale to image space
                            scale_x = w / self._model_input_size
                            scale_y = h / self._model_input_size
                            x1 = int(x1 * scale_x)
                            y1 = int(y1 * scale_y)
                            x2 = int(x2 * scale_x)
                            y2 = int(y2 * scale_y)
                        else:
                            # Already in image coordinates
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw bounding box
                    cv2.rectangle(draw_target, (x1, y1), (x2, y2), bbox_color, thickness=bbox_thickness)

                    # Update alpha mask if needed
                    if alpha_mask is not None:
                        cv2.rectangle(alpha_mask, (x1, y1), (x2, y2), (255,), thickness=bbox_thickness)

                if (draw_info & DrawInfo.DRAW_TEXT) != 0:
                    label = "person"
                    if (draw_info & DrawInfo.DRAW_CONF) != 0:
                        # Round score to 2 decimal places and ensure it's in 0-1 range
                        score_display = min(max(float(score), 0.0), 1.0)
                        label = f"{label} {score_display:.2f}"

                    # Text rendering settings
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8  # Slightly smaller for better fit
                    thickness = 2
                    text_color = (0, 255, 0)  # Bright green for visibility

                    # Get text size for proper positioning
                    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

                    # Position text above bounding box with padding (use scaled coordinates)
                    text_x = int(x1)  # Use the scaled x1 from bbox drawing
                    text_y = int(y1 - 5)  # 5 pixels above bbox

                    # Ensure text doesn't go off screen
                    if text_y - text_height < 0:
                        text_y = int(y2 + text_height + 5)  # Put below bbox if no room above

                    # Draw background rectangle for text (for readability)
                    bg_padding = 3
                    cv2.rectangle(draw_target,
                                (int(text_x - bg_padding), int(text_y - text_height - bg_padding)),
                                (int(text_x + text_width + bg_padding), int(text_y + bg_padding)),
                                (0, 0, 0), -1)  # Black background

                    # Draw label text
                    cv2.putText(draw_target, label, (int(text_x), int(text_y)),
                               font, font_scale, text_color, thickness)

                    # Update alpha mask for text area
                    if alpha_mask is not None:
                        # Use the actual text dimensions we calculated
                        cv2.rectangle(alpha_mask,
                                    (int(text_x - bg_padding), int(text_y - text_height - bg_padding)),
                                    (int(text_x + text_width + bg_padding), int(text_y + bg_padding)),
                                    255, -1)

        # Draw keypoints and skeleton if available
        if keypoints_data is not None and len(keypoints_data) > 0 and det_info is not None:
            # Have keypoints data
            for i in range(min(len(det_info.scores), len(keypoints_data))):
                kpts = keypoints_data[i].copy()  # [17, 3] - copy to avoid modifying original

                # Debug the raw keypoint values
                # Raw keypoints received

                # Check if keypoints are already in pixel coordinates
                # If max values are close to image dimensions, they're already scaled
                x_max = kpts[:, 0].max()
                y_max = kpts[:, 1].max()

                if x_max < 2.0 and y_max < 2.0:
                    # Normalized coordinates - scale to image size
                    kpts[:, 0] *= w
                    kpts[:, 1] *= h
                    pass  # Scaled normalized keypoints
                elif x_max > w/2 or y_max > h/2:
                    # Already in image pixel coordinates - DON'T SCALE!
                    pass  # Already in pixel coordinates
                else:
                    # Ambiguous - might be model coordinates
                    # Check if they're in model input size range
                    if x_max < self._model_input_size and y_max < self._model_input_size:
                        # Scale from model coordinates to image coordinates
                        scale_x = w / self._model_input_size
                        scale_y = h / self._model_input_size
                        kpts[:, 0] *= scale_x
                        kpts[:, 1] *= scale_y
                        pass  # Scaled from model coords
                    else:
                        pass  # Ambiguous coordinates - not scaling

                # Final keypoints processed

                # Draw skeleton connections first (so they appear behind keypoints)
                if (draw_info & DrawInfo.DRAW_SKELETON) != 0:  # Use separate skeleton flag
                    # Use pose confidence threshold from params
                    pose_threshold = params[ParamsIndex.POSE_THRESHOLD] if len(params) > 11 else 0.5
                    connections_drawn = 0
                    for connection in SKELETON_CONNECTIONS:
                        kpt1, kpt2 = connection
                        if kpts[kpt1, 2] > pose_threshold and kpts[kpt2, 2] > pose_threshold:  # use configurable threshold
                            # Fast coordinate extraction and clamping
                            pt1 = (int(np.clip(kpts[kpt1, 0], 0, w-1)), 
                                   int(np.clip(kpts[kpt1, 1], 0, h-1)))
                            pt2 = (int(np.clip(kpts[kpt2, 0], 0, w-1)), 
                                   int(np.clip(kpts[kpt2, 1], 0, h-1)))
                            # Make skeleton lines much thicker and more visible
                            line_thickness = 4
                            skeleton_color = (0, 255, 255)  # Bright cyan for visibility

                            # Draw skeleton line
                            cv2.line(draw_target, pt1, pt2, skeleton_color, line_thickness)
                            connections_drawn += 1

                            # Update alpha mask
                            if alpha_mask is not None:
                                cv2.line(alpha_mask, pt1, pt2, (255,), line_thickness)

                    # Skeleton drawing complete

                # Draw keypoints with color coding
                # Use pose confidence threshold from params
                pose_threshold = params[ParamsIndex.POSE_THRESHOLD] if len(params) > 11 else 0.5
                for j, (x, y, conf) in enumerate(kpts):
                    if conf > pose_threshold:  # use configurable threshold
                        # Get keypoint name and color
                        if j < len(COCO_KEYPOINTS):
                            kpt_name = COCO_KEYPOINTS[j]
                            color = self._keypoint_colors.get(kpt_name, (255, 255, 255))
                        else:
                            kpt_name = f"kpt_{j}"
                            color = (255, 255, 255)

                        # Draw much larger circles for visibility
                        keypoint_radius = 8
                        outline_thickness = 2

                        # Fast coordinate clamping
                        kx = int(np.clip(x, 0, w-1))
                        ky = int(np.clip(y, 0, h-1))
                        
                        # Draw keypoint
                        cv2.circle(draw_target, (kx, ky), keypoint_radius, color, -1)
                        # Add dark outline for contrast
                        cv2.circle(draw_target, (kx, ky), keypoint_radius, (0, 0, 0), outline_thickness)

                        # Update alpha mask
                        if alpha_mask is not None:
                            cv2.circle(alpha_mask, (kx, ky), keypoint_radius + outline_thickness, (255,), -1)

                        # Draw keypoint labels and/or confidence if enabled
                        if (draw_info & DrawInfo.DRAW_TEXT) != 0 or (draw_info & DrawInfo.DRAW_KEYPOINT_CONF) != 0:
                            # Build label text
                            label_parts = []

                            if (draw_info & DrawInfo.DRAW_TEXT) != 0:
                                # Check if we're in mirror mode (for webcam)
                                mirror_mode = (draw_info & DrawInfo.MIRROR_LABELS) != 0
                                
                                if mirror_mode:
                                    # Swap left/right labels for mirror view
                                    label_map = {
                                        'nose': 'N',
                                        'left_eye': 'RE', 'right_eye': 'LE',  # Swapped
                                        'left_ear': 'REa', 'right_ear': 'LEa',  # Swapped
                                        'left_shoulder': 'RS', 'right_shoulder': 'LS',  # Swapped
                                        'left_elbow': 'REl', 'right_elbow': 'LEl',  # Swapped
                                        'left_wrist': 'RW', 'right_wrist': 'LW',  # Swapped
                                        'left_hip': 'RH', 'right_hip': 'LH',  # Swapped
                                        'left_knee': 'RK', 'right_knee': 'LK',  # Swapped
                                        'left_ankle': 'RA', 'right_ankle': 'LA'  # Swapped
                                    }
                                else:
                                    # Normal labels
                                    label_map = {
                                        'nose': 'N',
                                        'left_eye': 'LE', 'right_eye': 'RE',
                                        'left_ear': 'LEa', 'right_ear': 'REa',
                                        'left_shoulder': 'LS', 'right_shoulder': 'RS',
                                        'left_elbow': 'LEl', 'right_elbow': 'REl',
                                        'left_wrist': 'LW', 'right_wrist': 'RW',
                                        'left_hip': 'LH', 'right_hip': 'RH',
                                        'left_knee': 'LK', 'right_knee': 'RK',
                                        'left_ankle': 'LA', 'right_ankle': 'RA'
                                    }
                                label_parts.append(label_map.get(kpt_name, str(j)))

                            if (draw_info & DrawInfo.DRAW_KEYPOINT_CONF) != 0:
                                # Add confidence value
                                label_parts.append(f"{conf:.2f}")

                            label = " ".join(label_parts)

                            # Draw label with background for readability
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.4
                            thickness = 1
                            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

                            # Position text to avoid overlap with circle
                            text_x = int(x) + 8
                            text_y = int(y) - 8

                            # Draw text background and text
                            cv2.rectangle(draw_target, 
                                        (text_x - 2, text_y - text_h - 2),
                                        (text_x + text_w + 2, text_y + 2),
                                        (0, 0, 0), -1)
                            cv2.putText(draw_target, label, (text_x, text_y),
                                       font, font_scale, color, thickness)

                            # Update alpha mask for text
                            if alpha_mask is not None:
                                cv2.rectangle(alpha_mask,
                                            (text_x - 2, text_y - text_h - 2),
                                            (text_x + text_w + 2, text_y + 2),
                                            255, -1)

        else:
            pass  # No keypoints data available

        # Handle transparent background if requested
        if overlay_only and transparent_bg and alpha_mask is not None:
            # Apply alpha mask to create transparency
            # We'll encode the alpha information in a special way TouchDesigner can use
            # Multiply the image by the mask to fade out non-drawn areas
            alpha_3channel = cv2.cvtColor(alpha_mask, cv2.COLOR_GRAY2BGR)
            alpha_normalized = alpha_3channel.astype(np.float32) / 255.0
            # Apply mask
            image[:] = (draw_target.astype(np.float32) * alpha_normalized).astype(
                np.uint8
            )

    def _smooth_keypoints(self, keypoints_data, det_info):  # pylint: disable=unused-argument
        """Apply exponential moving average smoothing to keypoints"""
        if keypoints_data is None or len(keypoints_data) == 0:
            return keypoints_data

        smoothed_data = []

        for i, kpts in enumerate(keypoints_data):
            # Try to match person to previous frame using simple index
            person_id = i  # Simple ID for now

            if person_id in self._smoothed_keypoints:
                # Apply exponential moving average
                prev_kpts = self._smoothed_keypoints[person_id]
                smoothed_kpts = kpts.copy()

                # Only smooth visible keypoints
                for j in range(17):
                    if kpts[j, 2] > 0.3:  # Keypoint is visible
                        if prev_kpts[j, 2] > 0.3:  # Was also visible before
                            # Smooth position
                            smoothed_kpts[j, 0] = self._smoothing_alpha * kpts[j, 0] + (1 - self._smoothing_alpha) * prev_kpts[j, 0]
                            smoothed_kpts[j, 1] = self._smoothing_alpha * kpts[j, 1] + (1 - self._smoothing_alpha) * prev_kpts[j, 1]

                kpts = smoothed_kpts

            # Store smoothed keypoints
            self._smoothed_keypoints[person_id] = kpts.copy()
            smoothed_data.append(kpts)

        return np.array(smoothed_data) if smoothed_data else keypoints_data

    def _draw_with_cached_data(self, image, original_image, det_info, keypoints_data, params):  # pylint: disable=unused-argument
        """Draw using cached detection data without reprocessing"""
        # This method is no longer needed - we handle duplicates in the main flow
        # Keeping for compatibility but it should not be called
        return image
