from typing import Optional, Dict, Tuple, Union
import json
import struct
from multiprocessing.shared_memory import ShareableList

import numpy as np
import torch
import cv2

from .info import ParamsIndex, DrawInfo
from .base_processing import BaseProcessServer
from ..detection import PyTorchYoloDetector
from .color_utils import hex_to_rgb, rgb_to_hex


class DetectorProcess(BaseProcessServer):

    def __init__(
        self,
        path_to_model: str,
        update_shared_mem_name: str,
        params_shared_mem_name: str,
        array_shared_mem_name: str,
        image_width: int,
        image_height: int,
        num_channels: int,
        image_dtype,
        device: Optional[str] = None,
    ):
        super().__init__(
            update_shared_mem_name,
            params_shared_mem_name,
            array_shared_mem_name,
            image_width,
            image_height,
            num_channels,
            image_dtype,
        )

        if device is None:
            if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available') and torch.backends.mps.is_available():  # type: ignore
                device_name = "mps"
            elif torch.cuda.is_available():
                device_name = "cuda"
            else:
                device_name = "cpu"
                self._logging.warning("GPU is not available. Using CPU. Can be slow")
        else:
            device_name = device
        
        self._logging.info(f"Detection model loaded: {path_to_model} | Device: {device_name}")

        self._detector = PyTorchYoloDetector(path_to_model,
                                             device=torch.device(device_name),
                                             trace=True,
                                             numpy_post_process=True)

        self._color_mapping = self.generate_class_colormap()

    @property
    def color_mapping(self):
        return self._color_mapping.copy()

    def generate_class_colormap(self) -> Dict[str, Tuple[int, int, int]]:
        return {class_name: (255, 0, 0) for class_name in self._detector.class_mapping.values()}

    def save_class_colormap(self, path_to_file: str):
        with open(path_to_file, "w", encoding="utf-8") as f:
            json.dump({class_name: rgb_to_hex(color)
                      for class_name, color in self.color_mapping.items()}, f, indent=2)

    def load_class_colormap(self, path_to_file: str):
        with open(path_to_file, "rb") as f:
            new_color_mapping = json.load(f)

        for key in self._color_mapping:
            if key not in new_color_mapping:
                raise KeyError(f"Cannot find: '{key}' in the new colormap")

            try:
                self._color_mapping[key] = hex_to_rgb(new_color_mapping[key])
            except ValueError as exc:
                raise KeyError(f"Invalid hex color by '{key}'") from exc

    def process(self,
                image: np.ndarray,
                params: Union[list, ShareableList],
                ):
        det_info = self._detector.predict(
            image,
            score_threshold=params[ParamsIndex.SCORE_THRESH],
            nms_threshold=params[ParamsIndex.IOU_THRESH],
            max_k=params[ParamsIndex.TOP_K],
            eta=params[ParamsIndex.ETA])

        # Write detection data to shared memory buffer
        if hasattr(self, '_sh_mem_detections') and self._sh_mem_detections is not None:
            try:
                # Format: [num_detections, detection1_data, detection2_data, ...]
                # Each detection: [x1, y1, x2, y2, score, class_id]
                # All as float32

                num_detections = len(det_info.scores)
                buffer_pos = 0

                # Write number of detections as int32
                self._sh_mem_detections.buf[buffer_pos:buffer_pos+4] = struct.pack('i', num_detections)
                buffer_pos += 4

                # Write each detection
                for i in range(num_detections):
                    # Get bounding box
                    x1, y1, x2, y2 = det_info.xyxy_boxes[i]
                    score = det_info.scores[i]

                    # Get class ID (need to map from string back to ID)
                    class_name = det_info.classes[i]
                    class_id = 0  # Default
                    for cid, cname in self._detector.class_mapping.items():
                        if cname == class_name:
                            class_id = cid
                            break

                    # Pack as 6 float32 values: x1, y1, x2, y2, score, class_id
                    detection_data = struct.pack('6f', 
                        float(x1), float(y1), float(x2), float(y2), 
                        float(score), float(class_id))

                    # Write to buffer if space available
                    if buffer_pos + 24 <= self._sh_mem_detections.size:  # 6 floats = 24 bytes
                        self._sh_mem_detections.buf[buffer_pos:buffer_pos+24] = detection_data
                        buffer_pos += 24
                    else:
                        self._logging.warning(f"Detection buffer full - wrote {i} of {num_detections} detections")
                        break

            except Exception as e:
                self._logging.error(f"Error writing detection data: {e}")

        draw_info = DrawInfo(params[ParamsIndex.DRAW_INFO])

        for class_label, score, xyxy in zip(det_info.classes, det_info.scores, det_info.xyxy_boxes):
            # Get color, using a default if class not in mapping
            color = self._color_mapping.get(class_label, (0, 255, 0))  # Default to green

            cv2.rectangle(image, xyxy[:2], xyxy[2:], color, thickness=1)

            if DrawInfo.DRAW_TEXT in draw_info:
                (_, text_height), _ = cv2.getTextSize(
                    class_label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
                new_y = xyxy[1] + text_height
                cv2.putText(image, class_label, (xyxy[0], new_y),
                            cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)

            if DrawInfo.DRAW_CONF in draw_info:
                cv2.putText(image, f"{score:.2f}", xyxy[:2], cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)
