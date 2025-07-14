from multiprocessing import shared_memory
from multiprocessing.shared_memory import ShareableList
import logging
import time
from typing import Union
from collections import deque

import numpy as np

from .info import ParamsIndex, BufferStates, States, DrawInfo


class BaseProcessServer:
    def __init__(self,
                 update_shared_mem_name: str,
                 params_shared_mem_name: str,
                 array_shared_mem_name: str,
                 image_width: int,
                 image_height: int,
                 num_channels: int,
                 image_dtype):
        self._logging = logging.getLogger("server_processing")
        dtype_size = np.dtype(image_dtype).itemsize
        self._image_size_bytes = image_height * image_width * num_channels * dtype_size
        self._image_width = image_width
        self._image_height = image_height
        self._num_channels = num_channels
        self._image_dtype = image_dtype
        self._sh_mem_update = None
        self._sh_mem_params = None
        self._sh_mem_array = None
        self._shared_array = None
        self._update_shared_mem_name = update_shared_mem_name
        self._params_shared_mem_name = params_shared_mem_name
        self._array_shared_mem_name = array_shared_mem_name
        self._detection_shared_mem_name = "detection_data"
        self._sh_mem_detections = None
        self._detection_buffer_size = 1024 * 16  # 16KB for detection data
        self._created_shared_memory = False  # Always false - TouchDesigner creates memory
        
        # FPS tracking
        self._frame_times = deque(maxlen=100)  # Store last 100 frame times
        self._last_frame_time = None
        self._total_frames = 0
        self._fps_log_interval = 30  # Log FPS every 30 frames
        self._last_fps_log_time = time.time()
        self._fps_log_time_interval = 5.0  # Also log every 5 seconds
        
        # Performance stats
        self._process_times = deque(maxlen=100)  # Track processing times

    def _get_shared_mem_update(self, create: bool):
        # macOS requires minimum 16384 bytes for shared memory
        # We only use the first 3 bytes for states, rest is unused
        min_size = max(16384, len(BufferStates))
        return shared_memory.SharedMemory(
            name=self._update_shared_mem_name, create=create, size=min_size)

    def _get_shared_mem_array(self, create: bool):
        return shared_memory.SharedMemory(
            name=self._array_shared_mem_name, create=create, size=self._image_size_bytes)

    def init_mem(self):
        assert self._sh_mem_update is None,  "Memory already initialized"
        self._logging.info(f"🔗 Connecting to TouchDesigner shared memory: states={self._update_shared_mem_name}, params={self._params_shared_mem_name}, array={self._array_shared_mem_name}")

        self._sh_mem_update = self._get_shared_mem_update(False)

        params = [None] * len(ParamsIndex)
        params[ParamsIndex.ETA] = 1.0
        params[ParamsIndex.IOU_THRESH] = 0.5
        params[ParamsIndex.SCORE_THRESH] = 0.5
        params[ParamsIndex.TOP_K] = 0
        params[ParamsIndex.IMAGE_WIDTH] = self._image_width
        params[ParamsIndex.IMAGE_HEIGHT] = self._image_height
        params[ParamsIndex.IMAGE_CHANNELS] = self._num_channels
        params[ParamsIndex.SHARED_ARRAY_MEM_NAME] = self._array_shared_mem_name
        params[ParamsIndex.SHARD_STATE_MEM_NAME] = self._update_shared_mem_name
        params[ParamsIndex.IMAGE_DTYPE] = self._image_dtype
        params[ParamsIndex.DRAW_INFO] = int(DrawInfo.DRAW_BBOX)

        self._sh_mem_params = shared_memory.ShareableList(
            name=self._params_shared_mem_name
        )

        self._sh_mem_array = self._get_shared_mem_array(False)

        self._shared_array = np.ndarray(
            (self._image_height, self._image_width, self._num_channels),
            dtype=self._image_dtype, buffer=self._sh_mem_array.buf)

        # Create detection data buffer
        try:
            self._sh_mem_detections = shared_memory.SharedMemory(
                name=self._detection_shared_mem_name,
                create=False,
                size=self._detection_buffer_size,
            )
        except:
            try:
                self._sh_mem_detections = shared_memory.SharedMemory(
                    name=self._detection_shared_mem_name,
                    create=True,
                    size=self._detection_buffer_size,
                )
            except:
                self._logging.warning(
                    "Could not create detection memory - detections won't be available"
                )
                self._sh_mem_detections = None
        
        # Create FPS stats buffer
        self._fps_stats_shared_mem_name = "fps_stats"
        self._fps_stats_size = 16384  # macOS minimum size - we'll use first 32 bytes for 4 floats
        try:
            self._sh_mem_fps = shared_memory.SharedMemory(
                name=self._fps_stats_shared_mem_name,
                create=False,
                size=self._fps_stats_size,
            )
        except:
            try:
                self._sh_mem_fps = shared_memory.SharedMemory(
                    name=self._fps_stats_shared_mem_name,
                    create=True,
                    size=self._fps_stats_size,
                )
                # Initialize with zeros
                np.frombuffer(self._sh_mem_fps.buf, dtype=np.float32)[:] = 0.0
            except:
                self._logging.warning("Could not create FPS stats memory")
                self._sh_mem_fps = None
        
        # Log connection status compactly
        shm_size = self._sh_mem_array.size // (1024*1024)  # Convert to MB
        self._logging.info(f"✅ Connected: {self._image_width}x{self._image_height} | 💾 {shm_size}MB shared memory | All buffers OK")

    def __enter__(self):
        self.init_mem()

        return self

    def __exit__(self, type, value, traceback):
        self._logging.info("Stop processing")
        self.dispose()

    def dispose(self):
        freed = []
        if self._sh_mem_update is not None:
            self._sh_mem_update.buf[BufferStates.SERVER_ALIVE] = States.NULL_STATE.value[0]
            self._sh_mem_update.close()
            if self._created_shared_memory:
                self._sh_mem_update.unlink()
            freed.append("states")

        del self._shared_array

        if self._sh_mem_array is not None:
            self._sh_mem_array.close()
            if self._created_shared_memory:
                self._sh_mem_array.unlink()
            freed.append("array")

        if self._sh_mem_params is not None:
            self._sh_mem_params.shm.close()
            if self._created_shared_memory:
                self._sh_mem_params.shm.unlink()
            freed.append("params")
        
        if self._sh_mem_detections is not None:
            self._sh_mem_detections.close()
            if self._created_shared_memory:
                self._sh_mem_detections.unlink()
            freed.append("detections")
        
        if hasattr(self, '_sh_mem_fps') and self._sh_mem_fps is not None:
            self._sh_mem_fps.close()
            if self._created_shared_memory:
                self._sh_mem_fps.unlink()
            freed.append("fps")
        
        if freed:
            self._logging.info(f"Freed shared memory: {', '.join(freed)}")

    def start_processing(self):
        if self._sh_mem_update is None or self._sh_mem_params is None or self._shared_array is None:
            raise RuntimeError("Shared memory not initialized. Call init_mem() first.")
            
        self._sh_mem_update.buf[BufferStates.SERVER] = States.NULL_STATE.value[0]
        self._sh_mem_update.buf[BufferStates.CLIENT] = States.NULL_STATE.value[0]
        self._sh_mem_update.buf[BufferStates.SERVER_ALIVE] = States.IS_SERVER_ALIVE.value[0]

        # Log FPS limit configuration
        fps_msg = "FPS limit: "
        if len(self._sh_mem_params) > ParamsIndex.FPS_LIMIT:
            fps_limit = self._sh_mem_params[ParamsIndex.FPS_LIMIT]
            if fps_limit > 60:
                fps_msg += f"{60} (capped from {fps_limit})"
            elif fps_limit > 0:
                fps_msg += f"{fps_limit}"
            else:
                fps_msg += "Unlimited"
        else:
            fps_msg += "Not configured (unlimited)"
        
        self._logging.info(f"🚀 {fps_msg} | 👁️ Awaiting frames...")

        while True:
            while self._sh_mem_update.buf[BufferStates.SERVER] != States.READY_SERVER_MESSAGE.value[0]:
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)  # 1ms sleep

            self._sh_mem_update.buf[BufferStates.SERVER] = States.NULL_STATE.value[0]
            
            # Track frame timing (use proper timing)
            current_time = time.time()

            image_height = self._sh_mem_params[ParamsIndex.IMAGE_HEIGHT]
            image_width = self._sh_mem_params[ParamsIndex.IMAGE_WIDTH]

            actual_image = self._shared_array[:image_height, :image_width]

            # Track processing time
            process_start = time.time()
            self.process(actual_image, self._sh_mem_params)
            process_time = time.time() - process_start
            self._process_times.append(process_time)
            
            self._shared_array[:image_height, :image_width] = actual_image

            self._sh_mem_update.buf[BufferStates.CLIENT] = States.READY_CLIENT_MESSAGE.value[0]
            
            # FPS limiting if configured
            if len(self._sh_mem_params) > ParamsIndex.FPS_LIMIT:
                fps_limit = self._sh_mem_params[ParamsIndex.FPS_LIMIT]
                # Cap at 60 FPS max
                if fps_limit > 60:
                    if self._total_frames == 1:  # Log once on first frame
                        self._logging.warning(f"FPS limit capped at 60 (requested: {fps_limit})")
                    fps_limit = 60
                if fps_limit > 0:
                    # Calculate target frame time
                    target_frame_time = 1.0 / fps_limit
                    # Calculate total time since last frame started
                    if self._last_frame_time is not None:
                        total_elapsed = time.time() - self._last_frame_time
                        # Sleep to maintain consistent frame rate
                        if total_elapsed < target_frame_time:
                            time.sleep(target_frame_time - total_elapsed)
            
            # Update last frame time after FPS limiting and calculate frame time
            frame_end_time = time.time()
            if self._last_frame_time is not None:
                frame_time = frame_end_time - self._last_frame_time
                self._frame_times.append(frame_time)
            self._last_frame_time = frame_end_time
            
            # Update frame counter and log FPS
            self._total_frames += 1
            self._update_fps_stats()
            
            # Log FPS periodically
            if (self._total_frames % self._fps_log_interval == 0 or 
                current_time - self._last_fps_log_time > self._fps_log_time_interval):
                self._log_fps()
                self._last_fps_log_time = current_time

    def process(self, image: np.ndarray, params: Union[list, ShareableList]):
        pass
    
    def _update_fps_stats(self):
        """Update FPS statistics in shared memory"""
        if self._sh_mem_fps is None or len(self._frame_times) == 0:
            return
            
        # Calculate current FPS
        if len(self._frame_times) > 1:
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        else:
            current_fps = 0.0
        
        # Calculate average processing time
        avg_process_time = sum(self._process_times) / len(self._process_times) if self._process_times else 0.0
        
        # Write to shared memory - only use first 16 bytes (4 floats) of the 16KB buffer
        stats = np.frombuffer(self._sh_mem_fps.buf[:16], dtype=np.float32)
        stats[0] = current_fps  # Server FPS
        stats[1] = avg_process_time * 1000  # Average process time in ms
        stats[2] = float(self._total_frames)  # Total frames processed
        # stats[3] is reserved for TD FPS (written by TouchDesigner)
    
    def _log_fps(self):
        """Log current FPS and performance stats"""
        if len(self._frame_times) == 0:
            return
            
        # Calculate stats
        avg_frame_time = sum(self._frame_times) / len(self._frame_times)
        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        avg_process_time = sum(self._process_times) / len(self._process_times) if self._process_times else 0.0
        
        # Get min/max process times for variability info
        if self._process_times:
            min_process = min(self._process_times) * 1000
            max_process = max(self._process_times) * 1000
        else:
            min_process = max_process = 0.0
        
        self._logging.info(
            f"📊 FPS: {current_fps:.1f} | ⚡ Process: {avg_process_time*1000:.1f}ms | 🎬 Frames: {self._total_frames}"
        )
