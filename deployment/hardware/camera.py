import cv2
import numpy as np
from typing import Tuple, Optional
from pyorbbecsdk import *
import logging
import threading
import queue
import time

class OrbbecCamera:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        """
        Initialize Orbbec Gemini camera.
        Configures Color and Depth streams, as well as hardware synchronization (Align).
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.width = width
        self.height = height
        self.fps = fps
        
        self.pipeline = Pipeline()
        self.config = Config()
        self.device = None
        
        try:
            # 1. Device discovery
            ctx = Context()
            device_list = ctx.query_devices()
            if device_list.get_count() == 0:
                raise RuntimeError("Orbbec device not found. Check USB connection.")
            
            self.device = device_list.get_device_by_index(0)
            self.logger.info(f"Connected to: {self.device.get_device_info().get_name()}")

            # 2. Depth stream configuration
            # Y16 format: 16-bit depth format for Orbbec (equivalent to RealSense Z16)
            # Note: Y16 is the standard depth format for Orbbec cameras
            depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profile_list.get_video_stream_profile(
                self.width, self.height, OBFormat.Y16, self.fps
            )
            if not depth_profile:
                raise RuntimeError(
                    f"Depth profile {self.width}x{self.height}@{self.fps} not supported"
                )
            self.config.enable_stream(depth_profile)

            # 3. Color stream configuration
            color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profile_list.get_video_stream_profile(
                self.width, self.height, OBFormat.RGB, self.fps
            )
            if not color_profile:
                raise RuntimeError(
                    f"Color profile {self.width}x{self.height}@{self.fps} not supported"
                )
            self.config.enable_stream(color_profile)

            # 4. Enable alignment (Align)
            # This is critical: overlays depth map onto RGB frame.
            # Now pixel (x,y) in color corresponds to pixel (x,y) in depth.
            try:
                self.config.set_align_mode(OBAlignMode.HW_MODE)
            except OBError as e:
                 # Fallback to SW_MODE if HW_MODE is not supported
                self.logger.warning(f"HW Align failed ({e}), falling back to SW Align")
                self.config.set_align_mode(OBAlignMode.SW_MODE)

        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            raise e

    def start(self):
        """Start the pipeline."""
        try:
            self.pipeline.start(self.config)
            self.logger.info("Camera pipeline started.")
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            raise

    def stop(self):
        """Stop the pipeline."""
        try:
            self.pipeline.stop()
            self.logger.info("Camera pipeline stopped.")
        except Exception as e:
            self.logger.error(f"Failed to stop pipeline: {e}")

    def get_intrinsics(self) -> dict:
        """
        Get camera intrinsic parameters.
        Required for accurate conversion of 2D pixels to 3D coordinates (Point Cloud),
        accounting for actual focal length.
        """
        try:
            # Get parameters after pipeline start, as they depend on resolution
            param = self.pipeline.get_camera_param()
            
            # Parameters for Depth (or Aligned Color-to-Depth)
            intrinsic = param.depth_intrinsic

            return {
                'fx': intrinsic.fx,  # Focal length in X direction
                'fy': intrinsic.fy,  # Focal length in Y direction
                'cx': intrinsic.cx,  # Optical center in X direction
                'cy': intrinsic.cy,  # Optical center in Y direction
                'width': intrinsic.width,
                'height': intrinsic.height
            }
        except Exception as e:
            self.logger.error(f"Could not get intrinsics: {e}")
            return {}

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get synchronized frames.
        
        Returns:
            color_image (np.ndarray): BGR image (uint8).
            depth_image (np.ndarray): Depth image (uint16) in millimeters.
                                      Y16 format, equivalent to RealSense Z16.
                                      Values represent distance in millimeters.
        """
        try:
            # Wait for frames (timeout 100 ms)
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                self.logger.warning("No frames received")
                return None, None

            # Extract frames
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                return None, None

            # --- Color processing ---
            # Data comes in RGB format, convert to numpy
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            color_data = color_data.reshape((self.height, self.width, 3))
            # OpenCV uses BGR, so convert RGB -> BGR
            color_image = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)

            # --- Depth processing ---
            # Y16 format: 16-bit depth data
            # Check if scaling is needed to convert to millimeters
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((self.height, self.width))
            
            # Try to get value scale from depth frame (if available in pyorbbecsdk)
            # If getValueScale() is available, multiply by scale to get mm
            # Otherwise, assume data is already in millimeters (default for Y16)
            try:
                # Check if depth_frame has get_value_scale method
                if hasattr(depth_frame, 'get_value_scale'):
                    scale = depth_frame.get_value_scale()
                    if scale != 1.0:
                        depth_data = (depth_data * scale).astype(np.uint16)
                        self.logger.debug(f"Applied depth scale: {scale}")
            except (AttributeError, Exception):
                # If method doesn't exist, assume Y16 is already in millimeters
                # This is the default behavior for Orbbec Y16 format
                pass
            
            # Return depth image in millimeters (Z16 equivalent)
            depth_image = depth_data.copy()

            return color_image, depth_image

        except Exception as e:
            self.logger.error(f"Error getting frames: {e}")
            return None, None


class CameraProducer(threading.Thread):
    """
    Stage 1: Camera Source (Producer)
    
    Captures frames from Orbbec Gemini 336 camera and puts them into InputQueue.
    Implements FPS control by dropping old frames if queue is full.
    """
    
    def __init__(
        self,
        input_queue: queue.Queue,
        camera_config: dict,
        max_queue_size: int = 2
    ):
        """
        Initialize camera producer thread.
        
        Args:
            input_queue: Queue to put frames into. Format: (frame_id, bgr_image, depth_image_z16)
            camera_config: Camera configuration dict with width, height, fps, etc.
            max_queue_size: Maximum queue size. If exceeded, old frames are dropped.
        """
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.camera_config = camera_config
        self.max_queue_size = max_queue_size
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.running = False
        self.frame_id = 0
        
        # Initialize camera
        self.camera = OrbbecCamera(
            width=camera_config.get('width', 1280),
            height=camera_config.get('height', 720),
            fps=camera_config.get('fps', 30)
        )
        
    def start(self):
        """Start camera and producer thread."""
        self.logger.info("Starting camera producer...")
        self.camera.start()
        self.running = True
        super().start()
        self.logger.info("Camera producer started")
        
    def stop(self):
        """Stop producer thread and camera."""
        self.logger.info("Stopping camera producer...")
        self.running = False
        # Wait for thread to finish (with timeout)
        self.join(timeout=2.0)
        self.camera.stop()
        self.logger.info("Camera producer stopped")
        
    def run(self):
        """Main producer loop - captures frames and puts them in queue."""
        frame_interval = 1.0 / self.camera_config.get('fps', 30)
        last_frame_time = time.time()
        
        while self.running:
            try:
                # Control FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_frame_time = time.time()
                
                # Get frames from camera (returns BGR)
                bgr_image, depth_image_z16 = self.camera.get_frames()
                
                if bgr_image is None or depth_image_z16 is None:
                    continue
                
                # Pass BGR image directly to pipeline (matching batch mode behavior)
                # Do not convert to RGB, as YOLO/OpenCV pipeline expects BGR
                
                # Drop old frames if queue is full (FPS control)
                if self.input_queue.qsize() >= self.max_queue_size:
                    try:
                        # Remove oldest frame
                        self.input_queue.get_nowait()
                        self.logger.debug(
                            f"Dropped old frame (queue full: {self.max_queue_size})"
                        )
                    except queue.Empty:
                        pass
                
                # Put new frame in queue: (frame_id, bgr_image, depth_image_z16)
                frame_data = (self.frame_id, bgr_image, depth_image_z16)
                self.input_queue.put(frame_data, block=False)
                self.frame_id += 1
                
            except queue.Full:
                self.logger.warning("Queue full, dropping frame")
            except Exception as e:
                self.logger.error(f"Error in producer loop: {e}", exc_info=True)
                # Small delay to prevent tight error loop
                time.sleep(0.1)
                
    def get_intrinsics(self) -> dict:
        """Get camera intrinsic parameters."""
        return self.camera.get_intrinsics()


if __name__ == "__main__":
    # Example 1: Direct camera usage
    print("=== Example 1: Direct Camera Usage ===")
    cam = OrbbecCamera()
    cam.start()
    
    print("Camera intrinsics:", cam.get_intrinsics())
    
    try:
        while True:
            color, depth = cam.get_frames()
            if color is not None:
                # Depth visualization for testing
                depth_display = cv2.normalize(
                    depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                
                cv2.imshow("Color", color)
                cv2.imshow("Depth (Visual)", depth_display)
                
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    
    # Example 2: Producer with queue
    print("\n=== Example 2: Camera Producer with Queue ===")
    input_queue = queue.Queue(maxsize=2)
    camera_config = {
        'width': 1280,
        'height': 720,
        'fps': 30
    }
    
    producer = CameraProducer(
        input_queue=input_queue,
        camera_config=camera_config,
        max_queue_size=2
    )
    
    producer.start()
    
    try:
        frame_count = 0
        while frame_count < 100:  # Process 100 frames
            try:
                frame_id, rgb_image, depth_image_z16 = input_queue.get(timeout=1.0)
                frame_count += 1
                print(f"Received frame {frame_id}: RGB shape={rgb_image.shape}, "
                      f"Depth shape={depth_image_z16.shape}")
            except queue.Empty:
                print("No frames received")
                break
    finally:
        producer.stop()
        print("Producer stopped")