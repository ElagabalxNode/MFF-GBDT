import cv2
import numpy as np
from typing import Tuple, Optional
from pyorbbecsdk import *
import logging

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
            # Y16 format (16-bit) is similar to RealSense Z16
            depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profile_list.get_video_stream_profile(
                self.width, self.height, OBFormat.Y16, self.fps
            )
            if not depth_profile:
                raise RuntimeError(f"Depth profile {width}x{height}@{fps} not supported")
            self.config.enable_stream(depth_profile)

            # 3. Color stream configuration
            color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profile_list.get_video_stream_profile(
                self.width, self.height, OBFormat.RGB, self.fps
            )
            if not color_profile:
                raise RuntimeError(f"Color profile {width}x{height}@{fps} not supported")
            self.config.enable_stream(color_profile)

            # 4. Enable alignment (Align)
            # This is critical: overlays depth map onto RGB frame.
            # Now pixel (x,y) in color corresponds to pixel (x,y) in depth.
            self.config.set_align_mode(OBAlignMode.HW_MODE)

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
            distortion = param.depth_distortion
            
            return {
                'fx': intrinsic.fx, # Focal length in X direction
                'fy': intrinsic.fy, # Focal length in Y direction
                'cx': intrinsic.cx, # Optical center in X direction
                'cy': intrinsic.cy, # Optical center in Y direction
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
                                      This is the equivalent of RAW Z16.
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
            # Data comes as Y16 (uint16), each pixel is distance in mm.
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((self.height, self.width))
            
            # No scaling needed for raw data, return as-is (Z16)
            depth_image = depth_data.copy()

            return color_image, depth_image

        except Exception as e:
            self.logger.error(f"Error getting frames: {e}")
            return None, None

if __name__ == "__main__":
    # Usage example
    cam = OrbbecCamera()
    cam.start()
    
    print("Camera intrinsics:", cam.get_intrinsics())
    
    try:
        while True:
            color, depth = cam.get_frames()
            if color is not None:
                # Depth visualization for testing (dynamic normalization, not static alpha)
                # This is only for human visualization!
                depth_display = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                
                cv2.imshow("Color", color)
                cv2.imshow("Depth (Visual)", depth_display)
                
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()