"""
Segmentation module for MVP inference
Loads YOLOv8-seg and performs instance segmentation on depth images
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Import ModelLoader for MLflow integration
from deployment.utils.load_model_from_mlflow import ModelLoader


class SegmentationInference:
    """YOLOv8-seg inference for broiler segmentation"""
    
    def __init__(self, model_path: str = None, device: str = None, 
                 confidence_threshold: float = 0.90, config: dict = None):
        """
        Initialize segmentation model
        
        Args:
            model_path: Path to trained YOLOv8 weights (.pt file) - used if source='local'
            device: 'cuda', 'cpu', or None (auto-detect)
            confidence_threshold: Minimum confidence score for detections
            config: Configuration dict with MLflow settings (if source='mlflow')
        """
        self.confidence_threshold = confidence_threshold
        self.config = config or {}
        
        # Device setup for YOLO (YOLO uses string format)
        if device is None:
            if torch.cuda.is_available():
                self.device = "0"  # YOLO uses "0" for cuda:0
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            # Convert torch device to YOLO format
            if device == 'cuda':
                self.device = "0"
            else:
                self.device = device
        
        # Load YOLOv8 model
        model_source = self.config.get('segmentation', {}).get('model', {}).get('source', 'local')
        
        if model_source == 'mlflow':
            # Load from MLflow using ModelLoader
            mlflow_config = self.config.get('mlflow', {})
            tracking_uri = mlflow_config.get('tracking_uri', 'http://localhost:5000')
            cache_dir = mlflow_config.get('cache_dir', 'models_cache')
            
            model_loader = ModelLoader(tracking_uri=tracking_uri, cache_dir=cache_dir)
            
            seg_config = self.config.get('segmentation', {}).get('model', {})
            run_id = seg_config.get('mlflow_run_id')
            artifact_path = seg_config.get('artifact_path', 'weights/best.pt')
            
            if not run_id or run_id == "REPLACE_WITH_YOLO_RUN_ID":
                raise ValueError(
                    "MLflow run_id not configured. "
                    "Set segmentation.model.mlflow_run_id in config.yaml"
                )
            
            model_path = model_loader.load_yolo_model(run_id, artifact_path)
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        self.model = YOLO(model_path)
        
        print(f"Segmentation model loaded from {model_path}")
        print(f"Using device: {self.device}")
    
    def segment_image(self, image_path: str) -> list:
        """
        Segment a single depth image and return instances
        
        Args:
            image_path: Path to depth image
            
        Returns:
            List of dicts, each containing:
                - 'mask': binary mask (numpy array, uint8)
                - 'maskImg': masked image (numpy array, RGB)
                - 'box': bounding box [x1, y1, x2, y2]
                - 'score': confidence score
                - 'instance_id': unique instance identifier
        """
        # Read original image for maskImg
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img_height, img_width = original_img.shape[:2]
        
        # Run YOLOv8 inference
        results = self.model.predict(
            image_path,
            device=self.device,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # Extract results from first (and only) image
        result_obj = results[0]
        
        # Get boxes, masks, scores
        boxes = result_obj.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        scores = result_obj.boxes.conf.cpu().numpy()
        masks = result_obj.masks  # YOLO masks object
        
        instances = []
        for idx in range(len(boxes)):
            if scores[idx] >= self.confidence_threshold:
                # Get mask for this instance
                if masks is not None and masks.data is not None:
                    # YOLO masks are in shape [N, H, W] where N is number of instances
                    mask_tensor = masks.data[idx].cpu().numpy()  # Shape: [H, W]
                    
                    # Resize mask to original image size if needed
                    if mask_tensor.shape != (img_height, img_width):
                        mask_tensor = cv2.resize(mask_tensor, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                    
                    # Convert to uint8 binary mask (0 or 255)
                    mask = (mask_tensor * 255).astype(np.uint8)
                else:
                    # Fallback: create mask from bounding box
                    x1, y1, x2, y2 = boxes[idx]
                    mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    mask[int(y1):int(y2), int(x1):int(x2)] = 255
                
                # Threshold mask
                _, mask_thresh = cv2.threshold(np.uint8(mask), 100, 255, 0)
                mask_3d = np.dstack((mask_thresh, mask_thresh, mask_thresh))
                
                # Create maskImg (bird on black background)
                maskImg = cv2.bitwise_and(original_img, mask_3d)
                
                # Extract bounding box
                x1, y1, x2, y2 = boxes[idx]
                box = [int(x1), int(y1), int(x2), int(y2)]
                
                instances.append({
                    'mask': mask_thresh,  # Binary mask
                    'maskImg': maskImg,  # Masked image (BGR, same as original)
                    'box': box,
                    'score': float(scores[idx]),
                    'instance_id': idx
                })
        
        return instances
    
    def segment_frame(self, rgb_image: np.ndarray) -> list:
        """
        Segment a single RGB frame (numpy array) and return instances.
        
        This method is designed for real-time inference from camera stream.
        Includes Level 1 filtering: confidence threshold and border checks.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3), dtype uint8
            
        Returns:
            List of dicts, each containing:
                - 'box': bounding box [x1, y1, x2, y2]
                - 'mask': binary mask (numpy array, uint8)
                - 'maskImg': masked image (numpy array, RGB)
                - 'score': confidence score
        """
        if rgb_image is None or rgb_image.size == 0:
            return []
        
        img_height, img_width = rgb_image.shape[:2]
        
        # Run YOLOv8 inference on numpy array
        # YOLO expects RGB format, which we have
        results = self.model.predict(
            rgb_image,
            device=self.device,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # Extract results from first (and only) image
        result_obj = results[0]
        
        # Get boxes, masks, scores
        boxes = result_obj.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        scores = result_obj.boxes.conf.cpu().numpy()
        masks = result_obj.masks  # YOLO masks object
        
        instances = []
        for idx in range(len(boxes)):
            score = float(scores[idx])
            
            # Level 1 Filter: Confidence threshold
            if score < self.confidence_threshold:
                continue
            
            x1, y1, x2, y2 = boxes[idx]
            
            # Level 1 Filter: Border check - reject if bbox intersects frame border
            if x1 <= 0 or y1 <= 0 or x2 >= img_width or y2 >= img_height:
                continue
            
            # Get mask for this instance
            if masks is not None and masks.data is not None:
                # YOLO masks are in shape [N, H, W] where N is number of instances
                mask_tensor = masks.data[idx].cpu().numpy()  # Shape: [H, W]
                
                # Resize mask to original image size if needed
                if mask_tensor.shape != (img_height, img_width):
                    mask_tensor = cv2.resize(
                        mask_tensor, 
                        (img_width, img_height), 
                        interpolation=cv2.INTER_NEAREST
                    )
                
                # Convert to uint8 binary mask (0 or 255)
                mask = (mask_tensor * 255).astype(np.uint8)
            else:
                # Fallback: create mask from bounding box
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                mask[int(y1):int(y2), int(x1):int(x2)] = 255
            
            # Threshold mask
            _, mask_thresh = cv2.threshold(np.uint8(mask), 100, 255, 0)
            mask_3d = np.dstack((mask_thresh, mask_thresh, mask_thresh))
            
            # Create maskImg (bird on black background) - RGB format
            maskImg = cv2.bitwise_and(rgb_image, mask_3d)
            
            # Extract bounding box
            box = [int(x1), int(y1), int(x2), int(y2)]
            
            instances.append({
                'box': box,
                'mask': mask_thresh,  # Binary mask
                'maskImg': maskImg,  # Masked image (RGB)
                'score': score
            })
        
        return instances
    
    def process_images(self, image_paths: list) -> dict:
        """
        Process multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dict mapping image_path -> list of instances
        """
        results = {}
        for img_path in image_paths:
            instances = self.segment_image(img_path)
            results[img_path] = instances
        return results
