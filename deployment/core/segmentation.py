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
        
        # Load filtering options from config
        seg_config = self.config.get('segmentation', {})
        self.filter_border = seg_config.get('filter_border', False)
        self.border_margin = seg_config.get('border_margin', 5)
        
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
        
        elif model_source == 'local':
            # Load from local file
            seg_config = self.config.get('segmentation', {}).get('model', {})
            local_path = seg_config.get('local_path')
            
            # If local_path not specified, try using artifact_path as fallback
            if not local_path:
                local_path = seg_config.get('artifact_path')
            
            if not local_path:
                raise ValueError(
                    "Local path for YOLO model not configured. "
                    "Set segmentation.model.local_path in config.yaml"
                )
            
            # Resolve path: if relative, assume relative to deployment/ directory
            from pathlib import Path
            deployment_dir = Path(__file__).parent.parent
            model_path_obj = Path(local_path)
            
            if not model_path_obj.is_absolute():
                model_path_obj = deployment_dir / model_path_obj
            
            model_path = str(model_path_obj)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"YOLO model file not found: {model_path}. "
                    f"Check segmentation.model.local_path in config.yaml"
                )
        else:
            raise ValueError(
                f"Unknown YOLO model source: {model_source}. "
                "Must be 'mlflow' or 'local'."
            )
        
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
    
    def segment_frame(self, bgr_image: np.ndarray, visualize: bool = False, save_path: str = None):
        """
        Segment a single BGR frame (numpy array) and return instances.
        
        This method is designed for real-time inference from camera stream.
        Includes Level 1 filtering: confidence threshold and border checks.
        
        Args:
            bgr_image: BGR image as numpy array (H, W, 3), dtype uint8
            visualize: If True, create visualization image with all detections
            save_path: Path to save visualization (if visualize=True)
            
        Returns:
            If visualize=False:
                List of dicts, each containing:
                    - 'box': bounding box [x1, y1, x2, y2]
                    - 'mask': binary mask (numpy array, uint8)
                    - 'maskImg': masked image (numpy array, BGR)
                    - 'score': confidence score
            If visualize=True:
                Tuple (instances, vis_image_bgr)
        """
        import logging
        logger = logging.getLogger(self.__class__.__name__)
        
        if bgr_image is None or bgr_image.size == 0:
            return ([] if not visualize else ([], None))
        
        img_height, img_width = bgr_image.shape[:2]
        
        # Use a threshold slightly lower than the target threshold to filter
        # noise early. This prevents clogging the max_det buffer with
        # very low confidence detections.
        yolo_conf = max(0.2, self.confidence_threshold - 0.1)
        logger.debug(
            f"YOLO inference with conf={yolo_conf}, "
            f"final threshold={self.confidence_threshold}"
        )
        
        # Run YOLOv8 inference on numpy array
        # YOLO (Ultralytics) expects BGR format when passed a numpy array
        # Use track() for tracking (BoT-SORT is default or configurable)
        results = self.model.track(
            bgr_image,
            device=self.device,
            conf=yolo_conf,
            persist=True,
            tracker="botsort.yaml",
            verbose=False
        )
        
        # Extract results from first (and only) image
        result_obj = results[0]
        
        # Get boxes, masks, scores
        boxes = result_obj.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        scores = result_obj.boxes.conf.cpu().numpy()
        masks = result_obj.masks  # YOLO masks object
        
        # Get track IDs if available
        track_ids = None
        if result_obj.boxes.is_track and result_obj.boxes.id is not None:
            track_ids = result_obj.boxes.id.cpu().numpy()
        
        # Create visualization if requested
        vis_image_bgr = None
        if visualize:
            # Input is BGR, so copy is BGR
            vis_image_bgr = bgr_image.copy()
        
        total_detections = len(boxes)
        filtered_by_conf = 0
        filtered_by_border = 0
        instances = []
        
        for idx in range(len(boxes)):
            score = float(scores[idx])
            x1, y1, x2, y2 = boxes[idx]
            
            # Log all detections for debugging
            logger.debug(f"Detection {idx}: score={score:.3f}, box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
            
            # Visualize all detections (even filtered ones)
            if visualize:
                color = (0, 255, 0) if score >= self.confidence_threshold else (0, 0, 255)  # Green if passed, Red if filtered
                thickness = 2 if score >= self.confidence_threshold else 1
                cv2.rectangle(vis_image_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                label = f"{score:.2f}"
                cv2.putText(vis_image_bgr, label, (int(x1), int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Level 1 Filter: Confidence threshold
            if score < self.confidence_threshold:
                filtered_by_conf += 1
                continue
            
            # Level 1 Filter: Border check - reject if bbox intersects frame border
            # NOTE: This is more strict than test.py - can be disabled in config
            if self.filter_border:
                if (x1 <= self.border_margin or y1 <= self.border_margin or 
                    x2 >= (img_width - self.border_margin) or 
                    y2 >= (img_height - self.border_margin)):
                    filtered_by_border += 1
                    if visualize:
                        cv2.putText(vis_image_bgr, "BORDER", 
                                   (int(x1), int(y2) + 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                                   (0, 165, 255), 1)
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
            
            # Create maskImg (bird on black background) - BGR format (since input is BGR)
            maskImg = cv2.bitwise_and(bgr_image, mask_3d)
            
            # Extract bounding box
            box = [int(x1), int(y1), int(x2), int(y2)]
            
            # Get track ID for this instance
            track_id = -1
            if track_ids is not None and idx < len(track_ids):
                track_id = int(track_ids[idx])
            
            instances.append({
                'box': box,
                'mask': mask_thresh,  # Binary mask
                'maskImg': maskImg,  # Masked image (BGR)
                'score': score,
                'track_id': track_id
            })
            
            # Draw accepted detection on visualization
            if visualize:
                cv2.rectangle(vis_image_bgr, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                label = f"ID:{track_id} {score:.2f}" if track_id != -1 else f"ACCEPTED {score:.2f}"
                cv2.putText(vis_image_bgr, label, (box[0], box[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Log statistics with more details
        if total_detections == 0:
            logger.warning(
                f"YOLO found NO detections (conf={yolo_conf}). "
                f"Image may not contain chickens or model needs retraining."
            )
        else:
            logger.info(
                f"Segmentation stats: total={total_detections}, "
                f"accepted={len(instances)} (threshold={self.confidence_threshold}), "
                f"filtered_by_conf={filtered_by_conf}, "
                f"filtered_by_border={filtered_by_border}"
            )
            # Log score distribution for debugging
            if len(scores) > 0:
                min_score = float(scores.min())
                max_score = float(scores.max())
                mean_score = float(scores.mean())
                logger.debug(
                    f"Score distribution: min={min_score:.3f}, "
                    f"max={max_score:.3f}, mean={mean_score:.3f}"
                )
        
        # Save visualization if requested
        if visualize:
            if save_path:
                cv2.imwrite(save_path, vis_image_bgr)
                logger.info(f"Segmentation visualization saved to: {save_path}")
            return instances, vis_image_bgr
        
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
