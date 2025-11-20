"""
Segmentation module for MVP inference
Loads Mask R-CNN and performs instance segmentation on depth images
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from segmentation.models.Mask_rcnn_Model import get_model_instance_segmentation


class SegmentationInference:
    """Mask R-CNN inference for broiler segmentation"""
    
    def __init__(self, model_path: str, device: str = None, confidence_threshold: float = 0.90):
        """
        Initialize segmentation model
        
        Args:
            model_path: Path to trained Mask R-CNN weights (.pth file)
            device: 'cuda', 'cpu', or None (auto-detect)
            confidence_threshold: Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        num_classes = 2  # Background + chicken
        self.model = get_model_instance_segmentation(num_classes)
        self.model.to(self.device)
        self.model.eval()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
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
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_tensor = transforms.Compose([transforms.ToTensor()])(img)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model([img_tensor.to(self.device)])
        
        boxes = prediction[0]['boxes'].cpu()
        labels = prediction[0]['labels'].cpu()
        scores = prediction[0]['scores'].cpu()
        masks = prediction[0]['masks'].cpu()
        
        # Read original image for maskImg
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        instances = []
        for idx in range(boxes.shape[0]):
            if scores[idx] >= self.confidence_threshold:
                # Extract mask
                mask = masks[idx, 0].mul(255).byte().numpy()
                
                # Threshold mask
                _, mask_thresh = cv2.threshold(np.uint8(mask), 100, 255, 0)
                mask_3d = np.dstack((mask_thresh, mask_thresh, mask_thresh))
                
                # Create maskImg (bird on black background)
                maskImg = cv2.bitwise_and(original_img, mask_3d)
                
                # Extract bounding box
                box = [
                    int(boxes[idx][0].item()),
                    int(boxes[idx][1].item()),
                    int(boxes[idx][2].item()),
                    int(boxes[idx][3].item())
                ]
                
                instances.append({
                    'mask': mask_thresh,  # Binary mask
                    'maskImg': maskImg,  # Masked image (RGB)
                    'box': box,
                    'score': float(scores[idx].item()),
                    'instance_id': idx
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

