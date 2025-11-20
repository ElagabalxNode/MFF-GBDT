"""
Feature extraction module for MVP inference
Extracts 25 hand-crafted 2D/3D features + 2048 ResNet50 features
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
import torchvision.models as models

# Import manual feature extraction
from features.extraction.manual_features import chickenFeatureExt  # Will need to adapt this


class FeatureExtractor:
    """Extract 25 manual + 2048 ResNet features for each instance"""
    
    def __init__(self, resnet_weights_path: str = None, device: str = None):
        """
        Initialize feature extractor
        
        Args:
            resnet_weights_path: Path to trained ResNet50 weights (optional)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
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
        
        # Load ResNet50 for feature extraction
        self.resnet = models.resnet50(pretrained=False)
        # Remove final classification layer, keep features
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.eval()
        self.resnet.to(self.device)
        
        if resnet_weights_path and os.path.exists(resnet_weights_path):
            # Load trained weights if provided
            checkpoint = torch.load(resnet_weights_path, map_location=self.device)
            self.resnet.load_state_dict(checkpoint)
            print(f"ResNet weights loaded from {resnet_weights_path}")
        else:
            print("Using ImageNet-pretrained ResNet50 (no fine-tuning)")
        
        # Image preprocessing for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_manual_features(self, mask: np.ndarray, depth_image: np.ndarray = None) -> np.ndarray:
        """
        Extract 25 hand-crafted 2D/3D features from mask and depth image
        
        Args:
            mask: Binary mask (numpy array, uint8)
            depth_image: Depth image (numpy array, optional)
            
        Returns:
            Array of 25 features
        """
        # TODO: Implement or adapt from features/extraction/manual_features.py
        # This should extract:
        # - 2D features: area, perimeter, minRect, hull, convexity defects, etc.
        # - 3D features: volume, height statistics (max, min, mean, std), etc.
        
        # Placeholder - needs implementation
        features = np.zeros(25)
        
        # Example: basic 2D features
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            features[0] = cv2.contourArea(largest_contour)  # Area
            features[1] = cv2.arcLength(largest_contour, True)  # Perimeter
            # ... add more features
        
        return features
    
    def extract_resnet_features(self, maskImg: np.ndarray) -> np.ndarray:
        """
        Extract 2048 ResNet50 features from masked image
        
        Args:
            maskImg: Masked image (numpy array, RGB)
            
        Returns:
            Array of 2048 features
        """
        # Convert to PIL Image
        if maskImg.dtype != np.uint8:
            maskImg = (maskImg * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(cv2.cvtColor(maskImg, cv2.COLOR_BGR2RGB))
        
        # Preprocess and extract features
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.resnet(img_tensor)
        
        # Flatten to 2048-dim vector
        features = features.squeeze().cpu().numpy()
        
        return features
    
    def extract_all_features(self, mask: np.ndarray, maskImg: np.ndarray, 
                            depth_image: np.ndarray = None) -> np.ndarray:
        """
        Extract all features: 25 manual + 2048 ResNet = 2073 features
        
        Args:
            mask: Binary mask
            maskImg: Masked image
            depth_image: Depth image (optional)
            
        Returns:
            Combined feature vector (2073 dim)
        """
        manual_features = self.extract_manual_features(mask, depth_image)
        resnet_features = self.extract_resnet_features(maskImg)
        
        # Combine: 25 manual + 2048 ResNet = 2073
        all_features = np.concatenate([manual_features, resnet_features])
        
        return all_features

