"""
FusionNet auto feature extraction module.

Extracts 2048-dim feature vector from FusionNet backbone.
"""

import sys
import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training_workspace.features.models.FusonNet import fusonnet50


class FusionNetFeatureExtractor:
    """
    Extract 2048-dim auto features from FusionNet backbone.
    
    Uses only the ResNet backbone part (up to avgpool), without
    the classification head (fc layers).
    """
    
    def __init__(self, model, device: str = None):
        """
        Initialize FusionNet feature extractor.
        
        Args:
            model: Loaded FusionNet model (from MLflow)
            device: Device to run inference on ('cuda', 'cpu', or None)
        """
        self.model = model
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing: same as training (Resize to 360x640)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((360, 640))
        ])
    
    def extract_features(self, maskImg: np.ndarray) -> np.ndarray:
        """
        Extract 2048-dim auto features from masked RGB image.
        
        Args:
            maskImg: Masked image (RGB, numpy array, uint8), shape (H, W, 3)
                    Bird on black background
        
        Returns:
            numpy array of 2048 features
        """
        # Convert numpy array to PIL Image
        if maskImg.dtype != np.uint8:
            maskImg = (maskImg * 255).astype(np.uint8)
        
        # Ensure RGB format
        if len(maskImg.shape) == 3 and maskImg.shape[2] == 3:
            # Already RGB
            pil_img = Image.fromarray(maskImg)
        else:
            # Convert BGR to RGB if needed
            pil_img = Image.fromarray(cv2.cvtColor(maskImg, cv2.COLOR_BGR2RGB))
        
        # Preprocess
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Extract features using FusionNet backbone
        # Forward through backbone only (up to avgpool)
        with torch.no_grad():
            x = self.model.conv1(img_tensor)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            
            # x is now 2048-dim feature vector
            auto_features = x.squeeze().cpu().numpy()
        
        return auto_features

