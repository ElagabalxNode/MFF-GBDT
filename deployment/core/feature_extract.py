"""
Feature extraction module for MVP inference pipeline.

Extracts 25 hand-crafted features (with MinMaxScaler normalization) +
2048 auto features from FusionNet backbone.
"""

import sys
import os
import logging
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deployment.core.features import extract_manual_features
from deployment.core.fusion_features import FusionNetFeatureExtractor
from deployment.utils.load_model_from_mlflow import ModelLoader


class FeatureExtractor:
    """
    Extract features for GBDT model: 25 manual (normalized) + 2048 auto.
    
    Features:
    - Manual: 25 hand-crafted 2D/3D features, normalized with MinMaxScaler
    - Auto: 2048-dim vector from FusionNet backbone
    """
    
    def __init__(self, config: dict, device: str = None):
        """
        Initialize feature extractor with MLflow models.
        
        Args:
            config: Configuration dict with MLflow settings
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Device setup
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Initialize ModelLoader
        mlflow_config = config.get('mlflow', {})
        tracking_uri = mlflow_config.get('tracking_uri', 'http://localhost:5000')
        cache_dir = mlflow_config.get('cache_dir', 'models_cache')
        self.model_loader = ModelLoader(tracking_uri=tracking_uri, cache_dir=cache_dir)
        
        # Load MinMaxScaler for manual features
        features_config = config.get('features', {})
        manual_config = features_config.get('manual', {}).get('scaler', {})
        
        if manual_config.get('source') == 'mlflow':
            run_id = manual_config.get('mlflow_run_id')
            artifact_path = manual_config.get('artifact_path', 'feature_scaler.pkl')
            
            if not run_id or run_id == "REPLACE_WITH_SCALER_RUN_ID":
                raise ValueError(
                    "MLflow run_id for scaler not configured. "
                    "Set features.manual.scaler.mlflow_run_id in config.yaml"
                )
            
            self.logger.info("Loading MinMaxScaler from MLflow...")
            self.scaler = self.model_loader.load_scaler(run_id, artifact_path)
            self.logger.info("MinMaxScaler loaded successfully")
        else:
            raise ValueError(
                "Scaler source must be 'mlflow'. "
                "Local scaler loading not implemented."
            )
        
        # Load FusionNet for auto features
        auto_config = features_config.get('auto', {}).get('weights', {})
        
        if auto_config.get('source') == 'mlflow':
            run_id = auto_config.get('mlflow_run_id')
            artifact_path = auto_config.get('artifact_path', 'best_model.pth')
            
            if not run_id or run_id == "REPLACE_WITH_FUSION_RUN_ID":
                raise ValueError(
                    "MLflow run_id for FusionNet not configured. "
                    "Set features.auto.weights.mlflow_run_id in config.yaml"
                )
            
            self.logger.info("Loading FusionNet from MLflow...")
            fusionnet_model = self.model_loader.load_fusionnet(
                run_id, artifact_path, device=self.device
            )
            self.fusion_extractor = FusionNetFeatureExtractor(
                fusionnet_model, device=self.device
            )
            self.logger.info("FusionNet loaded successfully")
        else:
            raise ValueError(
                "FusionNet source must be 'mlflow'. "
                "Local model loading not implemented."
            )
    
    def extract_manual_features(self, mask: np.ndarray, depth_image_z16: np.ndarray = None) -> np.ndarray:
        """
        Extract 25 hand-crafted features and apply MinMaxScaler normalization.
        
        Args:
            mask: Binary mask (uint8, 0 or 255), shape (H, W)
            depth_image_z16: Depth image in Z16 format (int16, mm), shape (H, W)
            
        Returns:
            Normalized array of 25 features
        """
        # Extract raw features
        raw_features = extract_manual_features(mask, depth_image_z16)
        
        # Apply MinMaxScaler (fitted on training data)
        # CRITICAL: Use transform, NOT fit_transform!
        normalized_features = self.scaler.transform(raw_features.reshape(1, -1))
        
        return normalized_features.flatten()
    
    def extract_auto_features(self, maskImg: np.ndarray) -> np.ndarray:
        """
        Extract 2048-dim auto features from FusionNet backbone.
        
        Args:
            maskImg: Masked image (RGB, numpy array, uint8), shape (H, W, 3)
                    Bird on black background
            
        Returns:
            Array of 2048 features
        """
        return self.fusion_extractor.extract_features(maskImg)
    
    def extract_all_features(self, mask: np.ndarray, maskImg: np.ndarray,
                            depth_image_z16: np.ndarray = None) -> np.ndarray:
        """
        Extract all features: 25 manual (normalized) + 2048 auto = 2073 features.
        
        Args:
            mask: Binary mask (uint8, 0 or 255)
            maskImg: Masked image (RGB, uint8)
            depth_image_z16: Depth image in Z16 format (int16, mm)
            
        Returns:
            Combined feature vector (2073 dim)
        """
        manual_features = self.extract_manual_features(mask, depth_image_z16)
        auto_features = self.extract_auto_features(maskImg)
        
        # Combine: 25 manual + 2048 auto = 2073
        all_features = np.concatenate([manual_features, auto_features])
        
        return all_features

