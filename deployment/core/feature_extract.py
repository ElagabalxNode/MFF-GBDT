"""
Feature extraction module for MVP inference pipeline.

Extracts 25 hand-crafted features (with MinMaxScaler normalization) +
2048 auto features from FusionNet backbone.
"""

import sys
import os
import logging
import numpy as np
import joblib
import torch
from pathlib import Path

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
        elif manual_config.get('source') == 'local':
            local_path = manual_config.get('local_path')
            if not local_path:
                raise ValueError(
                    "Local path for scaler not configured. "
                    "Set features.manual.scaler.local_path in config.yaml"
                )
            
            # Resolve path: if relative, assume relative to deployment/ directory
            deployment_dir = Path(__file__).parent.parent
            scaler_path = Path(local_path)
            if not scaler_path.is_absolute():
                scaler_path = deployment_dir / scaler_path
            
            if not scaler_path.exists():
                raise FileNotFoundError(
                    f"Scaler file not found: {scaler_path}. "
                    f"Check features.manual.scaler.local_path in config.yaml"
                )
            
            self.logger.info(f"Loading MinMaxScaler from local file: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            self.logger.info("MinMaxScaler loaded successfully")
        else:
            raise ValueError(
                f"Unknown scaler source: {manual_config.get('source')}. "
                "Must be 'mlflow' or 'local'."
            )
        
        # Load FusionNet for auto features
        auto_config = features_config.get('auto', {}).get('weights', {})
        auto_source = auto_config.get('source', 'mlflow')
        
        if auto_source == 'registry':
            model_name = auto_config.get('model_name')
            version = auto_config.get('version')
            stage = auto_config.get('stage')
            
            # Convert null/None to None for proper handling
            if stage is None or stage == "null" or stage == "":
                stage = None
            if version is None or version == "null" or version == "":
                version = None
            
            if not model_name:
                raise ValueError(
                    "Model name for FusionNet not configured. "
                    "Set features.auto.weights.model_name in config.yaml"
                )
            
            self.logger.info("Loading FusionNet from Model Registry...")
            try:
                fusionnet_model = self.model_loader.load_fusionnet_from_registry(
                    model_name, version=version, stage=stage, device=self.device
                )
                self.fusion_extractor = FusionNetFeatureExtractor(
                    fusionnet_model, device=self.device
                )
                self.logger.info("FusionNet loaded successfully from Model Registry")
            except Exception as e:
                self.logger.warning(
                    f"Failed to load from registry ({e}), "
                    "trying fallback to mlflow or local..."
                )
                # Fallback: try mlflow if run_id available
                run_id = auto_config.get('mlflow_run_id')
                if run_id and run_id != "REPLACE_WITH_FUSION_RUN_ID":
                    artifact_path = auto_config.get('artifact_path', 'best_model.pth')
                    self.logger.info("Trying MLflow fallback...")
                    fusionnet_model = self.model_loader.load_fusionnet(
                        run_id, artifact_path, device=self.device
                    )
                    self.fusion_extractor = FusionNetFeatureExtractor(
                        fusionnet_model, device=self.device
                    )
                    self.logger.info("FusionNet loaded successfully from MLflow (fallback)")
                else:
                    # Last resort: try local if path available
                    local_path = auto_config.get('local_path')
                    if local_path:
                        self.logger.info("Trying local fallback...")
                        deployment_dir = Path(__file__).parent.parent
                        model_path = Path(local_path)
                        if not model_path.is_absolute():
                            model_path = deployment_dir / model_path
                        
                        if model_path.exists():
                            from training_workspace.features.models.FusonNet import fusonnet50
                            
                            # Load model from file
                            # Use weights_only=False because model may contain full object (not just state_dict)
                            # This is safe as we trust the source (local file)
                            # For cross-device compatibility, load to CPU first if target is MPS/CPU
                            # (models saved on CUDA may not map directly to MPS)
                            if self.device in ('mps', 'cpu') or not torch.cuda.is_available():
                                map_loc = 'cpu'  # Load to CPU first for cross-device compatibility
                            else:
                                map_loc = self.device
                            loaded_obj = torch.load(model_path, map_location=map_loc, weights_only=False)
                            
                            # Handle different save formats:
                            # 1. Full model object (nn.Module)
                            # 2. State dict (dict)
                            # 3. Checkpoint dict with 'model_state_dict' key
                            if isinstance(loaded_obj, torch.nn.Module):
                                # Full model object - use it directly
                                model = loaded_obj
                            elif isinstance(loaded_obj, dict):
                                # Check if it's a checkpoint dict or state_dict
                                if 'model_state_dict' in loaded_obj:
                                    state_dict = loaded_obj['model_state_dict']
                                elif 'state_dict' in loaded_obj:
                                    state_dict = loaded_obj['state_dict']
                                else:
                                    # Assume it's a state_dict
                                    state_dict = loaded_obj
                                
                                # Create model and load state_dict
                                model = fusonnet50()
                                model.load_state_dict(state_dict)
                            else:
                                raise TypeError(
                                    f"Unexpected type loaded from {model_path}: {type(loaded_obj)}. "
                                    "Expected nn.Module or dict (state_dict/checkpoint)."
                                )
                            
                            model.eval()
                            model = model.to(self.device)
                            
                            self.fusion_extractor = FusionNetFeatureExtractor(
                                model, device=self.device
                            )
                            self.logger.info("FusionNet loaded successfully from local (fallback)")
                        else:
                            raise RuntimeError(
                                f"All loading methods failed. Registry error: {e}. "
                                f"Local path not found: {model_path}"
                            )
                    else:
                        raise RuntimeError(
                            f"Cannot load FusionNet from registry: {e}. "
                            "No fallback options available (mlflow run_id or local_path)."
                        )
            
        elif auto_source == 'mlflow':
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
            
        elif auto_source == 'local':
            local_path = auto_config.get('local_path')
            if not local_path:
                raise ValueError(
                    "Local path for FusionNet not configured. "
                    "Set features.auto.weights.local_path in config.yaml"
                )
            
            # Resolve path: if relative, assume relative to deployment/ directory
            deployment_dir = Path(__file__).parent.parent
            model_path = Path(local_path)
            if not model_path.is_absolute():
                model_path = deployment_dir / model_path
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"FusionNet model file not found: {model_path}. "
                    f"Check features.auto.weights.local_path in config.yaml"
                )
            
            from training_workspace.features.models.FusonNet import fusonnet50
            
            self.logger.info(f"Loading FusionNet from local file: {model_path}")
            # Load model from file
            # Use weights_only=False because model may contain full object (not just state_dict)
            # This is safe as we trust the source (local file)
            # For cross-device compatibility, load to CPU first if target is MPS/CPU
            # (models saved on CUDA may not map directly to MPS)
            if self.device in ('mps', 'cpu') or not torch.cuda.is_available():
                map_loc = 'cpu'  # Load to CPU first for cross-device compatibility
            else:
                map_loc = self.device
            loaded_obj = torch.load(model_path, map_location=map_loc, weights_only=False)
            
            # Handle different save formats:
            # 1. Full model object (nn.Module)
            # 2. State dict (dict)
            # 3. Checkpoint dict with 'model_state_dict' key
            if isinstance(loaded_obj, torch.nn.Module):
                # Full model object - use it directly
                model = loaded_obj
            elif isinstance(loaded_obj, dict):
                # Check if it's a checkpoint dict or state_dict
                if 'model_state_dict' in loaded_obj:
                    state_dict = loaded_obj['model_state_dict']
                elif 'state_dict' in loaded_obj:
                    state_dict = loaded_obj['state_dict']
                else:
                    # Assume it's a state_dict
                    state_dict = loaded_obj
                
                # Create model and load state_dict
                model = fusonnet50()
                model.load_state_dict(state_dict)
            else:
                raise TypeError(
                    f"Unexpected type loaded from {model_path}: {type(loaded_obj)}. "
                    "Expected nn.Module or dict (state_dict/checkpoint)."
                )
            
            model.eval()
            model = model.to(self.device)
            
            self.fusion_extractor = FusionNetFeatureExtractor(model, device=self.device)
            self.logger.info("FusionNet loaded successfully from local file")
        else:
            raise ValueError(
                f"Unknown FusionNet source: {auto_source}. "
                "Must be 'registry', 'mlflow', or 'local'."
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

