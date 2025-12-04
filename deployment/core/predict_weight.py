"""
Weight prediction module for MVP inference pipeline.

Uses trained LightGBM model to predict broiler weight from features.
Supports optional PCA transformation for auto features.
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path
import joblib
import pickle

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deployment.utils.load_model_from_mlflow import ModelLoader


class WeightPredictor:
    """
    LightGBM-based weight prediction with optional PCA for auto features.
    
    Features:
    - Loads LightGBM model from MLflow
    - Optionally applies PCA to auto features (if enabled in config)
    - Predicts weight from [25 Manual Normalized + (PCA'd Auto or Raw Auto)]
    """
    
    def __init__(self, config: dict):
        """
        Initialize weight predictor with MLflow models.
        
        Args:
            config: Configuration dict with MLflow settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize ModelLoader
        mlflow_config = config.get('mlflow', {})
        tracking_uri = mlflow_config.get('tracking_uri', 'http://localhost:5000')
        cache_dir = mlflow_config.get('cache_dir', 'models_cache')
        self.model_loader = ModelLoader(tracking_uri=tracking_uri, cache_dir=cache_dir)
        
        # Check if PCA is enabled
        features_config = config.get('features', {})
        pca_config = features_config.get('pca', {})
        self.pca_enabled = pca_config.get('enabled', False)
        
        # Load PCA components if enabled
        self.auto_scaler = None
        self.pca = None
        
        if self.pca_enabled:
            self.logger.info("PCA enabled - loading auto_scaler and PCA model...")
            
            # Load auto_scaler for PCA
            scaler_config = pca_config.get('scaler', {})
            scaler_source = scaler_config.get('source', 'mlflow')
            
            if scaler_source == 'mlflow':
                run_id = scaler_config.get('mlflow_run_id')
                artifact_path = scaler_config.get('artifact_path', 'models/auto_scaler.pkl')
                
                if not run_id or run_id == "REPLACE_WITH_PCA_RUN_ID":
                    raise ValueError(
                        "MLflow run_id for PCA scaler not configured. "
                        "Set features.pca.scaler.mlflow_run_id in config.yaml"
                    )
                
                self.auto_scaler = self.model_loader.load_scaler(run_id, artifact_path)
                self.logger.info("Auto scaler loaded for PCA")
            elif scaler_source == 'local':
                local_path = scaler_config.get('local_path')
                if not local_path:
                    raise ValueError(
                        "Local path for PCA scaler not configured. "
                        "Set features.pca.scaler.local_path in config.yaml"
                    )
                
                # Resolve path: if relative, assume relative to deployment/ directory
                deployment_dir = Path(__file__).parent.parent
                scaler_path = Path(local_path)
                if not scaler_path.is_absolute():
                    scaler_path = deployment_dir / scaler_path
                
                if not scaler_path.exists():
                    raise FileNotFoundError(
                        f"PCA scaler file not found: {scaler_path}. "
                        f"Check features.pca.scaler.local_path in config.yaml"
                    )
                
                self.logger.info(f"Loading PCA scaler from local file: {scaler_path}")
                self.auto_scaler = joblib.load(scaler_path)
                self.logger.info("Auto scaler loaded for PCA")
            else:
                raise ValueError(
                    f"Unknown PCA scaler source: {scaler_source}. "
                    "Must be 'mlflow' or 'local'."
                )
            
            # Load PCA model
            pca_model_config = pca_config.get('model', {})
            pca_model_source = pca_model_config.get('source', 'mlflow')
            
            if pca_model_source == 'mlflow':
                run_id = pca_model_config.get('mlflow_run_id')
                artifact_path = pca_model_config.get('artifact_path', 'models/auto_pca.pkl')
                
                if not run_id or run_id == "REPLACE_WITH_PCA_RUN_ID":
                    raise ValueError(
                        "MLflow run_id for PCA model not configured. "
                        "Set features.pca.model.mlflow_run_id in config.yaml"
                    )
                
                self.pca = self.model_loader.load_pca(run_id, artifact_path)
                self.logger.info(f"PCA model loaded: {self.pca.n_components_} components")
            elif pca_model_source == 'local':
                local_path = pca_model_config.get('local_path')
                if not local_path:
                    raise ValueError(
                        "Local path for PCA model not configured. "
                        "Set features.pca.model.local_path in config.yaml"
                    )
                
                # Resolve path: if relative, assume relative to deployment/ directory
                deployment_dir = Path(__file__).parent.parent
                pca_path = Path(local_path)
                if not pca_path.is_absolute():
                    pca_path = deployment_dir / pca_path
                
                if not pca_path.exists():
                    raise FileNotFoundError(
                        f"PCA model file not found: {pca_path}. "
                        f"Check features.pca.model.local_path in config.yaml"
                    )
                
                self.logger.info(f"Loading PCA model from local file: {pca_path}")
                self.pca = joblib.load(pca_path)
                n_components = getattr(self.pca, 'n_components_', None)
                if n_components is None:
                    n_components = getattr(self.pca, 'n_components', None)
                self.logger.info(
                    f"PCA model loaded: {n_components} components "
                    f"(expected 50 for LGBM_PCA50_Optimized)"
                )
                if n_components != 50:
                    self.logger.warning(
                        f"PCA has {n_components} components, but model expects 50. "
                        "This may cause feature mismatch!"
                    )
            else:
                raise ValueError(
                    f"Unknown PCA model source: {pca_model_source}. "
                    "Must be 'mlflow' or 'local'."
                )
        
        # Load LightGBM model
        gbdt_config = config.get('gbdt', {}).get('model', {})
        gbdt_source = gbdt_config.get('source', 'mlflow')
        
        if gbdt_source == 'registry':
            model_name = gbdt_config.get('model_name')
            version = gbdt_config.get('version')
            stage = gbdt_config.get('stage')
            load_method = gbdt_config.get('load_method', 'mlflow')
            
            if not model_name:
                raise ValueError(
                    "Model name for LightGBM not configured. "
                    "Set gbdt.model.model_name in config.yaml"
                )
            
            self.logger.info("Loading LightGBM model from Model Registry...")
            self.model = self.model_loader.load_lightgbm_from_registry(
                model_name, version=version, stage=stage, load_method=load_method
            )
            n_features = getattr(self.model, 'n_features_in_', None)
            self.logger.info(
                f"LightGBM model loaded successfully from Model Registry. "
                f"Expected features: {n_features} (should be 75 for PCA50 model: 25 manual + 50 PCA)"
            )
            if n_features and n_features != 75:
                self.logger.warning(
                    f"Model expects {n_features} features, but should be 75 for PCA50 model. "
                    "Check if correct model is loaded!"
                )
            
        elif gbdt_source == 'mlflow':
            run_id = gbdt_config.get('mlflow_run_id')
            model_uri = gbdt_config.get('model_uri', 'model')
            load_method = gbdt_config.get('load_method', 'mlflow')
            
            if not run_id or run_id == "REPLACE_WITH_LGBM_RUN_ID":
                raise ValueError(
                    "MLflow run_id for LightGBM not configured. "
                    "Set gbdt.model.mlflow_run_id in config.yaml"
                )
            
            self.logger.info("Loading LightGBM model from MLflow...")
            self.model = self.model_loader.load_lightgbm(
                run_id, model_uri, load_method
            )
            self.logger.info("LightGBM model loaded successfully")
            
        elif gbdt_source == 'local':
            local_path = gbdt_config.get('local_path')
            if not local_path:
                raise ValueError(
                    "Local path for LightGBM not configured. "
                    "Set gbdt.model.local_path in config.yaml"
                )
            
            # Resolve path: if relative, assume relative to deployment/ directory
            deployment_dir = Path(__file__).parent.parent
            model_path = Path(local_path)
            if not model_path.is_absolute():
                model_path = deployment_dir / model_path
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"LightGBM model file not found: {model_path}. "
                    f"Check gbdt.model.local_path in config.yaml"
                )
            
            self.logger.info(f"Loading LightGBM from local file: {model_path}")
            try:
                self.model = joblib.load(model_path)
            except Exception:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            self.logger.info("LightGBM model loaded successfully from local file")
        else:
            raise ValueError(
                f"Unknown LightGBM source: {gbdt_source}. "
                "Must be 'registry', 'mlflow', or 'local'."
            )
    
    def _apply_pca(self, auto_features: np.ndarray) -> np.ndarray:
        """
        Apply PCA transformation to auto features.
        
        Args:
            auto_features: Raw auto features (2048 dim)
            
        Returns:
            PCA-transformed features (N dim, usually 50)
        """
        if not self.pca_enabled or self.auto_scaler is None or self.pca is None:
            self.logger.warning("PCA enabled but components not loaded, returning raw features")
            return auto_features
        
        # Check input shape
        if len(auto_features) != 2048:
            self.logger.warning(
                f"Expected 2048 auto features, got {len(auto_features)}. "
                "This may cause issues."
            )
        
        # Standardize auto features
        auto_scaled = self.auto_scaler.transform(auto_features.reshape(1, -1))
        
        # Apply PCA
        auto_pca = self.pca.transform(auto_scaled)
        
        # Check output shape
        expected_components = getattr(self.pca, 'n_components_', None)
        if expected_components and auto_pca.shape[1] != expected_components:
            self.logger.warning(
                f"PCA returned {auto_pca.shape[1]} components, "
                f"expected {expected_components}"
            )
        
        result = auto_pca.flatten()
        self.logger.info(
            f"PCA: {len(auto_features)} -> {len(result)} components"
        )
        return result
    
    def predict(self, features: np.ndarray) -> float:
        """
        Predict weight from feature vector.
        
        Args:
            features: Feature vector [25 Manual Normalized + 2048 Auto Raw]
            
        Returns:
            Predicted weight in kg
        """
        # Log input features shape
        self.logger.info(f"Input features shape: {features.shape}, length: {len(features)}")
        
        # Split features into manual and auto
        manual_features = features[:25]  # First 25: manual (already normalized)
        auto_features = features[25:]    # Rest: auto (2048 dim)
        
        self.logger.info(
            f"Split: {len(manual_features)} manual + {len(auto_features)} auto"
        )
        
        # Apply PCA to auto features if enabled
        if self.pca_enabled:
            if self.auto_scaler is None or self.pca is None:
                raise RuntimeError(
                    "PCA is enabled but auto_scaler or PCA model is not loaded. "
                    "Check features.pca configuration in config.yaml"
                )
            
            # Log PCA info
            pca_components = getattr(self.pca, 'n_components_', 'unknown')
            self.logger.info(
                f"PCA enabled: {pca_components} components expected"
            )
            
            auto_transformed = self._apply_pca(auto_features)
            
            # Combine: manual + PCA'd auto
            final_features = np.concatenate([manual_features, auto_transformed])
            
            self.logger.info(
                f"Features: {len(manual_features)} manual + "
                f"{len(auto_transformed)} PCA = {len(final_features)} total "
                f"(model expects {self.model.n_features_in_} features)"
            )
        else:
            # Use raw features: manual + auto
            final_features = features
            self.logger.debug(
                f"Features: {len(features)} raw (no PCA)"
            )
        
        # Ensure correct shape for prediction
        if final_features.ndim == 1:
            final_features = final_features.reshape(1, -1)
        
        # Log feature count for debugging
        self.logger.info(
            f"Predicting with {final_features.shape[1]} features "
            f"(model expects {self.model.n_features_in_} features)"
        )
        
        # Predict
        prediction = self.model.predict(final_features)
        
        # Flush logs to ensure they're written to file
        for handler in self.logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        # Return single value
        if isinstance(prediction, np.ndarray):
            return float(prediction[0])
        return float(prediction)
    
    def predict_batch(self, features_list: list) -> list:
        """
        Predict weights for multiple instances.
        
        Args:
            features_list: List of feature vectors
            
        Returns:
            List of predicted weights
        """
        predictions = []
        for features in features_list:
            weight = self.predict(features)
            predictions.append(weight)
        return predictions

