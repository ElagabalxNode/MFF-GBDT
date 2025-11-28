"""
Weight prediction module for MVP inference pipeline.

Uses trained LightGBM model to predict broiler weight from features.
Supports optional PCA transformation for auto features.
"""

import sys
import os
import logging
import numpy as np

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
            if scaler_config.get('source') == 'mlflow':
                run_id = scaler_config.get('mlflow_run_id')
                artifact_path = scaler_config.get('artifact_path', 'models/auto_scaler.pkl')
                
                if not run_id or run_id == "REPLACE_WITH_PCA_RUN_ID":
                    raise ValueError(
                        "MLflow run_id for PCA scaler not configured. "
                        "Set features.pca.scaler.mlflow_run_id in config.yaml"
                    )
                
                self.auto_scaler = self.model_loader.load_scaler(run_id, artifact_path)
                self.logger.info("Auto scaler loaded for PCA")
            
            # Load PCA model
            pca_model_config = pca_config.get('model', {})
            if pca_model_config.get('source') == 'mlflow':
                run_id = pca_model_config.get('mlflow_run_id')
                artifact_path = pca_model_config.get('artifact_path', 'models/auto_pca.pkl')
                
                if not run_id or run_id == "REPLACE_WITH_PCA_RUN_ID":
                    raise ValueError(
                        "MLflow run_id for PCA model not configured. "
                        "Set features.pca.model.mlflow_run_id in config.yaml"
                    )
                
                self.pca = self.model_loader.load_pca(run_id, artifact_path)
                self.logger.info(f"PCA model loaded: {self.pca.n_components_} components")
        
        # Load LightGBM model
        gbdt_config = config.get('gbdt', {}).get('model', {})
        if gbdt_config.get('source') == 'mlflow':
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
        else:
            raise ValueError(
                "LightGBM source must be 'mlflow'. "
                "Local model loading not implemented."
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
            return auto_features
        
        # Standardize auto features
        auto_scaled = self.auto_scaler.transform(auto_features.reshape(1, -1))
        
        # Apply PCA
        auto_pca = self.pca.transform(auto_scaled)
        
        return auto_pca.flatten()
    
    def predict(self, features: np.ndarray) -> float:
        """
        Predict weight from feature vector.
        
        Args:
            features: Feature vector [25 Manual Normalized + 2048 Auto Raw]
            
        Returns:
            Predicted weight in kg
        """
        # Split features into manual and auto
        manual_features = features[:25]  # First 25: manual (already normalized)
        auto_features = features[25:]    # Rest: auto (2048 dim)
        
        # Apply PCA to auto features if enabled
        if self.pca_enabled:
            auto_transformed = self._apply_pca(auto_features)
            # Combine: manual + PCA'd auto
            final_features = np.concatenate([manual_features, auto_transformed])
        else:
            # Use raw features: manual + auto
            final_features = features
        
        # Ensure correct shape for prediction
        if final_features.ndim == 1:
            final_features = final_features.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(final_features)
        
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

