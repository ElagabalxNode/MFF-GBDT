"""
Weight prediction module for MVP inference
Uses trained LightGBM/XGBoost model to predict broiler weight from features
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pickle
import numpy as np
import pandas as pd


class WeightPredictor:
    """GBDT-based weight prediction"""
    
    def __init__(self, model_path: str, model_type: str = 'lightgbm'):
        """
        Initialize weight predictor
        
        Args:
            model_path: Path to trained GBDT model (.pkl file)
            model_type: 'lightgbm' or 'xgboost'
        """
        self.model_type = model_type
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"GBDT model loaded from {model_path}")
        print(f"Model type: {model_type}")
    
    def predict(self, features: np.ndarray) -> float:
        """
        Predict weight from feature vector
        
        Args:
            features: Feature vector (2073 dim: 25 manual + 2048 ResNet)
            
        Returns:
            Predicted weight in kg
        """
        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(features)
        
        # Return single value
        if isinstance(prediction, np.ndarray):
            return float(prediction[0])
        return float(prediction)
    
    def predict_batch(self, features_list: list) -> list:
        """
        Predict weights for multiple instances
        
        Args:
            features_list: List of feature vectors
            
        Returns:
            List of predicted weights
        """
        features_array = np.array(features_list)
        predictions = self.model.predict(features_array)
        return predictions.tolist()

