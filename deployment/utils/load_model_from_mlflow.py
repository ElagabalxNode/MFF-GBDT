"""
Model loading utilities for deployment pipeline

This module provides:
1. ModelLoader class for downloading and caching models from MLflow
2. Functions for loading models by run_id or from Model Registry
"""

import sys
import os
import argparse
import logging
import hashlib
import torch
from torchvision import transforms
import mlflow
import mlflow.pytorch
from pathlib import Path

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class ModelLoader:
    """
    Model loader with MLflow integration and local caching.
    
    Implements download-and-cache strategy:
    - Downloads artifacts from MLflow to cache_dir
    - Uses cached files if MLflow is unavailable (offline mode)
    - Generates unique cache keys based on run_id and artifact_path
    """
    
    def __init__(self, tracking_uri: str, cache_dir: str):
        """
        Initialize ModelLoader.
        
        Args:
            tracking_uri: MLflow tracking URI
            cache_dir: Directory for caching downloaded models (relative to deployment/ or absolute)
        """
        self.tracking_uri = tracking_uri
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Resolve cache directory path
        deployment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.isabs(cache_dir):
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(deployment_dir) / cache_dir
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        self.logger.info(f"ModelLoader initialized: cache_dir={self.cache_dir}, tracking_uri={tracking_uri}")
    
    def _get_cache_key(self, run_id: str, artifact_path: str) -> str:
        """
        Generate cache key from run_id and artifact_path.
        
        Args:
            run_id: MLflow run ID
            artifact_path: Path to artifact within run
            
        Returns:
            Cache key (filename-safe hash)
        """
        key_string = f"{run_id}:{artifact_path}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, run_id: str, artifact_path: str) -> Path:
        """
        Get local cache path for artifact.
        
        Args:
            run_id: MLflow run ID
            artifact_path: Path to artifact within run
            
        Returns:
            Path to cached file
        """
        cache_key = self._get_cache_key(run_id, artifact_path)
        # Preserve file extension from artifact_path
        ext = Path(artifact_path).suffix or '.pt'
        return self.cache_dir / f"{cache_key}{ext}"
    
    def load_yolo_model(self, run_id: str, artifact_path: str) -> str:
        """
        Load YOLO model from MLflow with caching.
        
        Downloads .pt file from MLflow artifacts and caches it locally.
        If MLflow is unavailable but file exists in cache, uses cached version.
        
        Args:
            run_id: MLflow run ID
            artifact_path: Path to .pt file within MLflow artifacts (e.g., "weights/best.pt")
            
        Returns:
            Path to local .pt file (can be used with YOLO(path))
            
        Raises:
            FileNotFoundError: If model not found in MLflow and not in cache
            RuntimeError: If MLflow connection fails and no cache available
        """
        cache_path = self._get_cache_path(run_id, artifact_path)
        
        # Check if file exists in cache
        if cache_path.exists():
            self.logger.info(f"Using cached model: {cache_path}")
            return str(cache_path)
        
        # Try to download from MLflow
        try:
            self.logger.info(f"Downloading YOLO model from MLflow: run_id={run_id}, artifact_path={artifact_path}")
            artifact_uri = f"runs:/{run_id}/{artifact_path}"
            
            # Download artifact to temporary directory
            temp_dir = mlflow.artifacts.download_artifacts(artifact_uri, dst_path=str(self.cache_dir))
            
            # Find the downloaded file
            temp_path = Path(temp_dir)
            if temp_path.is_file():
                # Single file downloaded
                downloaded_file = temp_path
            else:
                # Directory downloaded, find the file
                artifact_name = Path(artifact_path).name
                downloaded_file = temp_path / artifact_name
                if not downloaded_file.exists():
                    # Try to find any .pt file in the directory
                    pt_files = list(temp_path.glob("*.pt"))
                    if pt_files:
                        downloaded_file = pt_files[0]
                    else:
                        raise FileNotFoundError(f"Could not find .pt file in downloaded artifacts: {temp_dir}")
            
            # Copy to cache location
            import shutil
            shutil.copy2(downloaded_file, cache_path)
            self.logger.info(f"Model cached to: {cache_path}")
            
            return str(cache_path)
            
        except Exception as e:
            # If download fails, check if we have a cached version
            if cache_path.exists():
                self.logger.warning(
                    f"MLflow download failed ({e}), but using cached model: {cache_path}"
                )
                return str(cache_path)
            else:
                self.logger.error(f"Failed to load model from MLflow and no cache available: {e}")
                raise RuntimeError(
                    f"Cannot load YOLO model: MLflow unavailable and no cached version. "
                    f"Error: {e}"
                )
    
    def load_scaler(self, run_id: str, artifact_path: str):
        """
        Load scaler (MinMaxScaler, etc.) from MLflow with caching.
        
        Downloads .pkl file from MLflow artifacts and caches it locally.
        If MLflow is unavailable but file exists in cache, uses cached version.
        
        Args:
            run_id: MLflow run ID
            artifact_path: Path to .pkl file within MLflow artifacts (e.g., "feature_scaler.pkl")
            
        Returns:
            Loaded scaler object (e.g., MinMaxScaler)
            
        Raises:
            FileNotFoundError: If scaler not found in MLflow and not in cache
            RuntimeError: If MLflow connection fails and no cache available
        """
        import joblib
        
        cache_path = self._get_cache_path(run_id, artifact_path)
        
        # Check if file exists in cache
        if cache_path.exists():
            self.logger.info(f"Using cached scaler: {cache_path}")
            return joblib.load(cache_path)
        
        # Try to download from MLflow
        try:
            self.logger.info(
                f"Downloading scaler from MLflow: "
                f"run_id={run_id}, artifact_path={artifact_path}"
            )
            artifact_uri = f"runs:/{run_id}/{artifact_path}"
            
            # Download artifact to temporary directory
            temp_dir = mlflow.artifacts.download_artifacts(
                artifact_uri, dst_path=str(self.cache_dir)
            )
            
            # Find the downloaded file
            temp_path = Path(temp_dir)
            if temp_path.is_file():
                downloaded_file = temp_path
            else:
                artifact_name = Path(artifact_path).name
                downloaded_file = temp_path / artifact_name
                if not downloaded_file.exists():
                    # Try to find any .pkl file in the directory
                    pkl_files = list(temp_path.glob("*.pkl"))
                    if pkl_files:
                        downloaded_file = pkl_files[0]
                    else:
                        raise FileNotFoundError(
                            f"Could not find .pkl file in downloaded artifacts: {temp_dir}"
                        )
            
            # Copy to cache location
            import shutil
            shutil.copy2(downloaded_file, cache_path)
            self.logger.info(f"Scaler cached to: {cache_path}")
            
            # Load and return
            return joblib.load(cache_path)
            
        except Exception as e:
            # If download fails, check if we have a cached version
            if cache_path.exists():
                self.logger.warning(
                    f"MLflow download failed ({e}), but using cached scaler: {cache_path}"
                )
                return joblib.load(cache_path)
            else:
                self.logger.error(
                    f"Failed to load scaler from MLflow and no cache available: {e}"
                )
                raise RuntimeError(
                    f"Cannot load scaler: MLflow unavailable and no cached version. "
                    f"Error: {e}"
                )
    
    def load_fusionnet(self, run_id: str, artifact_path: str, device: str = None):
        """
        Load FusionNet model from MLflow with caching.
        
        Downloads .pth file from MLflow artifacts and caches it locally.
        Loads model and returns it ready for inference.
        
        Args:
            run_id: MLflow run ID
            artifact_path: Path to .pth file within MLflow artifacts (e.g., "best_model.pth")
            device: Device to load model on ('cuda', 'cpu', or None for auto-detect)
            
        Returns:
            Loaded FusionNet model (PyTorch nn.Module)
            
        Raises:
            FileNotFoundError: If model not found in MLflow and not in cache
            RuntimeError: If MLflow connection fails and no cache available
        """
        import torch
        
        cache_path = self._get_cache_path(run_id, artifact_path)
        
        # Check if file exists in cache
        if cache_path.exists():
            self.logger.info(f"Using cached FusionNet: {cache_path}")
            model_path = str(cache_path)
        else:
            # Try to download from MLflow
            try:
                self.logger.info(
                    f"Downloading FusionNet from MLflow: "
                    f"run_id={run_id}, artifact_path={artifact_path}"
                )
                artifact_uri = f"runs:/{run_id}/{artifact_path}"
                
                # Download artifact to temporary directory
                temp_dir = mlflow.artifacts.download_artifacts(
                    artifact_uri, dst_path=str(self.cache_dir)
                )
                
                # Find the downloaded file
                temp_path = Path(temp_dir)
                if temp_path.is_file():
                    downloaded_file = temp_path
                else:
                    artifact_name = Path(artifact_path).name
                    downloaded_file = temp_path / artifact_name
                    if not downloaded_file.exists():
                        # Try to find any .pth file in the directory
                        pth_files = list(temp_path.glob("*.pth"))
                        if pth_files:
                            downloaded_file = pth_files[0]
                        else:
                            raise FileNotFoundError(
                                f"Could not find .pth file in downloaded artifacts: {temp_dir}"
                            )
                
                # Copy to cache location
                import shutil
                shutil.copy2(downloaded_file, cache_path)
                self.logger.info(f"FusionNet cached to: {cache_path}")
                model_path = str(cache_path)
                
            except Exception as e:
                # If download fails, check if we have a cached version
                if cache_path.exists():
                    self.logger.warning(
                        f"MLflow download failed ({e}), but using cached FusionNet: {cache_path}"
                    )
                    model_path = str(cache_path)
                else:
                    self.logger.error(
                        f"Failed to load FusionNet from MLflow and no cache available: {e}"
                    )
                    raise RuntimeError(
                        f"Cannot load FusionNet: MLflow unavailable and no cached version. "
                        f"Error: {e}"
                    )
        
        # Load model from file
        # Import FusionNet architecture
        from training_workspace.features.models.FusonNet import fusonnet50
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        # Load model from file
        # Use weights_only=False because model may contain full object (not just state_dict)
        # This is safe as we trust the source (MLflow artifacts)
        # For cross-device compatibility, load to CPU first if target is MPS/CPU
        # (models saved on CUDA may not map directly to MPS)
        if device in ('mps', 'cpu') or not torch.cuda.is_available():
            map_loc = 'cpu'  # Load to CPU first for cross-device compatibility
        else:
            map_loc = device
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
        model = model.to(device)
        
        self.logger.info(f"FusionNet loaded on device: {device}")
        return model
    
    def load_fusionnet_from_registry(self, model_name: str, version: int = None, 
                                    stage: str = None, device: str = None):
        """
        Load FusionNet model from MLflow Model Registry.
        
        Args:
            model_name: Name of the registered model in Model Registry
            version: Specific version number (optional, ignored if stage is provided)
            stage: Model stage - Production, Staging, Archived (optional, takes precedence over version)
            device: Device to load model on ('cuda', 'cpu', or None for auto-detect)
            
        Returns:
            Loaded FusionNet model (PyTorch nn.Module)
            
        Raises:
            RuntimeError: If model cannot be loaded
        """
        import torch
        
        # Try loading with different strategies: stage -> version -> latest
        strategies = []
        if stage:
            strategies.append(("stage", f"models:/{model_name}/{stage}", f"{model_name} (stage: {stage})"))
        if version:
            strategies.append(("version", f"models:/{model_name}/{version}", f"{model_name} (version: {version})"))
        if not strategies:
            strategies.append(("latest", f"models:/{model_name}/latest", f"{model_name} (latest)"))
        
        last_error = None
        for strategy_name, model_uri, log_msg in strategies:
            try:
                self.logger.info(f"Trying to load FusionNet from Model Registry: {log_msg}")
                # Try to load using mlflow.pytorch.load_model first
                # This works if model was registered using mlflow.pytorch.log_model
                model = mlflow.pytorch.load_model(model_uri)
                self.logger.info(f"FusionNet loaded successfully from Model Registry via {strategy_name}")
                
                # Determine device and move model
                if device is None:
                    if torch.cuda.is_available():
                        device = 'cuda'
                    elif torch.backends.mps.is_available():
                        device = 'mps'
                    else:
                        device = 'cpu'
                
                model = model.to(device)
                model.eval()
                self.logger.info(f"FusionNet loaded on device: {device}")
                return model
            except Exception as e:
                last_error = e
                self.logger.warning(f"Failed to load with {strategy_name}: {e}")
                # Continue to next strategy
                continue
        
        # If all strategies failed, try fallback with artifact download
        if last_error:
            self.logger.warning(
                "All registry strategies failed, trying artifact download fallback..."
            )
            # Try artifact download with the last attempted URI
            try:
                # Use the last strategy's URI for artifact download
                last_uri = strategies[-1][1] if strategies else f"models:/{model_name}/latest"
                temp_dir = mlflow.artifacts.download_artifacts(
                    last_uri, dst_path=str(self.cache_dir)
                )
                temp_path = Path(temp_dir)
                
                # Find .pth file
                pth_files = list(temp_path.rglob("*.pth"))
                if not pth_files:
                    raise FileNotFoundError(
                        f"No .pth file found in model artifacts: {temp_dir}"
                    )
                
                model_path = pth_files[0]
                
                # Import FusionNet architecture
                from training_workspace.features.models.FusonNet import fusonnet50
                
                # Determine device
                if device is None:
                    if torch.cuda.is_available():
                        device = 'cuda'
                    elif torch.backends.mps.is_available():
                        device = 'mps'
                    else:
                        device = 'cpu'
                
                # Load model from file
                # Use weights_only=False because model may contain full object (not just state_dict)
                # This is safe as we trust the source (MLflow artifacts)
                # For cross-device compatibility, load to CPU first if target is MPS/CPU
                # (models saved on CUDA may not map directly to MPS)
                if device in ('mps', 'cpu') or not torch.cuda.is_available():
                    map_loc = 'cpu'  # Load to CPU first for cross-device compatibility
                else:
                    map_loc = device
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
                model = model.to(device)
                
                self.logger.info(
                    f"FusionNet loaded on device: {device} (from artifact fallback)"
                )
                return model
                
            except Exception as e2:
                self.logger.error(f"Failed to load FusionNet from Model Registry: {e2}")
                raise RuntimeError(
                    f"Cannot load FusionNet from Model Registry. "
                    f"Tried all strategies and artifact download. "
                    f"Last error: {last_error}, Artifact error: {e2}"
                )
        
        # Should not reach here, but just in case
        raise RuntimeError(
            f"Cannot load FusionNet from Model Registry: {model_name}. "
            "No strategies available."
        )
    
    def load_pca(self, run_id: str, artifact_path: str):
        """
        Load PCA transformer from MLflow with caching.
        
        Downloads .pkl file from MLflow artifacts and caches it locally.
        If MLflow is unavailable but file exists in cache, uses cached version.
        
        Args:
            run_id: MLflow run ID
            artifact_path: Path to .pkl file within MLflow artifacts (e.g., "models/auto_pca.pkl")
            
        Returns:
            Loaded PCA transformer (sklearn.decomposition.PCA)
            
        Raises:
            FileNotFoundError: If PCA not found in MLflow and not in cache
            RuntimeError: If MLflow connection fails and no cache available
        """
        import joblib
        
        cache_path = self._get_cache_path(run_id, artifact_path)
        
        # Check if file exists in cache
        if cache_path.exists():
            self.logger.info(f"Using cached PCA: {cache_path}")
            return joblib.load(cache_path)
        
        # Try to download from MLflow
        try:
            self.logger.info(
                f"Downloading PCA from MLflow: "
                f"run_id={run_id}, artifact_path={artifact_path}"
            )
            artifact_uri = f"runs:/{run_id}/{artifact_path}"
            
            # Download artifact to temporary directory
            temp_dir = mlflow.artifacts.download_artifacts(
                artifact_uri, dst_path=str(self.cache_dir)
            )
            
            # Find the downloaded file
            temp_path = Path(temp_dir)
            if temp_path.is_file():
                downloaded_file = temp_path
            else:
                artifact_name = Path(artifact_path).name
                downloaded_file = temp_path / artifact_name
                if not downloaded_file.exists():
                    # Try to find any .pkl file in the directory
                    pkl_files = list(temp_path.glob("*.pkl"))
                    if pkl_files:
                        downloaded_file = pkl_files[0]
                    else:
                        raise FileNotFoundError(
                            f"Could not find .pkl file in downloaded artifacts: {temp_dir}"
                        )
            
            # Copy to cache location
            import shutil
            shutil.copy2(downloaded_file, cache_path)
            self.logger.info(f"PCA cached to: {cache_path}")
            
            # Load and return
            return joblib.load(cache_path)
            
        except Exception as e:
            # If download fails, check if we have a cached version
            if cache_path.exists():
                self.logger.warning(
                    f"MLflow download failed ({e}), but using cached PCA: {cache_path}"
                )
                return joblib.load(cache_path)
            else:
                self.logger.error(
                    f"Failed to load PCA from MLflow and no cache available: {e}"
                )
                raise RuntimeError(
                    f"Cannot load PCA: MLflow unavailable and no cached version. "
                    f"Error: {e}"
                )
    
    def load_lightgbm_from_registry(self, model_name: str, version: int = None, 
                                   stage: str = None, load_method: str = "mlflow"):
        """
        Load LightGBM model from MLflow Model Registry.
        
        Args:
            model_name: Name of the registered model in Model Registry
            version: Specific version number (optional, ignored if stage is provided)
            stage: Model stage - Production, Staging, Archived (optional, takes precedence over version)
            load_method: "mlflow" to use mlflow.lightgbm.load_model, "sklearn" for pickle files
            
        Returns:
            Loaded LightGBM model
            
        Raises:
            RuntimeError: If model cannot be loaded
        """
        # Construct model URI
        if stage:
            model_uri = f"models:/{model_name}/{stage}"
            self.logger.info(f"Loading LightGBM from Model Registry: {model_name} (stage: {stage})")
        elif version:
            model_uri = f"models:/{model_name}/{version}"
            self.logger.info(f"Loading LightGBM from Model Registry: {model_name} (version: {version})")
        else:
            model_uri = f"models:/{model_name}/latest"
            self.logger.info(f"Loading latest LightGBM from Model Registry: {model_name}")
        
        if load_method == "mlflow":
            # Use MLflow's LightGBM loader
            try:
                import mlflow.lightgbm
                model = mlflow.lightgbm.load_model(model_uri)
                self.logger.info("LightGBM loaded successfully from Model Registry")
                return model
            except Exception as e:
                self.logger.error(f"Failed to load LightGBM from Model Registry: {e}")
                raise RuntimeError(f"Cannot load LightGBM model from Model Registry. Error: {e}")
        else:
            # For sklearn/pickle, we need to download artifacts first
            # This is a workaround - ideally models should be registered with log_model
            try:
                import joblib
                import pickle
                
                # Download model artifacts
                temp_dir = mlflow.artifacts.download_artifacts(model_uri, dst_path=str(self.cache_dir))
                temp_path = Path(temp_dir)
                
                # Try to find model file
                pkl_files = list(temp_path.rglob("*.pkl"))
                if pkl_files:
                    model = joblib.load(pkl_files[0])
                else:
                    # Try pickle
                    with open(temp_path / "model.pkl", 'rb') as f:
                        model = pickle.load(f)
                
                self.logger.info("LightGBM loaded successfully from Model Registry (as pickle)")
                return model
            except Exception as e:
                self.logger.error(f"Failed to load LightGBM from Model Registry: {e}")
                raise RuntimeError(f"Cannot load LightGBM model from Model Registry. Error: {e}")
    
    def load_lightgbm(self, run_id: str, model_uri: str = "model", 
                     load_method: str = "mlflow"):
        """
        Load LightGBM model from MLflow with caching.
        
        Supports two loading methods:
        - "mlflow": Use mlflow.lightgbm.load_model (for models logged with log_model)
        - "sklearn": Load pickle file directly (for models saved as .pkl)
        
        Args:
            run_id: MLflow run ID
            model_uri: URI within run (usually "model" for log_model, or path to .pkl)
            load_method: "mlflow" or "sklearn"
            
        Returns:
            Loaded LightGBM model
            
        Raises:
            RuntimeError: If model cannot be loaded
        """
        if load_method == "mlflow":
            # Use MLflow's LightGBM loader
            try:
                import mlflow.lightgbm
                model_uri_full = f"runs:/{run_id}/{model_uri}"
                self.logger.info(
                    f"Loading LightGBM from MLflow: {model_uri_full}"
                )
                model = mlflow.lightgbm.load_model(model_uri_full)
                self.logger.info("LightGBM loaded successfully via MLflow")
                return model
            except Exception as e:
                self.logger.error(f"Failed to load LightGBM via MLflow: {e}")
                raise RuntimeError(
                    f"Cannot load LightGBM model via MLflow. Error: {e}"
                )
        
        elif load_method == "sklearn":
            # Load as pickle file (artifact)
            import joblib
            import pickle
            
            cache_path = self._get_cache_path(run_id, model_uri)
            
            # Check if file exists in cache
            if cache_path.exists():
                self.logger.info(f"Using cached LightGBM: {cache_path}")
                try:
                    return joblib.load(cache_path)
                except:
                    # Try pickle if joblib fails
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
            
            # Try to download from MLflow
            try:
                self.logger.info(
                    f"Downloading LightGBM from MLflow: "
                    f"run_id={run_id}, model_uri={model_uri}"
                )
                artifact_uri = f"runs:/{run_id}/{model_uri}"
                
                # Download artifact to temporary directory
                temp_dir = mlflow.artifacts.download_artifacts(
                    artifact_uri, dst_path=str(self.cache_dir)
                )
                
                # Find the downloaded file
                temp_path = Path(temp_dir)
                if temp_path.is_file():
                    downloaded_file = temp_path
                else:
                    # Try to find .pkl or .txt file (LightGBM can be saved as text)
                    pkl_files = list(temp_path.glob("*.pkl"))
                    txt_files = list(temp_path.glob("*.txt"))
                    if pkl_files:
                        downloaded_file = pkl_files[0]
                    elif txt_files:
                        downloaded_file = txt_files[0]
                    else:
                        # Try model_uri as filename
                        artifact_name = Path(model_uri).name
                        downloaded_file = temp_path / artifact_name
                        if not downloaded_file.exists():
                            raise FileNotFoundError(
                                f"Could not find model file in downloaded artifacts: {temp_dir}"
                            )
                
                # Copy to cache location
                import shutil
                shutil.copy2(downloaded_file, cache_path)
                self.logger.info(f"LightGBM cached to: {cache_path}")
                
                # Load and return
                try:
                    return joblib.load(cache_path)
                except:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                
            except Exception as e:
                # If download fails, check if we have a cached version
                if cache_path.exists():
                    self.logger.warning(
                        f"MLflow download failed ({e}), but using cached LightGBM: {cache_path}"
                    )
                    try:
                        return joblib.load(cache_path)
                    except:
                        with open(cache_path, 'rb') as f:
                            return pickle.load(f)
                else:
                    self.logger.error(
                        f"Failed to load LightGBM from MLflow and no cache available: {e}"
                    )
                    raise RuntimeError(
                        f"Cannot load LightGBM: MLflow unavailable and no cached version. "
                        f"Error: {e}"
                    )
        else:
            raise ValueError(
                f"Invalid load_method: {load_method}. Must be 'mlflow' or 'sklearn'"
            )


def load_model_by_run_id(run_id: str, tracking_uri: str = None):
    """
    Load model from MLflow by run ID
    
    Args:
        run_id: MLflow run ID
        tracking_uri: MLflow tracking URI (default: training_workspace/mlruns)
    
    Returns:
        Loaded PyTorch model
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri("training_workspace/mlruns")
    
    # Load model from run
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri)
    return model


def load_model_from_registry(model_name: str, version: int = None, stage: str = None, tracking_uri: str = None):
    """
    Load model from MLflow Model Registry
    
    Args:
        model_name: Name of the registered model
        version: Model version (if None, loads latest)
        stage: Model stage (Production, Staging, Archived) - takes precedence over version
        tracking_uri: MLflow tracking URI
    
    Returns:
        Loaded PyTorch model
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri("training_workspace/mlruns")
    
    if stage:
        model_uri = f"models:/{model_name}/{stage}"
        print(f"Loading model from registry: {model_name} (stage: {stage})")
    elif version:
        model_uri = f"models:/{model_name}/{version}"
        print(f"Loading model from registry: {model_name} (version: {version})")
    else:
        model_uri = f"models:/{model_name}/latest"
        print(f"Loading latest model from registry: {model_name}")
    
    model = mlflow.pytorch.load_model(model_uri)
    return model


def predict_weight(model, image_path: str, device: str = "cpu"):
    """
    Use loaded model to predict weight from image
    
    Args:
        model: Loaded PyTorch model
        image_path: Path to input image
        device: Device to run inference on
    
    Returns:
        Predicted weight (kg)
    """
    from PIL import Image
    
    # Prepare image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((360, 640))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        prediction = model(image_tensor)
        weight = prediction.item()
    
    return weight


def list_runs(experiment_name: str = "MFF_ResNet_Training", limit: int = 10):
    """
    List recent MLflow runs with their metrics
    
    Args:
        experiment_name: Name of the experiment
        limit: Maximum number of runs to display
    """
    mlflow.set_tracking_uri("training_workspace/mlruns")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return
    
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=limit,
        order_by=["metrics.best_val_mae ASC"]  # Best models first
    )
    
    print(f"\nRecent runs in '{experiment_name}':")
    print("-" * 100)
    print(f"{'Run ID':<40} {'Run Name':<30} {'Best Val MAE':<15} {'Status':<10}")
    print("-" * 100)
    
    for run in runs:
        run_id = run.info.run_id
        run_name = run.data.tags.get('mlflow.runName', 'N/A')
        best_mae = run.data.metrics.get('best_val_mae', 'N/A')
        status = run.info.status
        
        print(f"{run_id:<40} {run_name:<30} {best_mae:<15} {status:<10}")
    
    print("-" * 100)


def main():
    parser = argparse.ArgumentParser(description="Load and use models from MLflow")
    parser.add_argument("--run_id", type=str, help="MLflow run ID to load model from")
    parser.add_argument("--model_name", type=str, help="Model name in Model Registry")
    parser.add_argument("--version", type=int, help="Model version in registry")
    parser.add_argument("--stage", type=str, choices=["Production", "Staging", "Archived"], 
                       help="Model stage in registry")
    parser.add_argument("--image_path", type=str, help="Path to image for inference")
    parser.add_argument("--list_runs", action="store_true", help="List recent runs")
    parser.add_argument("--experiment_name", type=str, default="MFF_ResNet_Training",
                       help="MLflow experiment name")
    parser.add_argument("--tracking_uri", type=str, default=None,
                       help="MLflow tracking URI (default: training_workspace/mlruns)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference")
    
    args = parser.parse_args()
    
    if args.list_runs:
        list_runs(args.experiment_name)
        return
    
    # Load model
    if args.model_name:
        # Load from Model Registry
        model = load_model_from_registry(
            args.model_name, 
            version=args.version, 
            stage=args.stage,
            tracking_uri=args.tracking_uri
        )
    elif args.run_id:
        # Load by run ID
        model = load_model_by_run_id(args.run_id, tracking_uri=args.tracking_uri)
    else:
        print("Error: Either --run_id or --model_name must be provided")
        print("Use --list_runs to see available runs")
        return
    
    print("Model loaded successfully!")
    
    # Run inference if image provided
    if args.image_path:
        weight = predict_weight(model, args.image_path, args.device)
        print(f"\nPredicted weight: {weight:.4f} kg")
    else:
        print("\nModel loaded. Use --image_path to run inference")


if __name__ == "__main__":
    main()

