"""
Train FusionNet with MLflow tracking.

IMPORTANT: To avoid data leakage, use prepare_fusion_split.py first to create
properly split and normalized features. Then pass the generated files here.

Usage:
    # Step 1: Prepare data (once)
    python training_workspace/features/preprocessing/prepare_fusion_split.py
    
    # Step 2: Train with aligned splits
    python training_workspace/features/training/train_fusion_mlflow.py \
        --train_features_csv training_workspace/data/processed/csvData/fusion_split/train_features_normalized.csv \
        --test_features_csv training_workspace/data/processed/csvData/fusion_split/test_features_normalized.csv \
        --split_info_csv training_workspace/data/processed/csvData/fusion_split/split_info.csv
"""

import sys
import os
import argparse
import time
import copy
import re
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.pytorch
import tempfile

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 3 levels: training/ -> features/ -> training_workspace/ -> project_root/
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training_workspace.features.datasets.chicken200 import Chicken_200_trainset, Chicken_200_testset
from training_workspace.features.models.FusonNet import fusonnet50
from training_workspace.utils.mlflow_utils import setup_mlflow_experiment, log_params_from_args

# --- Helper Classes from original train_fusion.py ---

class myresnet_base(nn.Module):
    def __init__(self):
        super(myresnet_base, self).__init__()
        self.model_ft = models.resnet50(pretrained=False)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.model_ft(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = torch.flatten(x)
        return x

# --- Helper Functions ---

def fizze_resnet_parameter(model, fizze_resnet):
    if fizze_resnet:
        keylist = ['fc.weight', 'fc.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
        for name, param in model.named_parameters():
            if name not in keylist:
                param.requires_grad = False

def init_Fusonmodel(fusonnet, weightPath, fizze_resnet):
    # This logic is kept from original script to preserve weight loading behavior
    if not os.path.exists(weightPath):
        print(f"Warning: Initial weights not found at {weightPath}. initializing randomly.")
        return fusonnet

    model_pretrained = myresnet_base()
    try:
        model_pretrained.load_state_dict(torch.load(weightPath, map_location='cpu'))
    except Exception as e:
        print(f"Error loading pretrained weights: {e}")
        return fusonnet

    fusonnet_dict = fusonnet.state_dict()
    
    # Compare parameters and remove different ones (classification vs regression head differences likely)
    pretrained_dict = {k[9:]: v for k, v in model_pretrained.state_dict().items() if k.startswith('model_ft.')}
    if 'fc.weight' in pretrained_dict: pretrained_dict.pop('fc.weight')
    if 'fc.bias' in pretrained_dict: pretrained_dict.pop('fc.bias')
    
    fusonnet_dict.update(pretrained_dict)
    fusonnet.load_state_dict(fusonnet_dict)

    if fizze_resnet:
        fizze_resnet_parameter(fusonnet, True)

    return fusonnet

def load_weights_from_mlflow(run_id: str, tracking_uri: str = None, experiment_name: str = "MFF_ResNet_Training"):
    """
    Load ResNet weights from MLflow run and save to temporary file
    
    Args:
        run_id: MLflow run ID containing the ResNet model
        tracking_uri: MLflow tracking URI (default: training_workspace/mlruns)
        experiment_name: Experiment name (for validation)
    
    Returns:
        Path to temporary file with weights
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri("training_workspace/mlruns")
    
    # Load model from MLflow
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading ResNet weights from MLflow: {model_uri}")
    
    try:
        # Load the model
        model = mlflow.pytorch.load_model(model_uri)
        
        # Extract state_dict
        state_dict = model.state_dict()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
        temp_path = temp_file.name
        temp_file.close()
        
        torch.save(state_dict, temp_path)
        print(f"ResNet weights extracted and saved to temporary file: {temp_path}")
        
        return temp_path
    except Exception as e:
        raise RuntimeError(f"Failed to load model from MLflow run {run_id}: {e}")


def get_manual_features(csv_path: str, split: str = None):
    """
    Load manual features from CSV.
    
    Args:
        csv_path: Path to features CSV (train or test)
        split: Optional split name for logging
    
    Returns:
        DataFrame indexed by imgName
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Manual features CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path, index_col='imgName')
    
    if split:
        print(f"Loaded {split} features: {len(df)} samples, {len(df.columns)-1} features")
    
    return df


def path_to_imgname(path: str) -> str:
    """
    Convert image path to imgName format used in CSV.
    
    Path: ".../maskImg/1.1_Depth-0-0.png" -> imgName: "1.1_Depth-0.png"
    """
    base = os.path.basename(path)
    # Remove the extra "-0" before .png
    if base.endswith('-0.png'):
        return base[:-6] + '.png'
    return base


def train_model(model, dataloaders, loss_fn, optimizer, scheduler, device, num_epochs, 
                train_features_df, val_features_df, patience=15, min_delta=0.001):
    """
    Train the FusionNet model with early stopping.
    
    Args:
        model: FusionNet model
        dataloaders: Dict with "train" and "val" DataLoaders
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: torch device
        num_epochs: Number of training epochs
        train_features_df: DataFrame with normalized train features (indexed by imgName)
        val_features_df: DataFrame with normalized val/test features (indexed by imgName)
        patience: Early stopping patience (epochs without improvement)
        min_delta: Minimum improvement to reset patience counter
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = float('inf')
    epochs_without_improvement = 0
    
    # Select appropriate features DataFrame for each phase
    features_by_phase = {
        "train": train_features_df,
        "val": val_features_df
    }
    
    # Get feature columns (exclude 'weight' if present)
    feature_cols = [c for c in train_features_df.columns if c != 'weight']
    num_features = len(feature_cols)
    print(f"Using {num_features} manual features")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print('-' * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            weight_gt = []
            weight_pr = []
            
            # Get features for this phase
            features_df = features_by_phase[phase]

            for inputs, labels, path in dataloaders[phase]:
                labels_gt = labels.numpy()
                weight_gt.extend(labels_gt)

                inputs, labels = inputs.to(device), labels.to(device)

                # Prepare manual features
                batch_manual_features = []
                for p in range(len(path)):
                    x_path = path[p]
                    # Convert path to imgName format
                    key = path_to_imgname(x_path)
                    
                    if key in features_df.index:
                        features = features_df.loc[key, feature_cols].values
                    else:
                        print(f"Warning: {key} not found in CSV, using zeros")
                        features = np.zeros(num_features)
                    
                    batch_manual_features.append(features)
                
                manual_features_tensor = torch.as_tensor(
                    np.array(batch_manual_features), 
                    dtype=torch.float32
                ).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs, _ = model(inputs, manual_features_tensor)
                    loss = loss_fn(outputs.float(), labels.float())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                
                predict_weight = outputs.cpu().detach().numpy()
                weight_pr.extend(predict_weight)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            mae = mean_absolute_error(weight_gt, weight_pr)
            mse = mean_squared_error(weight_gt, weight_pr)
            rmse = mse ** 0.5
            r2 = r2_score(weight_gt, weight_pr)

            print(f"{phase} Loss: {epoch_loss:.4f} MAE: {mae:.4f} RMSE: {rmse:.4f} R2: {r2:.4f}")

            # Log metrics to MLflow
            mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
            mlflow.log_metric(f"{phase}_mae", mae, step=epoch)
            mlflow.log_metric(f"{phase}_rmse", rmse, step=epoch)
            mlflow.log_metric(f"{phase}_r2", r2, step=epoch)

            # Early stopping check on validation
            if phase == "val":
                if mae < best_mae - min_delta:
                    best_mae = mae
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                    print(f"New best model! MAE: {best_mae:.4f}")
                    mlflow.log_metric("best_val_mae", best_mae, step=epoch)
                    # Save checkpoint locally and log as artifact
                    torch.save(best_model_wts, "best_model.pth")
                    mlflow.log_artifact("best_model.pth")
                else:
                    epochs_without_improvement += 1
                    print(f"No improvement for {epochs_without_improvement}/{patience} epochs")
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    mlflow.log_metric("stopped_epoch", epoch)
                    model.load_state_dict(best_model_wts)
                    return model

    print(f"Training completed. Best Validation MAE: {best_mae:.4f}")
    model.load_state_dict(best_model_wts)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train FusionNet with MLflow")
    
    # Data paths - NEW aligned mode
    parser.add_argument("--train_features_csv", type=str, default=None,
                        help="Path to normalized TRAIN features CSV (from prepare_fusion_split.py)")
    parser.add_argument("--test_features_csv", type=str, default=None,
                        help="Path to normalized TEST features CSV (from prepare_fusion_split.py)")
    parser.add_argument("--split_info_csv", type=str, default=None,
                        help="Path to split_info.csv for aligned image splits")
    
    # Legacy mode (backward compatible)
    parser.add_argument("--manual_features_csv", type=str, 
                        default="training_workspace/data/processed/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_normal_features.csv",
                        help="[LEGACY] Path to single manual features CSV (may have data leakage)")
    
    # Model initialization
    parser.add_argument("--init_weights", type=str, 
                        default="training_workspace/data/models/features/resnet_base.pth", 
                        help="Path to initial ResNet weights")
    parser.add_argument("--init_weights_from_mlflow", type=str, default=None,
                        help="MLflow run ID to load ResNet weights from (takes precedence over --init_weights)")
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None,
                        help="MLflow tracking URI (default: training_workspace/mlruns)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--freeze_resnet", action="store_true", default=True, help="Freeze ResNet backbone")
    
    # Early Stopping
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Minimum improvement for early stopping")
    
    # MLflow
    parser.add_argument("--experiment_name", type=str, default="MFF_FusionNet", help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow run name")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if using aligned mode or legacy mode
    using_aligned_mode = (
        args.train_features_csv is not None and 
        args.test_features_csv is not None and 
        args.split_info_csv is not None
    )
    
    if using_aligned_mode:
        print("="*60)
        print("ALIGNED MODE: Using properly split and normalized data")
        print("="*60)
    else:
        print("="*60)
        print("WARNING: LEGACY MODE - potential data leakage!")
        print("Consider using prepare_fusion_split.py first for proper splits")
        print("="*60)
    
    # Setup MLflow
    setup_mlflow_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=args.run_name):
        log_params_from_args(args)
        mlflow.log_param("data_mode", "aligned" if using_aligned_mode else "legacy")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # --- Data Loading ---
        if using_aligned_mode:
            # Load train and test features separately
            train_features_df = get_manual_features(args.train_features_csv, split="train")
            test_features_df = get_manual_features(args.test_features_csv, split="test")
            
            # Create datasets with aligned splits
            train_dataset = Chicken_200_trainset(
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((360, 640))
                ]),
                split_info_csv=args.split_info_csv
            )
            val_dataset = Chicken_200_testset(
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((360, 640))
                ]),
                split_info_csv=args.split_info_csv
            )
        else:
            # Legacy mode - single CSV for all features (may have data leakage!)
            train_features_df = get_manual_features(args.manual_features_csv, split="all")
            test_features_df = train_features_df  # Same features for train and val (leakage!)
            
            train_dataset = Chicken_200_trainset(transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((360, 640))
            ]))
            val_dataset = Chicken_200_testset(transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((360, 640))
            ]))
        
        dataloaders = {
            "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4),
            "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        }
        
        print(f"Train batches: {len(dataloaders['train'])}, Val batches: {len(dataloaders['val'])}")
        
        # --- Model Setup ---
        # Determine weights path: MLflow takes precedence over local file
        weights_path = args.init_weights
        temp_weights_file = None
        
        if args.init_weights_from_mlflow:
            # Load weights from MLflow
            temp_weights_file = load_weights_from_mlflow(
                args.init_weights_from_mlflow,
                tracking_uri=args.mlflow_tracking_uri
            )
            weights_path = temp_weights_file
            mlflow.log_param("init_weights_source", "mlflow")
            mlflow.log_param("init_weights_mlflow_run_id", args.init_weights_from_mlflow)
        else:
            mlflow.log_param("init_weights_source", "local_file")
            mlflow.log_param("init_weights_path", args.init_weights)
        
        fusonnet = fusonnet50()
        model = init_Fusonmodel(fusonnet, weights_path, args.freeze_resnet)
        model = model.to(device)
        
        # Clean up temporary file if created (after model initialization)
        if temp_weights_file and os.path.exists(temp_weights_file):
            try:
                os.remove(temp_weights_file)
                print(f"Cleaned up temporary weights file: {temp_weights_file}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file {temp_weights_file}: {e}")
        
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            momentum=0.9
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        loss_fn = nn.L1Loss()
        
        print(f"\nStarting training (patience={args.patience}, min_delta={args.min_delta})...")
        model = train_model(
            model, dataloaders, loss_fn, optimizer, scheduler, 
            device, args.epochs, train_features_df, test_features_df,
            patience=args.patience, min_delta=args.min_delta
        )
        
        # Save final model
        mlflow.pytorch.log_model(model, "model")
        
        # Log run ID for easy model retrieval
        run_id = mlflow.active_run().info.run_id
        print(f"\nTraining completed!")
        print(f"Run ID: {run_id}")
        print(f"Model URI: runs:/{run_id}/model")
        print(f"\nTo load this model later, use:")
        print(f"  python training_workspace/features/training/load_model_from_mlflow.py --run_id {run_id}")

if __name__ == '__main__':
    main()
