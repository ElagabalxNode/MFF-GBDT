"""
Test FusionNet with MLflow support.

Features:
1. Loads model from MLflow (run_id or model_uri) or local path.
2. Uses proper data splits via split_info.csv (aligned mode).
3. Evaluates metrics (MAE, RMSE, R2) and logs to MLflow.
4. Saves predictions to CSV.
5. Extracts and saves Auto-Features (2048 dim) for LightGBM training.

Usage:
    python training_workspace/features/testing/test_fusion_mlflow.py \
        --run_id <MLFLOW_RUN_ID> \
        --split_info_csv training_workspace/data/processed/csvData/fusion_split/split_info.csv \
        --train_features_csv training_workspace/data/processed/csvData/fusion_split/train_features_normalized.csv \
        --test_features_csv training_workspace/data/processed/csvData/fusion_split/test_features_normalized.csv
"""

import sys
import os
import argparse
import time
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.pytorch

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training_workspace.features.datasets.chicken200 import Chicken_200_trainset, Chicken_200_testset
from training_workspace.features.models.FusonNet import fusonnet50
from training_workspace.utils.mlflow_utils import setup_mlflow_experiment

def get_manual_features(csv_path: str, split: str = None):
    """Load manual features from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Manual features CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path, index_col='imgName')
    if split:
        print(f"Loaded {split} features: {len(df)} samples, {len(df.columns)-1} features")
    return df

def path_to_imgname(path: str) -> str:
    """Convert image path to imgName format used in CSV."""
    base = os.path.basename(path)
    if base.endswith('-0.png'):
        return base[:-6] + '.png'
    return base

def evaluate_model(model, dataloaders, device, train_features_df, test_features_df, output_dir):
    """
    Evaluate model on train and test sets.
    Returns metrics dict.
    """
    model.eval()
    metrics = {}
    
    features_by_phase = {
        "train": train_features_df,
        "val": test_features_df
    }
    
    # Feature columns (excluding weight if present)
    feature_cols = [c for c in train_features_df.columns if c != 'weight']
    num_features = len(feature_cols)
    print(f"Using {num_features} manual features for evaluation")

    for phase in ["train", "val"]:
        print(f"\nEvaluating on {phase} set...")
        weight_gt = []
        weight_pr = []
        
        features_df = features_by_phase[phase]
        
        with torch.no_grad():
            for inputs, labels, path in dataloaders[phase]:
                labels_gt = labels.numpy()
                weight_gt.extend(labels_gt)
                
                inputs = inputs.to(device)
                
                # Prepare manual features
                batch_manual_features = []
                for p in range(len(path)):
                    x_path = path[p]
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
                
                # Inference
                outputs, _ = model(inputs, manual_features_tensor)
                
                predict_weight = outputs.cpu().numpy()
                weight_pr.extend(predict_weight)
        
        # Calculate metrics
        mae = mean_absolute_error(weight_gt, weight_pr)
        mse = mean_squared_error(weight_gt, weight_pr)
        rmse = mse ** 0.5
        r2 = r2_score(weight_gt, weight_pr)
        
        print(f"{phase} Results: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        metrics[f"{phase}_mae"] = mae
        metrics[f"{phase}_rmse"] = rmse
        metrics[f"{phase}_r2"] = r2
        
        # Save predictions
        pred_df = pd.DataFrame({'gt': weight_gt, 'pr': weight_pr})
        pred_path = os.path.join(output_dir, f'{phase}_predictions.csv')
        pred_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path)
        
    return metrics

def extract_and_save_features(model, dataloaders, device, train_features_df, test_features_df, output_dir):
    """
    Extract and save combined features (Manual + Auto) for LightGBM.
    """
    print("\nExtracting auto-features...")
    model.eval()
    
    features_by_phase = {
        "train": train_features_df,
        "val": test_features_df
    }
    
    feature_cols = [c for c in train_features_df.columns if c != 'weight']
    num_manual = len(feature_cols)
    
    # Prepare CSV header
    header = ['weight', 'imgName'] + feature_cols + [f'auto_{i}' for i in range(2048)]
    header_str = ','.join(header) + '\n'
    
    for phase in ["train", "val"]:
        print(f"Processing {phase} set...")
        output_file = os.path.join(output_dir, f'features_with_auto_{phase}.csv')
        
        features_df = features_by_phase[phase]
        
        with open(output_file, 'w') as f:
            f.write(header_str)
            
            with torch.no_grad():
                for inputs, labels, path in dataloaders[phase]:
                    inputs = inputs.to(device)
                    batch_size = inputs.size(0)
                    
                    # Prepare manual features
                    manual_features_list = []
                    keys = []
                    
                    for b in range(batch_size):
                        x_path = path[b]
                        key = path_to_imgname(x_path)
                        keys.append(key)
                        
                        if key in features_df.index:
                            feat = features_df.loc[key, feature_cols].values
                        else:
                            feat = np.zeros(num_manual)
                        manual_features_list.append(feat)
                    
                    manual_tensor = torch.as_tensor(
                        np.array(manual_features_list), 
                        dtype=torch.float32
                    ).to(device)
                    
                    # Forward pass
                    _, auto_features = model(inputs, manual_tensor)
                    auto_features_np = auto_features.cpu().numpy()
                    
                    # Write to file
                    for b in range(batch_size):
                        row = []
                        row.append(str(labels[b].item())) # weight
                        row.append(keys[b])               # imgName
                        
                        # Manual features
                        row.extend([str(x) for x in manual_features_list[b]])
                        
                        # Auto features
                        row.extend([str(x) for x in auto_features_np[b]])
                        
                        f.write(','.join(row) + '\n')
        
        print(f"Saved {phase} features to: {output_file}")
        mlflow.log_artifact(output_file)

def main():
    parser = argparse.ArgumentParser(description="Test FusionNet with MLflow")
    
    # Model source
    parser.add_argument("--run_id", type=str, help="MLflow Run ID to load model from")
    parser.add_argument("--model_path", type=str, help="Local path to model weights (.pth)")
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None, help="MLflow tracking URI")
    
    # Data
    parser.add_argument("--split_info_csv", type=str, required=True, help="Path to split_info.csv")
    parser.add_argument("--train_features_csv", type=str, required=True, help="Normalized train features")
    parser.add_argument("--test_features_csv", type=str, required=True, help="Normalized test features")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="training_workspace/data/outputs/inference", help="Output directory")
    parser.add_argument("--save_features", action="store_true", default=True, help="Extract and save auto-features")
    
    args = parser.parse_args()
    
    # Setup
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    setup_mlflow_experiment("MFF_Inference")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"eval_{args.run_id}" if args.run_id else f"eval_local_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # --- Load Data ---
        print("Loading data...")
        train_features_df = get_manual_features(args.train_features_csv, "train")
        test_features_df = get_manual_features(args.test_features_csv, "test")
        
        # Datasets (Aligned mode)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((360, 640))
        ])
        
        train_dataset = Chicken_200_trainset(transform=transform, split_info_csv=args.split_info_csv)
        val_dataset = Chicken_200_testset(transform=transform, split_info_csv=args.split_info_csv)
        
        dataloaders = {
            "train": DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4),
            "val": DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
        }
        
        # --- Load Model ---
        print("Loading model...")
        model = fusonnet50()
        
        if args.run_id:
            print(f"Loading from MLflow run: {args.run_id}")
            model_uri = f"runs:/{args.run_id}/model"
            loaded_model = mlflow.pytorch.load_model(model_uri)
            # Copy weights
            model.load_state_dict(loaded_model.state_dict())
        elif args.model_path:
            print(f"Loading local weights: {args.model_path}")
            model.load_state_dict(torch.load(args.model_path, map_location=device))
        else:
            raise ValueError("Must provide either --run_id or --model_path")
            
        model = model.to(device)
        
        # --- Evaluate ---
        print("Starting evaluation...")
        metrics = evaluate_model(
            model, dataloaders, device, 
            train_features_df, test_features_df, 
            args.output_dir
        )
        
        mlflow.log_metrics(metrics)
        
        # --- Extract Features (Optional) ---
        if args.save_features:
            extract_and_save_features(
                model, dataloaders, device,
                train_features_df, test_features_df,
                args.output_dir
            )
            
        print("\nInference completed successfully!")
        print(f"Outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

