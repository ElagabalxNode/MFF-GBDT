import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import argparse
import sys
import mlflow
import mlflow.lightgbm

# Add project root to sys.path to allow importing utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.mlflow_utils import setup_mlflow_experiment, log_params_from_args

def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM model with MLflow tracking")
    
    # Data paths
    parser.add_argument("--train_path", type=str, default="data/processed/csvData/processed_fixed/train_fixed.csv", help="Path to training data")
    parser.add_argument("--val_path", type=str, default="data/processed/csvData/processed_fixed/val_fixed.csv", help="Path to validation data")
    parser.add_argument("--test_path", type=str, default="data/processed/csvData/processed_fixed/test_fixed.csv", help="Path to test data")
    
    # MLflow settings
    parser.add_argument("--experiment_name", type=str, default="MFF_GBDT_LightGBM", help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow run name")
    
    # Model hyperparameters (defaults set to DeeperRegStrong winner: test_mae=0.0882, test_r2=0.7272)
    parser.add_argument("--n_estimators", type=int, default=2000, help="Number of estimators")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_leaves", type=int, default=12, help="Number of leaves")
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth")
    parser.add_argument("--min_child_samples", type=int, default=40, help="Min child samples")
    parser.add_argument("--min_child_weight", type=float, default=0.01, help="Min child weight")
    parser.add_argument("--subsample", type=float, default=0.65, help="Subsample ratio")
    parser.add_argument("--colsample_bytree", type=float, default=0.55, help="Colsample by tree")
    parser.add_argument("--reg_alpha", type=float, default=0.02, help="L1 regularization (lambda_l1)")
    parser.add_argument("--reg_lambda", type=float, default=0.15, help="L2 regularization (lambda_l2)")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    
    # Feature selection
    parser.add_argument("--select_top_features", type=int, default=0, help="Select top N features based on importance (0 to disable)")
    
    return parser.parse_args()

def get_xy(df):
    if 'weight' not in df.columns:
        raise ValueError("Column 'weight' not found")
    y = df['weight']
    # Remove target and meta-info
    drop_cols = ['weight']
    if 'imgName' in df.columns:
        drop_cols.append('imgName')
    
    X = df.drop(drop_cols, axis=1)
    return X, y

def train_and_evaluate(args):
    # Setup MLflow
    setup_mlflow_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=args.run_name) as run:
        # Log parameters
        log_params_from_args(args)
        
        # Load data
        print(f"Loading data from:\nTrain: {args.train_path}\nVal: {args.val_path}\nTest: {args.test_path}")
        df_train = pd.read_csv(args.train_path)
        df_val = pd.read_csv(args.val_path)
        df_test = pd.read_csv(args.test_path)
        
        x_train, y_train = get_xy(df_train)
        x_val, y_val = get_xy(df_val)
        x_test, y_test = get_xy(df_test)
        
        print(f"Initial features: {len(x_train.columns)}")
        
        # --- Feature Selection Step ---
        if args.select_top_features > 0:
            print(f"Performing feature selection (Top {args.select_top_features})...")
            
            # Train a temporary model for feature importance
            temp_model = lgb.LGBMRegressor(
                n_estimators=500,  # Faster training for selection
                learning_rate=0.1,
                random_state=args.random_state,
                n_jobs=-1
            )
            
            temp_model.fit(
                x_train, y_train,
                eval_set=[(x_val, y_val)],
                eval_metric='l1',
                callbacks=[early_stopping(stopping_rounds=50)]
            )
            
            # Get importance
            importances = temp_model.feature_importances_
            feature_names = x_train.columns
            
            # Create dataframe of features and importance
            feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Value': importances})
            feature_imp_df = feature_imp_df.sort_values(by="Value", ascending=False)
            
            # Select top N features
            top_features = feature_imp_df.head(args.select_top_features)['Feature'].tolist()
            print(f"Selected {len(top_features)} features.")
            
            # Log selected features
            mlflow.log_param("selected_features_count", len(top_features))
            
            # Filter datasets
            x_train = x_train[top_features]
            x_val = x_val[top_features]
            x_test = x_test[top_features]
            
        
        print(f"Final features used: {len(x_train.columns)}")
        mlflow.log_param("num_features", len(x_train.columns))
        # mlflow.log_param("features_list", list(x_train.columns)) # Can be too long
        
        # Initialize model
        model = lgb.LGBMRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            min_child_samples=args.min_child_samples,
            min_child_weight=args.min_child_weight,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            random_state=args.random_state
        )
        
        # Train
        print("Starting final training...")
        
        # Enable autologging for metrics (disable model logging to avoid duplication with manual log_model)
        mlflow.lightgbm.autolog(log_models=False)

        model.fit(
            x_train, y_train,
            eval_set=[(x_val, y_val)],
            eval_metric='l1', # L1 = MAE
            callbacks=[early_stopping(stopping_rounds=500)]
        )
        
        # Evaluate
        test_predict = model.predict(x_test)
        train_predict = model.predict(x_train)
        
        def get_metrics(y_true, y_pred, prefix):
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = mse ** 0.5
            r2 = r2_score(y_true, y_pred)
            return {
                f"{prefix}_mae": mae,
                f"{prefix}_mse": mse,
                f"{prefix}_rmse": rmse,
                f"{prefix}_r2": r2
            }
            
        train_metrics = get_metrics(y_train, train_predict, "train")
        test_metrics = get_metrics(y_test, test_predict, "test")
        
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)
        
        # Log evaluation table
        eval_table = pd.DataFrame({
            "true_weight": y_test,
            "predicted_weight": test_predict
        })
        # Add residuals
        eval_table['residual'] = eval_table['true_weight'] - eval_table['predicted_weight']
        eval_table['abs_error'] = eval_table['residual'].abs()
        
        mlflow.log_table(data=eval_table, artifact_file="evaluation_table.json")
        
        print("Test Metrics:")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.6f}")
            
        # Feature Importance
        plt.figure(figsize=(10, 6))
        lgb.plot_importance(model, max_num_features=20)
        plt.tight_layout()
        importance_path = "importance.png"
        plt.savefig(importance_path)
        mlflow.log_artifact(importance_path)
        os.remove(importance_path) # Clean up local file
        
        # Log model
        mlflow.lightgbm.log_model(model, "model")
        
        print(f"Run completed. Experiment: {args.experiment_name}, Run ID: {run.info.run_id}")

if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args)
