import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import argparse
import sys
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import joblib

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from training_workspace.utils.mlflow_utils import setup_mlflow_experiment, log_params_from_args

def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM with PCA on Auto-Features")
    
    # Data paths
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test data")
    
    # PCA settings
    parser.add_argument("--n_components", type=int, default=50, help="Number of PCA components for auto-features")
    
    # MLflow settings
    parser.add_argument("--experiment_name", type=str, default="MFF_GBDT_PCA", help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow run name")
    
    # Model hyperparameters
    parser.add_argument("--n_estimators", type=int, default=2000, help="Number of estimators")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_leaves", type=int, default=31, help="Number of leaves")
    parser.add_argument("--max_depth", type=int, default=-1, help="Max depth")
    parser.add_argument("--min_child_samples", type=int, default=20, help="Min child samples")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio")
    parser.add_argument("--colsample_bytree", type=float, default=0.8, help="Colsample by tree")
    parser.add_argument("--reg_alpha", type=float, default=0.0, help="L1 regularization")
    parser.add_argument("--reg_lambda", type=float, default=0.0, help="L2 regularization")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    
    return parser.parse_args()

def load_and_preprocess(path, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['weight', 'imgName']
    
    df = pd.read_csv(path)
    y = df['weight']
    
    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Separate Manual and Auto features
    auto_cols = [c for c in feature_cols if c.startswith('auto_')]
    manual_cols = [c for c in feature_cols if not c.startswith('auto_')]
    
    X_manual = df[manual_cols]
    X_auto = df[auto_cols]
    
    return X_manual, X_auto, y

def train_pca_and_model(args):
    setup_mlflow_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=args.run_name) as run:
        log_params_from_args(args)
        
        print("Loading data...")
        X_man_train, X_auto_train, y_train = load_and_preprocess(args.train_path)
        X_man_val, X_auto_val, y_val = load_and_preprocess(args.val_path)
        X_man_test, X_auto_test, y_test = load_and_preprocess(args.test_path)
        
        print(f"Manual features: {X_man_train.shape[1]}")
        print(f"Auto features: {X_auto_train.shape[1]}")
        
        # --- PCA Step ---
        print(f"Fitting PCA with {args.n_components} components on Auto-features (Train only)...")
        
        # Standardize auto-features before PCA (important!)
        scaler = StandardScaler()
        X_auto_train_scaled = scaler.fit_transform(X_auto_train)
        X_auto_val_scaled = scaler.transform(X_auto_val)
        X_auto_test_scaled = scaler.transform(X_auto_test)
        
        pca = PCA(n_components=args.n_components, random_state=args.random_state)
        X_pca_train = pca.fit_transform(X_auto_train_scaled)
        X_pca_val = pca.transform(X_auto_val_scaled)
        X_pca_test = pca.transform(X_auto_test_scaled)
        
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"Explained Variance Ratio: {explained_variance:.4f}")
        mlflow.log_metric("pca_explained_variance", explained_variance)
        
        # Create DataFrames for PCA features
        pca_cols = [f"pca_{i}" for i in range(args.n_components)]
        
        def combine_features(X_man, X_pca_arr):
            X_pca_df = pd.DataFrame(X_pca_arr, columns=pca_cols, index=X_man.index)
            return pd.concat([X_man, X_pca_df], axis=1)
        
        X_train_final = combine_features(X_man_train, X_pca_train)
        X_val_final = combine_features(X_man_val, X_pca_val)
        X_test_final = combine_features(X_man_test, X_pca_test)
        
        print(f"Final feature count: {X_train_final.shape[1]}")
        
        # --- Train LightGBM ---
        print("Training LightGBM...")
        model = lgb.LGBMRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            random_state=args.random_state
        )
        
        mlflow.lightgbm.autolog(log_models=False)
        
        model.fit(
            X_train_final, y_train,
            eval_set=[(X_val_final, y_val)],
            eval_metric='l1',
            callbacks=[early_stopping(stopping_rounds=100)]
        )
        
        # --- Evaluate ---
        test_pred = model.predict(X_test_final)
        
        mae = mean_absolute_error(y_test, test_pred)
        r2 = r2_score(y_test, test_pred)
        rmse = mean_squared_error(y_test, test_pred) ** 0.5
        
        print(f"Test MAE: {mae:.4f}")
        print(f"Test R2: {r2:.4f}")
        
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)
        
        # Save PCA and Scaler
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/auto_scaler.pkl")
        joblib.dump(pca, "models/auto_pca.pkl")
        mlflow.log_artifact("models/auto_scaler.pkl")
        mlflow.log_artifact("models/auto_pca.pkl")
        mlflow.lightgbm.log_model(model, "model")

if __name__ == "__main__":
    args = parse_args()
    train_pca_and_model(args)

