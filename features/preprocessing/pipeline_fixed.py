import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import argparse

# 1. Global constants for reproducibility
# Fix the seed once. In science, 42 is commonly used.
RANDOM_SEED = 42 

def robust_data_pipeline(input_csv_path=None, train_csv_path=None, test_csv_path=None, output_dir='data/processed/csvData/processed_fixed/'):
    """
    Robust data preprocessing pipeline.
    
    Modes:
    1. Single Input File: Splits into Train/Val/Test.
    2. Train/Test Input Files: Respects existing split, further splits Train into Train/Val.
    
    Steps:
    1. Load data
    2. Split (if needed)
    3. Normalization (Scaler fit only on Train)
    4. Save processed datasets
    """
    
    df_train = None
    df_val = None
    df_test = None
    
    # --- STAGE 1: LOAD AND SPLIT ---
    if train_csv_path and test_csv_path:
        print(f"Loading pre-split data:\nTrain: {train_csv_path}\nTest: {test_csv_path}")
        df_train_full = pd.read_csv(train_csv_path)
        df_test = pd.read_csv(test_csv_path)
        
        # Split Train_Full into Train and Val (e.g., 80/20 of the training set)
        print("Splitting provided Train set into Train and Val...")
        # Ensure stratification if needed, but random split is usually fine for regression if data is homogeneous
        df_train, df_val = train_test_split(
            df_train_full, 
            test_size=0.2, 
            random_state=RANDOM_SEED, 
            shuffle=True
        )
    elif input_csv_path:
        print(f"Loading single file: {input_csv_path}")
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"File not found: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        
        # First, separate Test set (20%)
        df_train_val, df_test = train_test_split(
            df, 
            test_size=0.2, 
            random_state=RANDOM_SEED, 
            shuffle=True
        )
        
        # Split remaining 80% into Train and Val
        df_train, df_val = train_test_split(
            df_train_val, 
            test_size=0.25, # 0.25 * 0.8 = 0.2
            random_state=RANDOM_SEED, 
            shuffle=True
        )
    else:
        raise ValueError("Must provide either input_csv_path OR (train_csv_path and test_csv_path)")

    print(f"Dataset sizes: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # Helper to separate features/target
    def split_features_target(df):
        if 'weight' not in df.columns or 'imgName' not in df.columns:
            raise ValueError("CSV must contain 'weight' and 'imgName' columns")
        y = df['weight']
        meta = df['imgName']
        X = df.drop(['weight', 'imgName'], axis=1)
        return X, y, meta

    X_train, y_train, meta_train = split_features_target(df_train)
    X_val, y_val, meta_val = split_features_target(df_val)
    X_test, y_test, meta_test = split_features_target(df_test)
    
    # Check feature consistency
    feature_names = X_train.columns.tolist()
    print(f"Number of features: {len(feature_names)}")
    
    # --- STAGE 2: NORMALIZATION (SCALING) ---
    # Important: fit ONLY on X_train
    scaler = MinMaxScaler()
    
    print("Fitting scaler on Train set...")
    scaler.fit(X_train)
    
    # Apply transformation
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # --- STAGE 3: ASSEMBLY AND SAVING ---
    def save_subset(X_arr, y_series, meta_series, name):
        df_scaled = pd.DataFrame(X_arr, columns=feature_names)
        df_scaled.insert(0, 'imgName', meta_series.values)
        df_scaled.insert(1, 'weight', y_series.values)
        
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'{name}.csv')
        df_scaled.to_csv(path, index=False)
        print(f"Saved: {path}")

    save_subset(X_train_scaled, y_train, meta_train, 'train_fixed')
    save_subset(X_val_scaled, y_val, meta_val, 'val_fixed')
    save_subset(X_test_scaled, y_test, meta_test, 'test_fixed')
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
    print("Scaler saved as feature_scaler.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for LightGBM")
    parser.add_argument("--input", type=str, help="Path to single input CSV (manual features only)")
    parser.add_argument("--train_csv", type=str, help="Path to pre-split Train CSV (manual+auto)")
    parser.add_argument("--test_csv", type=str, help="Path to pre-split Test/Val CSV (manual+auto)")
    parser.add_argument("--output", type=str, default='data/processed/csvData/processed_fixed/', help="Output directory")
    
    args = parser.parse_args()
    
    # Default fallback for manual run if no args provided (legacy behavior)
    if not args.input and not args.train_csv:
        # Try to find best available data
        # Prefer withauto if available
        default_train = 'data/processed/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-withnormal-train.csv'
        default_test = 'data/processed/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-withnormal-val.csv'
        
        if os.path.exists(default_train) and os.path.exists(default_test):
            print("Using default Auto-Feature datasets...")
            robust_data_pipeline(train_csv_path=default_train, test_csv_path=default_test, output_dir=args.output)
        else:
            print("Auto-Feature datasets not found. Falling back to manual features...")
            default_input = 'data/processed/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_features.csv'
            robust_data_pipeline(input_csv_path=default_input, output_dir=args.output)
    else:
        robust_data_pipeline(input_csv_path=args.input, train_csv_path=args.train_csv, test_csv_path=args.test_csv, output_dir=args.output)
