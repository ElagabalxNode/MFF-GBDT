import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# 1. Global constants for reproducibility
# Fix the seed once. In science, 42 is commonly used.
RANDOM_SEED = 42 

def robust_data_pipeline(input_csv_path, output_dir):
    """
    Robust data preprocessing pipeline:
    1. Load data
    2. Split (Train/Val/Test)
    3. Normalization (Scaler fit only on Train)
    4. Save processed datasets
    """
    print(f"Loading data from {input_csv_path}...")
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"File not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)
    
    # Separate target variable (weight) and identifiers from features
    # Assume 'weight' is the target, 'imgName' is the ID, rest are features
    if 'weight' not in df.columns or 'imgName' not in df.columns:
        raise ValueError("CSV must contain 'weight' and 'imgName' columns")

    y = df['weight']
    meta = df['imgName']
    X = df.drop(['weight', 'imgName'], axis=1)
    
    # Save column names for DataFrame reconstruction later
    feature_names = X.columns.tolist()
    
    # --- STAGE 1: SPLIT ---
    # First, separate Test set (20%). random_state is fixed!
    # Stratify is not needed for regression, but shuffle=True is required.
    X_train_val, X_test, y_train_val, y_test, meta_train_val, meta_test = train_test_split(
        X, y, meta, 
        test_size=0.2, 
        random_state=RANDOM_SEED, 
        shuffle=True
    )
    
    # Split remaining 80% (Train_Val) into Train (60% of total) and Val (20% of total)
    # 0.25 of 0.8 = 0.2 of total
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_train_val, y_train_val, meta_train_val, 
        test_size=0.25, 
        random_state=RANDOM_SEED, 
        shuffle=True
    )
    
    print(f"Dataset sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # --- STAGE 2: NORMALIZATION (SCALING) ---
    # Important: fit ONLY on X_train
    scaler = MinMaxScaler()
    
    print("Fitting scaler on Train set...")
    scaler.fit(X_train)
    
    # Apply transformation to all datasets
    # X_test and X_val are scaled using parameters learned from X_train
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # --- STAGE 3: ASSEMBLY AND SAVING ---
    # Helper function to pack data back into CSV
    def save_subset(X_arr, y_series, meta_series, name):
        # Reconstruct DataFrame
        df_scaled = pd.DataFrame(X_arr, columns=feature_names)
        # Restore meta and target columns (using .values to avoid index mismatch)
        # Important: use .values to reset indices, otherwise NaN will appear due to index mismatch
        df_scaled.insert(0, 'imgName', meta_series.values)
        df_scaled.insert(1, 'weight', y_series.values)
        
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'{name}.csv')
        df_scaled.to_csv(path, index=False)
        print(f"Saved: {path}")

    save_subset(X_train_scaled, y_train, meta_train, 'train_fixed')
    save_subset(X_val_scaled, y_val, meta_val, 'val_fixed')
    save_subset(X_test_scaled, y_test, meta_test, 'test_fixed')
    
    # Save the scaler itself for use on new real-world images
    joblib.dump(scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
    print("Scaler saved as feature_scaler.pkl")

if __name__ == "__main__":
    # Paths adapted to project structure
    input_csv = 'data/processed/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_features.csv'
    output_folder = 'data/processed/csvData/processed_fixed/'
    
    robust_data_pipeline(input_csv, output_folder)

