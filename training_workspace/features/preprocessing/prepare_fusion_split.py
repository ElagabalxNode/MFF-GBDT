"""
Prepare data splits for FusionNet training with proper normalization.

This script ensures:
1. Train/Test split with fixed random_state=42
2. MinMaxScaler is fit ONLY on train data (no data leakage)
3. Image list is aligned with CSV features for consistent splits
4. Only includes samples where corresponding images exist

Usage:
    python prepare_fusion_split.py --output_dir training_workspace/data/processed/csvData/fusion_split/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import argparse

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Feature columns (excluding weight and imgName)
MANUAL_FEATURE_COLUMNS = [
    'area', 'perimeter', 'min_rect_width', 'min_rect_high', 'approx_area',
    'approx_perimeter', 'extent', 'hull_perimeter', 'hull_area', 'solidity',
    'max_defect_dist', 'sum_defect_dist', 'equi_diameter', 'ellipse_long',
    'ellipse_short', 'eccentricity', 'volume', 'maxHeight', 'minHeight',
    'max2min', 'meanHeight', 'mean2min', 'mean2max', 'stdHeight', 'heightSum'
]


def get_existing_imgnames(mask_img_dir: str) -> set:
    """
    Get set of imgNames (CSV format) for which image files exist.
    
    File: "1.1_Depth-0-0.png" -> imgName: "1.1_Depth-0.png"
    """
    existing = set()
    
    if not os.path.exists(mask_img_dir):
        print(f"Warning: maskImg directory not found: {mask_img_dir}")
        return existing
    
    for filename in os.listdir(mask_img_dir):
        if not filename.endswith('.png'):
            continue
        
        # Convert filename to imgName format
        if filename.endswith('-0.png'):
            imgName = filename[:-6] + '.png'
        else:
            imgName = filename
        
        existing.add(imgName)
    
    return existing


def prepare_fusion_data(
    input_csv: str,
    output_dir: str,
    mask_img_dir: str = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED
):
    """
    Create proper train/test split with correct normalization.
    
    Args:
        input_csv: Path to raw features CSV (NOT normalized)
        output_dir: Directory to save processed files
        mask_img_dir: Path to maskImg folder (filters to existing images only)
        test_size: Fraction for test set (default 0.2)
        random_state: Random seed for reproducibility (default 42)
    
    Saves:
        - train_features.csv: Normalized train features
        - test_features.csv: Normalized test features  
        - train_raw.csv: Raw (unnormalized) train features
        - test_raw.csv: Raw (unnormalized) test features
        - feature_scaler.pkl: Fitted MinMaxScaler
        - split_info.csv: Image names with train/test labels
    """
    print(f"Loading raw features from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Total samples in CSV: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Validate required columns
    if 'weight' not in df.columns:
        raise ValueError("CSV must contain 'weight' column")
    if 'imgName' not in df.columns:
        raise ValueError("CSV must contain 'imgName' column")
    
    # Filter to only existing images
    if mask_img_dir:
        existing_imgs = get_existing_imgnames(mask_img_dir)
        print(f"Found {len(existing_imgs)} existing images in {mask_img_dir}")
        
        before_count = len(df)
        df = df[df['imgName'].isin(existing_imgs)]
        after_count = len(df)
        
        if before_count != after_count:
            print(f"Filtered: {before_count} -> {after_count} samples (removed {before_count - after_count} missing images)")
    
    # --- STAGE 1: SPLIT DATA ---
    print(f"\nSplitting data with random_state={random_state}, test_size={test_size}")
    
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Train samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")
    
    # --- STAGE 2: SEPARATE FEATURES AND METADATA ---
    def extract_features(data):
        """Extract feature matrix, target, and metadata."""
        y = data['weight'].values
        img_names = data['imgName'].values
        
        # Select only feature columns
        X = data[MANUAL_FEATURE_COLUMNS].values
        return X, y, img_names
    
    X_train, y_train, img_train = extract_features(df_train)
    X_test, y_test, img_test = extract_features(df_test)
    
    print(f"\nFeature shape: {X_train.shape[1]} features")
    
    # --- STAGE 3: FIT SCALER ON TRAIN ONLY ---
    print("\nFitting MinMaxScaler on TRAIN data only...")
    scaler = MinMaxScaler()
    scaler.fit(X_train)  # FIT ONLY ON TRAIN!
    
    # Transform both sets using scaler fitted on train
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- STAGE 4: SAVE RESULTS ---
    os.makedirs(output_dir, exist_ok=True)
    
    def create_df(X, y, img_names, columns):
        """Reconstruct DataFrame with imgName and weight."""
        df_out = pd.DataFrame(X, columns=columns)
        df_out.insert(0, 'weight', y)
        df_out.insert(0, 'imgName', img_names)
        return df_out
    
    # Save normalized features
    df_train_norm = create_df(X_train_scaled, y_train, img_train, MANUAL_FEATURE_COLUMNS)
    df_test_norm = create_df(X_test_scaled, y_test, img_test, MANUAL_FEATURE_COLUMNS)
    
    train_norm_path = os.path.join(output_dir, 'train_features_normalized.csv')
    test_norm_path = os.path.join(output_dir, 'test_features_normalized.csv')
    
    df_train_norm.to_csv(train_norm_path, index=False)
    df_test_norm.to_csv(test_norm_path, index=False)
    print(f"Saved normalized train: {train_norm_path}")
    print(f"Saved normalized test: {test_norm_path}")
    
    # Save raw (unnormalized) features for reference
    df_train_raw = create_df(X_train, y_train, img_train, MANUAL_FEATURE_COLUMNS)
    df_test_raw = create_df(X_test, y_test, img_test, MANUAL_FEATURE_COLUMNS)
    
    df_train_raw.to_csv(os.path.join(output_dir, 'train_features_raw.csv'), index=False)
    df_test_raw.to_csv(os.path.join(output_dir, 'test_features_raw.csv'), index=False)
    print("Saved raw (unnormalized) train/test CSVs")
    
    # Save scaler for inference
    scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler: {scaler_path}")
    
    # Save split info (imgName -> split mapping)
    split_info = pd.DataFrame({
        'imgName': np.concatenate([img_train, img_test]),
        'split': ['train'] * len(img_train) + ['test'] * len(img_test)
    })
    split_info_path = os.path.join(output_dir, 'split_info.csv')
    split_info.to_csv(split_info_path, index=False)
    print(f"Saved split info: {split_info_path}")
    
    # Print statistics
    print("\n" + "="*50)
    print("SPLIT STATISTICS")
    print("="*50)
    print(f"Train weight: mean={y_train.mean():.3f}, std={y_train.std():.3f}")
    print(f"Test weight:  mean={y_test.mean():.3f}, std={y_test.std():.3f}")
    
    return {
        'train_csv': train_norm_path,
        'test_csv': test_norm_path,
        'scaler_path': scaler_path,
        'split_info_path': split_info_path,
        'train_size': len(df_train),
        'test_size': len(df_test)
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare FusionNet data with proper split and normalization")
    
    parser.add_argument(
        "--input_csv", 
        type=str,
        default="training_workspace/data/processed/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_features.csv",
        help="Path to raw (unnormalized) features CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_workspace/data/processed/csvData/fusion_split/",
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--mask_img_dir",
        type=str,
        default="training_workspace/data/outputs/exps/data_origin_weight_best_n-result/maskImg",
        help="Path to maskImg folder (filters to existing images only)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=TEST_SIZE,
        help=f"Test set fraction (default: {TEST_SIZE})"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    
    args = parser.parse_args()
    
    result = prepare_fusion_data(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        mask_img_dir=args.mask_img_dir,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    print("\n" + "="*50)
    print("DONE!")
    print("="*50)
    print(f"\nTo train FusionNet with this data, run:")
    print(f"  python training_workspace/features/training/train_fusion_mlflow.py \\")
    print(f"    --train_features_csv {result['train_csv']} \\")
    print(f"    --test_features_csv {result['test_csv']} \\")
    print(f"    --split_info_csv {result['split_info_path']}")


if __name__ == "__main__":
    main()
