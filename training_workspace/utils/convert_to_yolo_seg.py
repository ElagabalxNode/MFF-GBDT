import os
import cv2
import numpy as np
import shutil
import random
import yaml
from tqdm import tqdm
from pathlib import Path

def convert_mask_to_yolo_polygon(mask, class_id=0):
    """
    Convert a binary mask to YOLO polygon format (normalized coordinates).
    """
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 50:  # Filter small noise
            continue
        
        # Simplify contour
        contour = cv2.approxPolyDP(contour, 0.005 * cv2.arcLength(contour, True), True)
        
        if len(contour) < 3:  # Need at least 3 points for a polygon
            continue
            
        # Normalize coordinates
        polygon = contour.flatten().astype(float)
        polygon[0::2] /= width  # x coordinates
        polygon[1::2] /= height # y coordinates
        
        # Clip to [0, 1] to avoid errors
        polygon = np.clip(polygon, 0, 1)
        
        polygons.append([class_id] + polygon.tolist())
        
    return polygons

def main():
    # Configuration
    root_dir = Path("data/raw/coco_sets/mixData")
    output_dir = Path("data/processed/yolo_dataset")
    img_dir = root_dir / "origin"
    mask_dir = root_dir / "mask"
    
    # Clear existing output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Create YOLO structure
    (output_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images/val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/val").mkdir(parents=True, exist_ok=True)
    
    # Get file lists
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    assert len(img_files) == len(mask_files), "Mismatch between images and masks count"
    
    # Shuffle and split
    data_pairs = list(zip(img_files, mask_files))
    random.seed(42)
    random.shuffle(data_pairs)
    
    split_idx = int(len(data_pairs) * 0.8)
    train_pairs = data_pairs[:split_idx]
    val_pairs = data_pairs[split_idx:]
    
    print(f"Total images: {len(data_pairs)}")
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Process function
    def process_split(pairs, split_name):
        print(f"Processing {split_name} set...")
        for img_file, mask_file in tqdm(pairs):
            # Copy image
            src_img = img_dir / img_file
            dst_img = output_dir / "images" / split_name / img_file
            shutil.copy(src_img, dst_img)
            
            # Process mask
            mask_path = mask_dir / mask_file
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) # PennFudan: 0=bg, 1..N=instances
            
            # Get unique object IDs (excluding background 0)
            obj_ids = np.unique(mask_img)
            obj_ids = obj_ids[obj_ids > 0]
            
            label_content = []
            
            for obj_id in obj_ids:
                # Create binary mask for this instance
                instance_mask = (mask_img == obj_id).astype(np.uint8)
                
                # Convert to polygon
                polygons = convert_mask_to_yolo_polygon(instance_mask, class_id=0) # Class 0 = chicken
                
                for poly in polygons:
                    line = " ".join(map(str, poly))
                    label_content.append(line)
            
            # Save label file
            label_file = dst_img.stem + ".txt"
            label_path = output_dir / "labels" / split_name / label_file
            
            with open(label_path, "w") as f:
                f.write("\n".join(label_content))

    process_split(train_pairs, "train")
    process_split(val_pairs, "val")
    
    # Create dataset.yaml
    yaml_content = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "chicken"
        }
    }
    
    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    print(f"\nDataset conversion complete. Saved to {output_dir}")
    print(f"Config file: {output_dir / 'dataset.yaml'}")

if __name__ == "__main__":
    main()

