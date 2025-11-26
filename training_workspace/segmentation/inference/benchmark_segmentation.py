import time
import torch
import numpy as np
import argparse
import os
import sys
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training_workspace.segmentation.models.Mask_rcnn_Model import get_model_instance_segmentation
from training_workspace.utils import transforms as T

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Segmentation Models: Mask R-CNN vs YOLOv8 vs YOLO11")
    parser.add_argument("--maskrcnn_weights", type=str, default="training_workspace/data/models/segmentation/maskrcnn/best_model.pth", help="Path to Mask R-CNN weights")
    parser.add_argument("--yolo_nano_weights", type=str, default="training_workspace/data/models/segmentation/yolo/best_n.pt", help="Path to YOLOv8 Nano weights")
    parser.add_argument("--yolo_small_weights", type=str, default="training_workspace/data/models/segmentation/yolo/best_s.pt", help="Path to YOLOv8 Small weights")
    parser.add_argument("--yolo11_nano_weights", type=str, default="training_workspace/data/models/segmentation/yolo/best_yolo11n.pt", help="Path to YOLO11 Nano weights")
    parser.add_argument("--yolo11_small_weights", type=str, default="training_workspace/data/models/segmentation/yolo/best_yolo11s.pt", help="Path to YOLO11 Small weights")
    parser.add_argument("--test_dir", type=str, default="training_workspace/data/raw/coco_sets/mixData/origin", help="Directory with test images")
    parser.add_argument("--output_dir", type=str, default="training_workspace/data/outputs/exps/ablation_segmentation", help="Output directory for results")
    parser.add_argument("--device", type=str, default="", help="Device (cpu, cuda, mps)")
    parser.add_argument("--num_images", type=int, default=50, help="Number of images to use for benchmarking")
    return parser.parse_args()

def get_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_maskrcnn(weights_path, device):
    model = get_model_instance_segmentation(num_classes=2)
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Loaded Mask R-CNN from {weights_path}")
        except Exception as e:
            print(f"Error loading Mask R-CNN from {weights_path}: {e}")
            print("Using random weights.")
    else:
        print(f"Warning: Mask R-CNN weights not found at {weights_path}. Using random weights.")
    model.to(device)
    model.eval()
    return model

def load_yolo(weights_path, device, fallback_model="yolov8n-seg.pt", model_name="YOLO"):
    if os.path.exists(weights_path):
        model = YOLO(weights_path)
        print(f"Loaded {model_name} from {weights_path}")
    else:
        print(f"Warning: {model_name} weights not found at {weights_path}. Using pretrained {fallback_model}.")
        model = YOLO(fallback_model)
    return model

def benchmark_model(model, model_type, image_paths, device):
    times = []
    print(f"Benchmarking {model_type} on {device}...")
    
    # Warmup
    print("Warmup...")
    if len(image_paths) > 0:
        warmup_img = cv2.imread(str(image_paths[0]))
        if model_type == "Mask R-CNN":
            img_tensor = T.ToTensor()(warmup_img, {})[0].to(device).unsqueeze(0)
            with torch.no_grad():
                model(img_tensor)
        else:
            model.predict(warmup_img, device=device.type if device.type != 'cuda' else 0, verbose=False)

    print("Running inference...")
    for img_path in tqdm(image_paths):
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        start_time = time.time()
        
        if model_type == "Mask R-CNN":
            # Preprocess
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = T.ToTensor()(img_rgb, {})[0].to(device).unsqueeze(0)
            
            # Infer
            with torch.no_grad():
                model(img_tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
        elif "YOLO" in model_type:
            # Infer
            model.predict(img, device=device.type if device.type != 'cuda' else 0, verbose=False)
        
        end_time = time.time()
        times.append(end_time - start_time)
        
    avg_time = np.mean(times) if times else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    return avg_time, fps

def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Benchmarking on device: {device}")
    
    # Get images
    image_paths = sorted(list(Path(args.test_dir).glob("*.png")) + list(Path(args.test_dir).glob("*.jpg")))
    if len(image_paths) > args.num_images:
        image_paths = image_paths[:args.num_images]
    
    print(f"Found {len(image_paths)} images for benchmarking.")
    
    results = {
        "Model": [],
        "Avg Latency (s)": [],
        "FPS": [],
        "Device": []
    }

    # 1. Benchmark Mask R-CNN
    maskrcnn = load_maskrcnn(args.maskrcnn_weights, device)
    mrcnn_time, mrcnn_fps = benchmark_model(maskrcnn, "Mask R-CNN", image_paths, device)
    results["Model"].append("Mask R-CNN")
    results["Avg Latency (s)"].append(mrcnn_time)
    results["FPS"].append(mrcnn_fps)
    results["Device"].append(str(device))
    
    del maskrcnn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 2. Benchmark YOLOv8 Nano
    yolo_n = load_yolo(args.yolo_nano_weights, device, fallback_model="yolov8n-seg.pt", model_name="YOLOv8")
    yolo_n_time, yolo_n_fps = benchmark_model(yolo_n, "YOLOv8-Nano", image_paths, device)
    results["Model"].append("YOLOv8-Nano")
    results["Avg Latency (s)"].append(yolo_n_time)
    results["FPS"].append(yolo_n_fps)
    results["Device"].append(str(device))
    
    del yolo_n
    
    # 3. Benchmark YOLOv8 Small
    yolo_s = load_yolo(args.yolo_small_weights, device, fallback_model="yolov8s-seg.pt", model_name="YOLOv8")
    yolo_s_time, yolo_s_fps = benchmark_model(yolo_s, "YOLOv8-Small", image_paths, device)
    results["Model"].append("YOLOv8-Small")
    results["Avg Latency (s)"].append(yolo_s_time)
    results["FPS"].append(yolo_s_fps)
    results["Device"].append(str(device))
    
    del yolo_s
    
    # 4. Benchmark YOLO11 Nano
    yolo11_n = load_yolo(args.yolo11_nano_weights, device, fallback_model="yolo11n-seg.pt", model_name="YOLO11")
    yolo11_n_time, yolo11_n_fps = benchmark_model(yolo11_n, "YOLO11-Nano", image_paths, device)
    results["Model"].append("YOLO11-Nano")
    results["Avg Latency (s)"].append(yolo11_n_time)
    results["FPS"].append(yolo11_n_fps)
    results["Device"].append(str(device))
    
    del yolo11_n
    
    # 5. Benchmark YOLO11 Small
    yolo11_s = load_yolo(args.yolo11_small_weights, device, fallback_model="yolo11s-seg.pt", model_name="YOLO11")
    yolo11_s_time, yolo11_s_fps = benchmark_model(yolo11_s, "YOLO11-Small", image_paths, device)
    results["Model"].append("YOLO11-Small")
    results["Avg Latency (s)"].append(yolo11_s_time)
    results["FPS"].append(yolo11_s_fps)
    results["Device"].append(str(device))

    # Save Results
    df = pd.DataFrame(results)
    print("\n=== Benchmark Results ===")
    print(df)
    
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Plot
    plt.figure(figsize=(14, 6))
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    bars = plt.bar(df["Model"], df["FPS"], color=colors[:len(df)])
    plt.ylabel("FPS (Higher is better)")
    plt.title(f"Inference Speed Comparison ({device})")
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold')
        
    plot_path = os.path.join(args.output_dir, "fps_comparison.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
