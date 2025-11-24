import subprocess
import sys
import os
import argparse

def run_command(command, description):
    print(f"\n{'='*80}")
    print(f"Step: {description}")
    print(f"Command: {command}")
    print(f"{'='*80}")
    
    try:
        subprocess.check_call(command, shell=True)
        print(f"\n✅ {description} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error executing {description}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Full Segmentation Benchmark")
    parser.add_argument("--skip_maskrcnn", action="store_true", help="Skip Mask R-CNN training")
    parser.add_argument("--skip_yolo", action="store_true", help="Skip YOLO training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    args = parser.parse_args()

    python_exe = sys.executable

    # 1. Train Mask R-CNN
    if not args.skip_maskrcnn:
        cmd = f'{python_exe} segmentation/training/train_segmentation_mlflow.py --epochs {args.epochs} --batch_size 2 --experiment_name "MFF_Benchmark_MaskRCNN"'
        run_command(cmd, "Train Mask R-CNN Baseline")
    
    # 2. Train YOLOv8-Nano
    if not args.skip_yolo:
        cmd = f'{python_exe} segmentation/training/train_yolo_mlflow.py --model yolov8n-seg.pt --epochs {args.epochs} --batch_size 8 --workers 1 --experiment_name "MFF_Benchmark_YOLOv8" --save_dir "data/models/segmentation/yolo/nano"'
        run_command(cmd, "Train YOLOv8-Nano")
        
        # Rename best.pt to best_n.pt for benchmark script
        src = "data/models/segmentation/yolo/nano/best.pt"
        dst = "data/models/segmentation/yolo/best_n.pt"
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")

    # 3. Train YOLOv8-Small
    if not args.skip_yolo:
        cmd = f'{python_exe} segmentation/training/train_yolo_mlflow.py --model yolov8s-seg.pt --epochs {args.epochs} --batch_size 8 --workers 1 --experiment_name "MFF_Benchmark_YOLOv8" --save_dir "data/models/segmentation/yolo/small"'
        run_command(cmd, "Train YOLOv8-Small")
        
        # Rename best.pt to best_s.pt for benchmark script
        src = "data/models/segmentation/yolo/small/best.pt"
        dst = "data/models/segmentation/yolo/best_s.pt"
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")

    # 4. Run Benchmark
    cmd = f'{python_exe} segmentation/inference/benchmark_segmentation.py --maskrcnn_weights "data/models/segmentation/weight/best_model.pth" --yolo_nano_weights "data/models/segmentation/yolo/best_n.pt" --yolo_small_weights "data/models/segmentation/yolo/best_s.pt" --output_dir "data/outputs/exps/benchmark_final"'
    run_command(cmd, "Run Comparative Benchmark")

if __name__ == "__main__":
    main()

