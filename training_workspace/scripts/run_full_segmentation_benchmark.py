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
    parser.add_argument("--skip_yolo", action="store_true", help="Skip all YOLO training (v8 and 11)")
    parser.add_argument("--skip_yolo8", action="store_true", help="Skip YOLOv8 training (train only YOLO11)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    args = parser.parse_args()

    python_exe = sys.executable

    # 1. Train Mask R-CNN
    if not args.skip_maskrcnn:
        cmd = f'{python_exe} training_workspace/segmentation/training/train_segmentation_mlflow.py --epochs {args.epochs} --batch_size 2 --experiment_name "MFF_Benchmark_MaskRCNN"'
        run_command(cmd, "Train Mask R-CNN Baseline")
    
    # 2. Train YOLOv8-Nano
    if not args.skip_yolo and not args.skip_yolo8:
        cmd = f'{python_exe} training_workspace/segmentation/training/train_yolo_mlflow.py --model yolo11n-seg.pt --epochs {args.epochs} --batch_size 8 --workers 1 --experiment_name "MFF_Benchmark_YOLOv8" --save_dir "training_workspace/data/models/segmentation/yolo/nano"'
        run_command(cmd, "Train YOLOv8-Nano")
        
        # Rename best.pt to best_n.pt for benchmark script
        src = "training_workspace/data/models/segmentation/yolo/nano/best.pt"
        dst = "training_workspace/data/models/segmentation/yolo/best_n.pt"
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")

    # 3. Train YOLOv8-Small
    if not args.skip_yolo and not args.skip_yolo8:
        cmd = f'{python_exe} training_workspace/segmentation/training/train_yolo_mlflow.py --model yolov8s-seg.pt --epochs {args.epochs} --batch_size 8 --workers 1 --experiment_name "MFF_Benchmark_YOLOv8" --save_dir "training_workspace/data/models/segmentation/yolo/small"'
        run_command(cmd, "Train YOLOv8-Small")
        
        # Rename best.pt to best_s.pt for benchmark script
        src = "training_workspace/data/models/segmentation/yolo/small/best.pt"
        dst = "training_workspace/data/models/segmentation/yolo/best_s.pt"
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")

    # 4. Train YOLO11-Nano
    if not args.skip_yolo:
        cmd = f'{python_exe} training_workspace/segmentation/training/train_yolo_mlflow.py --model yolo11n-seg.pt --epochs {args.epochs} --batch_size 8 --workers 1 --experiment_name "MFF_Benchmark_YOLO11" --save_dir "training_workspace/data/models/segmentation/yolo/yolo11n"'
        run_command(cmd, "Train YOLO11-Nano")
        
        # Rename best.pt to best_yolo11n.pt for benchmark script
        src = "training_workspace/data/models/segmentation/yolo/yolo11n/best.pt"
        dst = "training_workspace/data/models/segmentation/yolo/best_yolo11n.pt"
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")

    # 5. Train YOLO11-Small
    if not args.skip_yolo:
        cmd = f'{python_exe} training_workspace/segmentation/training/train_yolo_mlflow.py --model yolo11s-seg.pt --epochs {args.epochs} --batch_size 8 --workers 1 --experiment_name "MFF_Benchmark_YOLO11" --save_dir "training_workspace/data/models/segmentation/yolo/yolo11s"'
        run_command(cmd, "Train YOLO11-Small")
        
        # Rename best.pt to best_yolo11s.pt for benchmark script
        src = "training_workspace/data/models/segmentation/yolo/yolo11s/best.pt"
        dst = "training_workspace/data/models/segmentation/yolo/best_yolo11s.pt"
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")

    # 6. Run Benchmark
    cmd = f'{python_exe} training_workspace/segmentation/inference/benchmark_segmentation.py --maskrcnn_weights "training_workspace/data/models/segmentation/maskrcnn/best_model.pth" --yolo_nano_weights "training_workspace/data/models/segmentation/yolo/best_n.pt" --yolo_small_weights "training_workspace/data/models/segmentation/yolo/best_s.pt" --yolo11_nano_weights "training_workspace/data/models/segmentation/yolo/best_yolo11n.pt" --yolo11_small_weights "training_workspace/data/models/segmentation/yolo/best_yolo11s.pt" --output_dir "training_workspace/data/outputs/exps/benchmark_final"'
    run_command(cmd, "Run Comparative Benchmark")

if __name__ == "__main__":
    main()

