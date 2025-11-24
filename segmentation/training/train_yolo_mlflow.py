import argparse
import os
import sys
import shutil
import mlflow
from ultralytics import YOLO
from pathlib import Path

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.mlflow_utils import setup_mlflow_experiment, log_params_from_args

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg with MLflow")
    
    # Data and Model
    parser.add_argument("--data_yaml", type=str, default="data/processed/yolo_dataset/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Pretrained model (e.g., yolov8n-seg.pt, yolov8s-seg.pt)")
    parser.add_argument("--img_size", type=int, default=640, help="Image size")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=1, help="Number of data loading workers (0-1 for Windows stability)")
    parser.add_argument("--device", type=str, default="", help="Device (cpu, cuda, mps)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    # MLflow
    parser.add_argument("--experiment_name", type=str, default="MFF_Segmentation_YOLOv8", help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--save_dir", type=str, default="data/models/segmentation/yolo", help="Directory to save final weights")

    return parser.parse_args()

class MLflowCallback:
    """Callback to log YOLOv8 metrics to MLflow"""
    def __init__(self):
        pass

    def on_train_epoch_end(self, trainer):
        # Log training metrics
        metrics = trainer.metrics
        for k, v in metrics.items():
            # Clean up metric names for consistency
            name = k.replace("(B)", "_box").replace("(M)", "_mask")
            mlflow.log_metric(name, v, step=trainer.epoch)
        
        # Log loss values from trainer.label_loss_items (box, cls, dfl) if available
        # YOLOv8 logs losses differently, accessing via validator results is safer for validation loss

    def on_fit_epoch_end(self, trainer):
        pass

def main():
    args = parse_args()
    setup_mlflow_experiment(args.experiment_name)
    
    # Check device
    device = args.device
    if not device:
        import torch
        if torch.cuda.is_available():
            device = "0" # YOLO uses "0" for cuda:0
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Using device: {device}")

    with mlflow.start_run(run_name=args.run_name):
        # Log params, excluding 'device' to avoid duplication
        log_params_from_args(args, exclude=['device'])
        mlflow.log_param("device", device)
        
        # Initialize YOLO model
        model = YOLO(args.model)
        
        # Disable Ultralytics MLflow integration to prevent conflict
        from ultralytics import settings
        settings.update({'mlflow': False})
        
        # Add callback (Ultralytics doesn't support direct callback class injection in 'train', 
        # but we can log manually or use their integration. For simplicity, we'll wrap training)
        
        # Train
        results = model.train(
            data=args.data_yaml,
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch_size,
            workers=args.workers,
            device=device,
            patience=args.patience,
            project="runs/segment", # Temporary YOLO output
            name="train_run",
            exist_ok=True,
            verbose=True
        )
        
        # Log final metrics from results object
        # results.results_dict contains final metrics
        if hasattr(results, 'results_dict'):
             for k, v in results.results_dict.items():
                 # Clean metric name: replace (B) -> _box, (M) -> _mask, remove other invalid chars
                 clean_name = k.replace("(B)", "_box").replace("(M)", "_mask").replace("(", "_").replace(")", "")
                 mlflow.log_metric(f"final_{clean_name}", v)

        # Log metrics history (manually reading CSV is reliable)
        results_dir = Path(results.save_dir)
        csv_path = results_dir / "results.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns] # Clean whitespace
            
            # Log all epochs
            for _, row in df.iterrows():
                epoch = int(row['epoch'])
                for col in df.columns:
                    if col != 'epoch':
                        # Clean metric name for MLflow compatibility
                        clean_col = col.replace("(B)", "_box").replace("(M)", "_mask").replace("(", "_").replace(")", "")
                        mlflow.log_metric(clean_col, row[col], step=epoch)

        # Save best model to project structure
        os.makedirs(args.save_dir, exist_ok=True)
        best_pt = results_dir / "weights" / "best.pt"
        final_path = os.path.join(args.save_dir, "best.pt")
        
        if best_pt.exists():
            shutil.copy(best_pt, final_path)
            print(f"Best model saved to {final_path}")
            mlflow.log_artifact(final_path)
        
        # Log confusion matrix and charts
        for img_path in results_dir.glob("*.png"):
            mlflow.log_artifact(str(img_path))

        print("Training completed.")

if __name__ == "__main__":
    main()

