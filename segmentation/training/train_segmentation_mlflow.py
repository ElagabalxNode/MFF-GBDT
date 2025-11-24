import sys
import os
import argparse
import torch
import mlflow
import numpy as np
from datetime import datetime

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from segmentation.datasets.Penn_Fudan_dataset import PennFudanDataset
from segmentation.models.Mask_rcnn_Model import get_model_instance_segmentation
from utils.engine import train_one_epoch, evaluate
from utils import general as utils
from utils import transforms as T
from utils.mlflow_utils import setup_mlflow_experiment, log_params_from_args

def parse_args():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN for Segmentation with MLflow")
    
    # Data
    parser.add_argument("--data_path", type=str, default="data/raw/coco_sets/mixData", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers (default: 1)")
    parser.add_argument("--train_val_split", type=float, default=0.2, help="Validation split ratio")
    
    # Model
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (including background)")
    parser.add_argument("--weights_save_dir", type=str, default="data/models/segmentation/weight", help="Directory to save weights")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--step_size", type=int, default=3, help="LR scheduler step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR scheduler gamma")
    
    # MLflow
    parser.add_argument("--experiment_name", type=str, default="MFF_Segmentation_MaskRCNN", help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow run name")
    
    return parser.parse_args()

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    args = parse_args()
    setup_mlflow_experiment(args.experiment_name)
    
    # Create save directory
    if args.weights_save_dir and not os.path.exists(args.weights_save_dir):
        os.makedirs(args.weights_save_dir, exist_ok=True)

    with mlflow.start_run(run_name=args.run_name):
        log_params_from_args(args)
        
        # Device setup
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        print(f"Using device: {device}")
        mlflow.log_param("device", str(device))

        # Dataset
        dataset = PennFudanDataset(args.data_path, get_transform(train=True))
        dataset_test = PennFudanDataset(args.data_path, get_transform(train=False))

        # Split
        torch.manual_seed(1)
        indices = torch.randperm(len(dataset)).tolist()
        val_len = int(len(dataset) * args.train_val_split)
        dataset = torch.utils.data.Subset(dataset, indices[:-val_len])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-val_len:])
        
        print(f"Training samples: {len(dataset)}")
        print(f"Validation samples: {len(dataset_test)}")
        mlflow.log_param("train_samples", len(dataset))
        mlflow.log_param("val_samples", len(dataset_test))

        # Dataloaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            collate_fn=utils.collate_fn
        )
        
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            collate_fn=utils.collate_fn
        )

        # Model
        model = get_model_instance_segmentation(args.num_classes)
        model.to(device)

        # Optimizer & Scheduler
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        # Training Loop
        best_map = 0.0
        
        for epoch in range(args.epochs):
            # Train
            metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            lr_scheduler.step()
            
            # Log training losses
            # metric_logger contains SmoothedValue objects, we take the global average
            for name, meter in metric_logger.meters.items():
                mlflow.log_metric(f"train_{name}", meter.global_avg, step=epoch)

            # Evaluate
            coco_evaluator = evaluate(model, data_loader_test, device)
            
            # Log evaluation metrics
            # coco_eval.stats: 
            # 0: AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
            # 1: AP @[ IoU=0.50      | area=   all | maxDets=100 ]
            # ...
            if coco_evaluator is not None:
                # BBox metrics
                if 'bbox' in coco_evaluator.coco_eval:
                    stats = coco_evaluator.coco_eval['bbox'].stats
                    mlflow.log_metric("val_bbox_mAP", stats[0], step=epoch)
                    mlflow.log_metric("val_bbox_mAP_50", stats[1], step=epoch)
                
                # Mask metrics
                if 'segm' in coco_evaluator.coco_eval:
                    stats = coco_evaluator.coco_eval['segm'].stats
                    current_map = stats[0]
                    mlflow.log_metric("val_segm_mAP", current_map, step=epoch)
                    mlflow.log_metric("val_segm_mAP_50", stats[1], step=epoch)
                    
                    # Save best model based on Segmentation mAP
                    if current_map > best_map:
                        best_map = current_map
                        best_model_path = os.path.join(args.weights_save_dir, "best_model.pth")
                        torch.save(model.state_dict(), best_model_path)
                        print(f"New best model saved to {best_model_path} (mAP: {best_map:.4f})")
                        mlflow.log_metric("best_val_segm_mAP", best_map, step=epoch)
            
            # Save periodic checkpoints
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(args.weights_save_dir, f"model-{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                # mlflow.log_artifact(checkpoint_path) # Optional: might be too large to log every time

        # Log final best model to MLflow
        best_model_path = os.path.join(args.weights_save_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path)

        print("Training completed.")

if __name__ == "__main__":
    main()

