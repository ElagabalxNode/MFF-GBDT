import sys
import os
import argparse
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.pytorch

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from features.datasets.chicken200 import Chicken_200_trainset, Chicken_200_testset
from utils.mlflow_utils import setup_mlflow_experiment, log_params_from_args

# --- Model Definition ---
class myresnet(nn.Module):
    def __init__(self, pretrained=False):
        super(myresnet, self).__init__()
        self.model_ft = models.resnet50(pretrained=pretrained)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.model_ft(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = torch.flatten(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 for Weight Estimation with MLflow")
    
    # Data
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained ResNet weights")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze ResNet backbone layers")
    
    # MLflow
    parser.add_argument("--experiment_name", type=str, default="MFF_ResNet_Training", help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow run name")
    
    return parser.parse_args()

def train_model(model, dataloaders, loss_fn, optimizer, scheduler, device, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print('-' * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            weight_gt = []
            weight_pr = []

            for inputs, labels, _ in dataloaders[phase]: # _ is path, ignored
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                
                if phase == "train" and scheduler:
                    scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                
                weight_gt.extend(labels.cpu().numpy())
                weight_pr.extend(outputs.detach().cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            mae = mean_absolute_error(weight_gt, weight_pr)
            mse = mean_squared_error(weight_gt, weight_pr)
            rmse = mse ** 0.5
            r2 = r2_score(weight_gt, weight_pr)

            print(f"{phase} Loss: {epoch_loss:.4f} MAE: {mae:.4f} RMSE: {rmse:.4f} R2: {r2:.4f}")

            # Log metrics
            mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
            mlflow.log_metric(f"{phase}_mae", mae, step=epoch)
            mlflow.log_metric(f"{phase}_rmse", rmse, step=epoch)
            mlflow.log_metric(f"{phase}_r2", r2, step=epoch)

            if phase == "val" and mae < best_mae:
                best_mae = mae
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best model found! MAE: {best_mae:.4f}")
                mlflow.log_metric("best_val_mae", best_mae, step=epoch)
                
                # Save checkpoint
                torch.save(best_model_wts, "best_model.pth")
                mlflow.log_artifact("best_model.pth")

    print(f"Best Validation MAE: {best_mae:.4f}")
    model.load_state_dict(best_model_wts)
    return model

def main():
    args = parse_args()
    setup_mlflow_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=args.run_name):
        log_params_from_args(args)
        
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"Using device: {device}")
        mlflow.log_param("device", str(device))

        # Data Transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((360, 640))
        ])

        # Datasets
        train_dataset = Chicken_200_trainset(transform=transform)
        val_dataset = Chicken_200_testset(transform=transform)
        
        dataloaders = {
            "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
            "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        }

        # Model
        model = myresnet(pretrained=args.pretrained)
        
        if args.freeze_backbone:
            for param in model.model_ft.parameters():
                param.requires_grad = False
            # Unfreeze custom heads
            for param in model.model_ft.fc.parameters(): param.requires_grad = True
            for param in model.fc2.parameters(): param.requires_grad = True
            for param in model.fc3.parameters(): param.requires_grad = True
            
        model = model.to(device)

        # Optimization
        loss_fn = nn.L1Loss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=args.lr, momentum=args.momentum)
        
        # Original script didn't use scheduler, but fusion did. Keeping it simple like original for now,
        # or we can add one. Original had no scheduler in "makeEnv" section, but "train_fusion" did.
        # Adding simple scheduler just in case, or None.
        # The original train_resnet.py did NOT use a scheduler in the main loop logic visible.
        scheduler = None 

        # Train
        model = train_model(model, dataloaders, loss_fn, optimizer, scheduler, device, args.epochs)
        
        # Log final model
        mlflow.pytorch.log_model(model, "model")
        print("Training completed.")

if __name__ == "__main__":
    main()

