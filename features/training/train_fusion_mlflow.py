import sys
import os
import argparse
import time
import copy
import re
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from features.datasets.chicken200 import Chicken_200_trainset, Chicken_200_testset
from features.models.FusonNet import fusonnet50
from utils.mlflow_utils import setup_mlflow_experiment, log_params_from_args

# --- Helper Classes from original train_fusion.py ---

class myresnet_base(nn.Module):
    def __init__(self):
        super(myresnet_base, self).__init__()
        self.model_ft = models.resnet50(pretrained=False)
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

# --- Helper Functions ---

def fizze_resnet_parameter(model, fizze_resnet):
    if fizze_resnet:
        keylist = ['fc.weight', 'fc.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
        for name, param in model.named_parameters():
            if name not in keylist:
                param.requires_grad = False

def init_Fusonmodel(fusonnet, weightPath, fizze_resnet):
    # This logic is kept from original script to preserve weight loading behavior
    if not os.path.exists(weightPath):
        print(f"Warning: Initial weights not found at {weightPath}. initializing randomly.")
        return fusonnet

    model_pretrained = myresnet_base()
    try:
        model_pretrained.load_state_dict(torch.load(weightPath, map_location='cpu'))
    except Exception as e:
        print(f"Error loading pretrained weights: {e}")
        return fusonnet

    fusonnet_dict = fusonnet.state_dict()
    
    # Compare parameters and remove different ones (classification vs regression head differences likely)
    pretrained_dict = {k[9:]: v for k, v in model_pretrained.state_dict().items() if k.startswith('model_ft.')}
    if 'fc.weight' in pretrained_dict: pretrained_dict.pop('fc.weight')
    if 'fc.bias' in pretrained_dict: pretrained_dict.pop('fc.bias')
    
    fusonnet_dict.update(pretrained_dict)
    fusonnet.load_state_dict(fusonnet_dict)

    if fizze_resnet:
        fizze_resnet_parameter(fusonnet, True)

    return fusonnet

def get_manual_features(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Manual features CSV not found at {csv_path}")
    df = pd.read_csv(csv_path, index_col='imgName')
    return df

def train_model(model, dataloaders, loss_fn, optimizer, scheduler, device, num_epochs, manual_features_df):
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

            for inputs, labels, path in dataloaders[phase]:
                labels_gt = labels.numpy()
                weight_gt.extend(labels_gt)

                inputs, labels = inputs.to(device), labels.to(device)

                # Prepare manual features
                batch_manual_features = []
                for p in range(len(path)):
                    x_path = path[p]
                    base = os.path.basename(x_path)
                    # Normalize filename: remove suffix like "-1" before extension if present
                    key = re.sub(r'-\d+\.png$', '.png', base)
                    
                    if key in manual_features_df.index:
                        features = manual_features_df.loc[key].values
                    else:
                        # print(f"Warning: {key} not found in CSV, using zeros")
                        features = np.zeros(25) # 25 manual features
                    
                    batch_manual_features.append(features)
                
                manual_features_tensor = torch.as_tensor(np.array(batch_manual_features), dtype=torch.float32).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs, _ = model(inputs, manual_features_tensor)
                    loss = loss_fn(outputs.float(), labels.float())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                
                predict_weight = outputs.cpu().detach().numpy()
                weight_pr.extend(predict_weight)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            mae = mean_absolute_error(weight_gt, weight_pr)
            mse = mean_squared_error(weight_gt, weight_pr)
            rmse = mse ** 0.5
            r2 = r2_score(weight_gt, weight_pr)

            print(f"{phase} Loss: {epoch_loss:.4f} MAE: {mae:.4f} RMSE: {rmse:.4f}")

            # Log metrics to MLflow
            mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
            mlflow.log_metric(f"{phase}_mae", mae, step=epoch)
            mlflow.log_metric(f"{phase}_rmse", rmse, step=epoch)
            mlflow.log_metric(f"{phase}_r2", r2, step=epoch)

            # Deep copy the model if it's the best validation MAE
            if phase == "val" and mae < best_mae:
                best_mae = mae
                best_model_wts = copy.deepcopy(model.state_dict())
                mlflow.log_metric("best_val_mae", best_mae, step=epoch)
                # Save checkpoint locally and log as artifact
                torch.save(best_model_wts, "best_model.pth")
                mlflow.log_artifact("best_model.pth")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train FusionNet with MLflow")
    
    parser.add_argument("--manual_features_csv", type=str, 
                        default="data/processed/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_normal_features.csv",
                        help="Path to manual features CSV")
    parser.add_argument("--init_weights", type=str, 
                        default="data/models/features/resnet_base.pth", 
                        help="Path to initial ResNet weights")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--freeze_resnet", action="store_true", default=True, help="Freeze ResNet backbone")
    parser.add_argument("--experiment_name", type=str, default="MFF_FusionNet", help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow run name")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup MLflow
    setup_mlflow_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=args.run_name):
        log_params_from_args(args)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Data loading
        manual_features_df = get_manual_features(args.manual_features_csv)
        
        train_dataset = Chicken_200_trainset(transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((360, 640))
        ]))
        val_dataset = Chicken_200_testset(transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((360, 640))
        ]))
        
        dataloaders = {
            "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4),
            "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        }
        
        # Model setup
        fusonnet = fusonnet50()
        model = init_Fusonmodel(fusonnet, args.init_weights, args.freeze_resnet)
        model = model.to(device)
        
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            momentum=0.9
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=20)
        loss_fn = nn.L1Loss()
        
        print("Starting training...")
        model = train_model(model, dataloaders, loss_fn, optimizer, scheduler, device, args.epochs, manual_features_df)
        
        # Save final model
        mlflow.pytorch.log_model(model, "model")
        print("Training completed.")

if __name__ == '__main__':
    main()

