import sys
import os
import argparse
import time
import copy
import tempfile
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
# Go up 3 levels: training/ -> features/ -> training_workspace/ -> project_root/
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training_workspace.features.datasets.chicken200 import Chicken_200_trainset, Chicken_200_testset
from training_workspace.utils.mlflow_utils import setup_mlflow_experiment, log_params_from_args

def load_weights_from_mlflow(run_id: str, tracking_uri: str = None):
    """
    Load weights from MLflow run and save to temporary file
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Load model from MLflow
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading weights from MLflow: {model_uri}")
    
    try:
        # Load the model
        model = mlflow.pytorch.load_model(model_uri)
        
        # Extract state_dict
        state_dict = model.state_dict()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
        temp_path = temp_file.name
        temp_file.close()
        
        torch.save(state_dict, temp_path)
        print(f"Weights extracted and saved to temporary file: {temp_path}")
        
        return temp_path
    except Exception as e:
        raise RuntimeError(f"Failed to load model from MLflow run {run_id}: {e}")

# --- Model Definition ---
class myresnet(nn.Module):
    def __init__(self, pretrained=False, dropout=0.0, pretrained_path=None):
        super(myresnet, self).__init__()
        # Create model without pretrained weights
        # Use new API: weights=None instead of deprecated pretrained=False
        self.model_ft = models.resnet50(weights=None)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1024)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc3 = nn.Linear(512, 1)
        
        # Load pretrained weights
        if pretrained_path:
            self._load_local_pretrained(pretrained_path)
        elif pretrained:
            # Standard loading via torchvision (requires internet)
            # Use new API: weights instead of pretrained
            from torchvision.models import ResNet50_Weights
            pretrained_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self._load_backbone_weights(pretrained_model)
    
    def _load_local_pretrained(self, pretrained_path):
        """Load pretrained weights from local file"""
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model file not found: {pretrained_path}")
        
        print(f"Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Extract state_dict from checkpoint
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Determine key format and adapt to our structure
        # Standard torchvision weights: conv1.weight, bn1.weight, layer1.0.conv1.weight, fc.weight
        # Our structure: model_ft.conv1.weight, model_ft.bn1.weight, model_ft.layer1.0.conv1.weight
        
        # Check key format in file
        sample_keys = list(state_dict.keys())[:5]
        has_model_ft_prefix = any(key.startswith('model_ft.') for key in sample_keys)
        
        backbone_state_dict = {}
        for key, value in state_dict.items():
            # Skip fc layer (it will be replaced with our custom one)
            if key == 'fc.weight' or key == 'fc.bias':
                continue
            
            # Skip custom layers (fc2, fc3, relu, dropout)
            if key.startswith('fc2') or key.startswith('fc3') or key.startswith('relu') or key.startswith('dropout'):
                continue
            
            # Adapt keys depending on file format
            if has_model_ft_prefix:
                # File already contains keys with model_ft. prefix
                if key.startswith('model_ft.'):
                    # Remove prefix for loading into self.model_ft (which expects keys without prefix)
                    new_key = key.replace('model_ft.', '', 1)
                else:
                    # Key without prefix - use as is
                    new_key = key
            else:
                # Standard torchvision format - keys without prefix
                # Use as is for loading into self.model_ft
                if key.startswith('model_ft.'):
                    # If prefix exists, remove it
                    new_key = key.replace('model_ft.', '', 1)
                else:
                    new_key = key
            
            backbone_state_dict[new_key] = value
        
        # Load weights into backbone (self.model_ft expects keys WITHOUT model_ft. prefix)
        missing_keys, unexpected_keys = self.model_ft.load_state_dict(backbone_state_dict, strict=False)
        if missing_keys:
            print(f"Warning: {len(missing_keys)} missing keys (showing first 5): {missing_keys[:5]}")
        if unexpected_keys:
            print(f"Warning: {len(unexpected_keys)} unexpected keys (showing first 5): {unexpected_keys[:5]}")
        print(f"Successfully loaded local pretrained weights ({len(backbone_state_dict)} layers)")
    
    def _load_backbone_weights(self, pretrained_model):
        """Load backbone weights from pretrained model"""
        # Save our custom fc
        custom_fc = self.model_ft.fc
        
        # Load weights from pretrained model
        pretrained_state = pretrained_model.state_dict()
        model_state = self.model_ft.state_dict()
        
        # Copy weights, excluding fc layer
        for key in pretrained_state:
            if key != 'fc.weight' and key != 'fc.bias':
                if key in model_state:
                    model_state[key] = pretrained_state[key]
        
        self.model_ft.load_state_dict(model_state)
        # Restore custom fc
        self.model_ft.fc = custom_fc

    def forward(self, x):
        x = self.model_ft(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = torch.flatten(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 for Weight Estimation with MLflow")
    
    # Data
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # Data Augmentation
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation for training")
    parser.add_argument("--hflip_prob", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--rotation_degrees", type=int, default=10, help="Random rotation degrees")
    parser.add_argument("--jitter_brightness", type=float, default=0.2, help="ColorJitter brightness")
    parser.add_argument("--jitter_contrast", type=float, default=0.2, help="ColorJitter contrast")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in FC layers")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained ResNet weights (requires internet)")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to local pretrained ResNet50 weights (.pth file)")
    parser.add_argument("--init_weights_from_mlflow", type=str, default=None, help="MLflow run ID to load weights from")
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze ResNet backbone layers")
    
    # Early Stopping
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Minimum improvement for early stopping")
    
    # MLflow
    parser.add_argument("--experiment_name", type=str, default="MFF_ResNet_Training", help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow run name")
    
    return parser.parse_args()

def train_model(model, dataloaders, loss_fn, optimizer, scheduler, device, num_epochs, patience=15, min_delta=0.001):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = float('inf')
    epochs_without_improvement = 0
    
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

            if phase == "val":
                # Early stopping check
                if mae < best_mae - min_delta:
                    best_mae = mae
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                    print(f"New best model found! MAE: {best_mae:.4f}")
                    mlflow.log_metric("best_val_mae", best_mae, step=epoch)
                    
                    # Save checkpoint
                    torch.save(best_model_wts, "best_model.pth")
                    mlflow.log_artifact("best_model.pth")
                else:
                    epochs_without_improvement += 1
                    print(f"No improvement for {epochs_without_improvement}/{patience} epochs")
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    mlflow.log_metric("stopped_epoch", epoch)
                    model.load_state_dict(best_model_wts)
                    return model

    print(f"Best Validation MAE: {best_mae:.4f}")
    model.load_state_dict(best_model_wts)
    return model

def main():
    args = parse_args()
    setup_mlflow_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=args.run_name):
        # Determine weights path: MLflow takes precedence over local file
        # Also normalize pretrained_path if provided
        pretrained_path_normalized = None
        temp_weights_file = None
        
        if args.init_weights_from_mlflow:
            # Load weights from MLflow
            temp_weights_file = load_weights_from_mlflow(
                args.init_weights_from_mlflow,
                tracking_uri=args.mlflow_tracking_uri
            )
            pretrained_path_normalized = temp_weights_file
            mlflow.log_param("init_weights_source", "mlflow")
            mlflow.log_param("init_weights_mlflow_run_id", args.init_weights_from_mlflow)
        elif args.pretrained_path:
            # If path is relative, resolve it relative to project root
            if not os.path.isabs(args.pretrained_path):
                pretrained_path_normalized = os.path.join(project_root, args.pretrained_path)
            else:
                pretrained_path_normalized = args.pretrained_path
            
            # Check file existence
            if not os.path.exists(pretrained_path_normalized):
                raise FileNotFoundError(
                    f"Pretrained model file not found: {pretrained_path_normalized}\n"
                    f"Original path: {args.pretrained_path}\n"
                    f"Project root: {project_root}"
                )
            pretrained_path_normalized = os.path.abspath(pretrained_path_normalized)
            mlflow.log_param("pretrained_source", "local_file")
        
        # Update args with normalized path for model init
        args.pretrained_path = pretrained_path_normalized
        
        # Log parameters (now with normalized path)
        log_params_from_args(args)
        
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"Using device: {device}")
        mlflow.log_param("device", str(device))

        # Data Transforms
        if args.use_augmentation:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((360, 640)),
                transforms.RandomHorizontalFlip(p=args.hflip_prob),
                transforms.RandomRotation(degrees=args.rotation_degrees),
                transforms.ColorJitter(brightness=args.jitter_brightness, contrast=args.jitter_contrast)
            ])
            print(f"Using augmentation: HFlip={args.hflip_prob}, Rotation=±{args.rotation_degrees}°, ColorJitter=({args.jitter_brightness},{args.jitter_contrast})")
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((360, 640))
            ])
            print("No augmentation enabled")
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((360, 640))
        ])

        # Datasets
        train_dataset = Chicken_200_trainset(transform=train_transform)
        val_dataset = Chicken_200_testset(transform=val_transform)
        
        dataloaders = {
            "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
            "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        }

        # Model
        # Use already normalized path (if provided)
        use_local = args.pretrained_path is not None
        use_standard = args.pretrained and not use_local
        
        if use_local:
            print(f"Using local pretrained model from: {args.pretrained_path}")
        elif use_standard:
            print("Using standard pretrained weights (requires internet connection)")
            mlflow.log_param("pretrained_source", "torchvision")
        else:
            print("Training from scratch (no pretrained weights)")
            mlflow.log_param("pretrained_source", "none")
        
        model = myresnet(pretrained=use_standard, dropout=args.dropout, pretrained_path=args.pretrained_path)
        
        # Clean up temporary file if created
        if temp_weights_file and os.path.exists(temp_weights_file):
            try:
                os.remove(temp_weights_file)
                print(f"Cleaned up temporary weights file: {temp_weights_file}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file {temp_weights_file}: {e}")
        
        if args.freeze_backbone:
            for param in model.model_ft.parameters():
                param.requires_grad = False
            # Unfreeze custom heads
            for param in model.model_ft.fc.parameters(): param.requires_grad = True
            for param in model.fc2.parameters(): param.requires_grad = True
            for param in model.fc3.parameters(): param.requires_grad = True
            
        model = model.to(device)
        
        # Log regularization settings
        print(f"Dropout: {args.dropout}, Weight Decay: {args.weight_decay}")
        mlflow.log_param("dropout", args.dropout)
        mlflow.log_param("weight_decay", args.weight_decay)
        mlflow.log_param("patience", args.patience)

        # Optimization
        loss_fn = nn.L1Loss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        scheduler = None 

        # Train
        model = train_model(model, dataloaders, loss_fn, optimizer, scheduler, device, 
                          args.epochs, patience=args.patience, min_delta=args.min_delta)
        
        # Log final model
        mlflow.pytorch.log_model(model, "model")
        
        # Log run ID for easy model retrieval
        run_id = mlflow.active_run().info.run_id
        print(f"\nTraining completed!")
        print(f"Run ID: {run_id}")
        print(f"Model URI: runs:/{run_id}/model")
        print(f"\nTo load this model later, use:")
        print(f"  python training_workspace/features/training/load_model_from_mlflow.py --run_id {run_id}")
        
        # Optionally register model in Model Registry
        # Uncomment the following lines to register the model:
        # model_name = "ResNet50_WeightEstimation"
        # mlflow.register_model(f"runs:/{run_id}/model", model_name)
        # print(f"Model registered as: {model_name}")

if __name__ == "__main__":
    main()

