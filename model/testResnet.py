## 用resnet 等训练分类
import sys
import os
import argparse
import glob

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import collections
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from torch import nn
from torchvision.transforms import transforms
import torchvision.models as models
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from dataset.chicken200.chicken200 import Chicken_200_trainset, Chicken_200_testset

# Constants for paths to weights
DEFAULT_RESNET_WEIGHTS_DIR = "exps/myresnet"  # Base directory with ResNet weights


def find_latest_resnet_weights(base_dir=DEFAULT_RESNET_WEIGHTS_DIR, prefer_final=True):
    """
    Finds the latest ResNet weights file.

    Args:
        base_dir: Base directory containing weights (default: exps/myresnet)
        prefer_final: If True, prefers fianlEpochWeights.pth, else picks the latest epoch-*-Weights.pth

    Returns:
        str: Path to the weights file, or None if not found
    """
    if not os.path.exists(base_dir):
        return None
    
    # Find all subdirectories with dates
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return None
    
    # Sort by date (last first)
    subdirs.sort(reverse=True)
    latest_dir = os.path.join(base_dir, subdirs[0])
    
    # If prefer final weights
    if prefer_final:
        final_weights = os.path.join(latest_dir, 'fianlEpochWeights.pth')
        if os.path.exists(final_weights):
            return final_weights
    
    # Otherwise, find the latest epoch-*-Weights.pth
    weight_files = glob.glob(os.path.join(latest_dir, 'epoch-*-Weights.pth'))
    if not weight_files:
        return None
    
    # Sort by modification time (last first)
    weight_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return weight_files[0]

def makeEnv():

    expName = 'test_myresnet'
    model_name = 'myresnet'
    nowTime = time.strftime("%Y-%m-%d %H-%M", time.localtime())
    expPath = os.path.join('exps',expName,nowTime)
    if not os.path.exists(expPath):
        os.makedirs(expPath)

    num_classes = 1 # Number of classes
    batch_size = 8
    num_epochs = 100
    lr = 0.001
    feature_extract = False # 【False】 Train the whole network finetune the whole model | 【True】 Extract features only update the reshaped layer params

    weightPath = os.path.join(expPath, 'fianlEpochWeights.pth')
    logFilePath = os.path.join(expPath, 'train.log' )

    with open(logFilePath, 'w') as f:
        lines = list()
        lines.append("model_name " + model_name + '\n')
        lines.append("num_classes " + str(num_classes) + '\n')
        lines.append("batch_size " + str(batch_size) + '\n')
        lines.append("num_epochs " + str(num_epochs) + '\n')
        lines.append("model_name " + model_name + '\n')
        lines.append("learningRate " + str(lr) + '\n')
        lines.append("feature_extract " + str(feature_extract) + '\n')
        lines.append("weightPath " + weightPath + '\n')
        f.writelines(lines)

    return expPath, model_name, num_epochs, num_classes, batch_size, lr, weightPath, feature_extract

class myresnet(nn.Module):
    def __init__(self):
        super( myresnet, self ).__init__()
        # self.model_ft=models.resnet18(pretrained=False)
        self.model_ft=models.resnet50(pretrained=False)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1048)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(1048,512)
        self.relu2=nn.ReLU()
        self.fc3 = nn.Linear(512,1)

    def forward(self, x):
        x=self.model_ft(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu2(x)
        x=self.fc3(x)
        x = torch.flatten(x) # Add when regression

        return x
class myresnet_base(nn.Module):
    def __init__(self):
        super( myresnet_base, self ).__init__()
        # self.model_ft=models.resnet18(pretrained=False)
        self.model_ft=models.resnet50(pretrained=False)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1024)  # Must match trainResnet.py structure
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(1024,512)  # Must match trainResnet.py structure
        self.relu2=nn.ReLU()
        self.fc3 = nn.Linear(512,1)

    def forward(self, x):
        x=self.model_ft(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu2(x)
        x=self.fc3(x)
        x = torch.flatten(x) # Add when regression

        return x
def test_model(model, dataloaders, expPath, device):
    model.eval()
    weight_gt = []
    weight_pr = []
    for phase in ["train", "val"]:
        for inputs, labels, path in dataloaders[phase]:
            # print(labels,type(labels))
            labels_gt = labels.numpy()
            # print(type(labels_gt))
            weight_gt.extend(labels_gt)
            inputs, labels= inputs.to(device), labels.to(device)

            # inputs is the picture, labels is the weight, path is the path
            # Process the path, path, read 25 manual parameters, throw them into training
            # print(path)
            path = path[0].split('/')[-1]
            # print(path)

            with torch.no_grad():
                outputs = model(inputs)

            # print(outputs,type(outputs))
            predict_weight = outputs.cpu().numpy()
            # print(predict_weight.shape)
            weight_pr.extend(predict_weight)

            # print(weight_gt,weight_pr)

        print(phase)
        print('Mean Absolute Error (MAE):', "{:.6f}".format(mean_absolute_error(weight_gt, weight_pr)))
        print('Mean Squared Error (MSE):', "{:.6f}".format(mean_squared_error(weight_gt, weight_pr)))
        print('Root Mean Squared Error (RMSE):', "{:.6f}".format(mean_squared_error(weight_gt, weight_pr) ** 0.5))
        print('R2 Score:', "{:.6f}".format(r2_score(weight_gt, weight_pr)))

    return

def before_test_resnet(weight_path=None):
    """
    Tests the ResNet model.
    
    Args:
        weight_path: Path to the weights file. If None, automatically finds the latest weights.
    """
    expPath, model_name, num_epochs, num_classes, batch_size, lr, _, feature_extract = makeEnv()
    
    # Define the path to the weights
    if weight_path is None:
        weight_path = find_latest_resnet_weights()
        if weight_path is None:
            raise FileNotFoundError(
                    f"Weights not found in {DEFAULT_RESNET_WEIGHTS_DIR}. "
                    f"Specify the path through --weights or ensure that the weights exist."
            )
    
    # Check if the file exists
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weights file not found: {weight_path}")
    
    print(f"Loading weights from: {weight_path}")

    train_dataset = Chicken_200_trainset(transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((360,640))
    ]))
    val_dataset = Chicken_200_testset(transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((360,640))
    ]))
    image_datasets = {"train":train_dataset, "val":val_dataset}
    dataloaders_dict = {x: DataLoader(image_datasets[x],batch_size=batch_size, shuffle=False, num_workers=4) for x in ["train", "val"]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet = myresnet_base()
    resnet.load_state_dict(torch.load(weight_path, map_location='cpu'))
    print(resnet)

    resnet = resnet.to(device)
    test_model(resnet, dataloaders_dict, expPath, device)

def make_pth(weight_path=None):
    """
    Converts ResNet weights for use in FusonNet (adds prefix "model_ft.").
    
    Args:
        weight_path: Path to the weights file. If None, automatically finds the latest weights.
    """
    if weight_path is None:
        weight_path = find_latest_resnet_weights(prefer_final=False)
        if weight_path is None:
            raise FileNotFoundError(
                f"Weights not found in {DEFAULT_RESNET_WEIGHTS_DIR}. "
                f"Specify the path through --weights or ensure that the weights exist."
            )
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weights file not found: {weight_path}")
    
    print(f"Processing weights from: {weight_path}")
    
    pth = torch.load(weight_path, map_location='cpu')
    print(pth.keys(), type(pth))
    exceptkey = ['fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']

    mydict = collections.OrderedDict()
    for k in pth.keys():
        print(k, pth[k].size())
        if k not in exceptkey:
            mydict['model_ft.'+k] = pth[k]
        else:
            mydict[k] = pth[k]

    print(mydict.keys(), type(pth))
    # torch.save(mydict, weight_path)  # Uncomment to save

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ResNet model for weight prediction')
    parser.add_argument('--weights', type=str, default=None,
                        help=f'Path to model weights file. If not specified, automatically finds latest weights from {DEFAULT_RESNET_WEIGHTS_DIR}')
    parser.add_argument('--make-pth', action='store_true',
                        help='Run make_pth() to convert weights for FusonNet instead of testing')
    args = parser.parse_args()
    
    if args.make_pth:
        make_pth(weight_path=args.weights)
    else:
        before_test_resnet(weight_path=args.weights)


