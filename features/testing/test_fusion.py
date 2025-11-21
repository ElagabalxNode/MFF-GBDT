# Test script for FusionNet evaluation (regression task)
import sys
import os

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import copy
import time
import re
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
import torchvision.models as models
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from features.datasets.chicken200 import Chicken_200_trainset, Chicken_200_testset
from features.models.FusonNet import fusonnet50


def logger(log_str):
    # with open(expPath+'/log.txt','a',encoding='utf-8') as file:
    with open('data/outputs/exps/train_fusonnet/2025-11-21 14-35/log.txt','a',encoding='utf-8') as file:
        file.write(log_str)

def makeEnv():
    global expPath

    expName = 'test_fusonnet'
    model_name = 'fusonnet'
    nowTime = time.strftime("%Y-%m-%d %H-%M", time.localtime())
    expPath = os.path.join('data/outputs/exps',expName,nowTime)
    if not os.path.exists(expPath):
        os.makedirs(expPath)

    num_classes = 1 # Number of classes
    batch_size = 8 # batch_size can only be set to 1, one picture at a time, one manual feature at a time, one time

    log_str = "model_name " + model_name + '\n' + \
              "num_classes " + str(num_classes) + '\n' \
              "batch_size " + str(batch_size) + '\n'
    logger(log_str)

    return expPath, model_name, num_classes, batch_size

def get_manual_features():
    # csv_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_features.csv'
    csv_path = 'data/processed/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_normal_features.csv'
    df = pd.read_csv(csv_path,index_col='imgName')
    logger('csv_path: '+csv_path+'\n')
    # print(df.head())
    # df = df.drop(['weight'],axis=1) # Get training set x, 1 means drop by column  normal features are already dropped
    # print(type(df.loc['1.1_Depth-0.png']))
    return df

def save_auto_features(model, dataloaders, expPath, device):
    df = get_manual_features()
    model.eval()

    for phase in ["train", "val"]:
        DATENOW = time.strftime('%Y%m%d', time.localtime())
        dir_name = f"{DATENOW}-withauto"
        os.makedirs(os.path.join("data/processed/csvData", dir_name), exist_ok=True)
        file_path = os.path.join(
            "data/processed/csvData", dir_name,
            f"{DATENOW}-withauto-withnormal-{phase}.csv"
        )
        with open(file_path, 'w') as file:
            head = 'weight,imgName,area,perimeter,min_rect_width,min_rect_high,approx_area,approx_perimeter,extent,hull_perimeter,hull_area,' \
                'solidity,max_defect_dist,sum_defect_dist,equi_diameter,ellipse_long,ellipse_short,eccentricity,volume,maxHeight,minHeight,' \
                'max2min,meanHeight,mean2min,mean2max,stdHeight,heightSum'
            for i in range(2048):
                head += ',' + str(i)
            head += '\n'
            file.write(head)

            for inputs, labels, path in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Обрабатываем каждый элемент батча
                batch_size = inputs.size(0)
                manual_features_list = []
                
                # Собираем manual features для всего батча
                for b in range(batch_size):
                    x_path = path[b]
                    base = os.path.basename(x_path)  # '186.2_Depth-0-0.png'
                    # убрать суффикс "-число" перед .png:
                    key = re.sub(r'-\d+\.png$', '.png', base)  # '186.2_Depth-0.png'
                    
                    # Поиск в CSV по нормализованному имени
                    if key in df.index:
                        features = df.loc[key].values
                    else:
                        # Fallback: если не найдено, используем нулевые признаки
                        print(f"Warning: {key} not found in CSV, using zero features")
                        features = np.zeros(25)
                    
                    manual_features_list.append(features)
                
                # Преобразуем в тензор для модели
                manual_features = torch.as_tensor(manual_features_list, dtype=torch.float32).to(device)
                
                # Получаем auto-features для всего батча
                with torch.no_grad():
                    outputs, auto_features = model(inputs, manual_features)
                
                # Сохраняем каждый элемент батча в CSV
                for b in range(batch_size):
                    labels_str = str(labels[b].item())
                    x_path = path[b]
                    base = os.path.basename(x_path)
                    key = re.sub(r'-\d+\.png$', '.png', base)
                    
                    # Manual features
                    features = manual_features_list[b]
                    features_str = ','.join([str(i) for i in features])
                    features_str = labels_str + ',' + key + ',' + features_str
                    
                    # Auto features (2048 dim)
                    auto_feat = auto_features[b].cpu().numpy()
                    auto_features_str = ','.join([str(i) for i in auto_feat])
                    
                    final_str = features_str + ',' + auto_features_str + '\n'
                    file.write(final_str)

                # exit()
    return

def test_model(model, dataloaders, expPath, device):
    df = get_manual_features()
    model.eval()
    for phase in ["train", "val"]:
        weight_gt = []
        weight_pr = []
        for inputs, labels, path in dataloaders[phase]:
            # print(labels,type(labels))
            labels_gt = labels.numpy()
            # print(type(labels_gt))
            weight_gt.extend(labels_gt)
            inputs, labels= inputs.to(device), labels.to(device)

            # inputs is the picture, labels is the weight, path is the path
            # Process the path, path, read 25 manual parameters, throw them into training
            manual_features = []
            # print(path)
            for p in range(len(path)):
                x_path = path[p]
                base = os.path.basename(x_path)  # '186.2_Depth-0-0.png'
                # убрать суффикс "-число" перед .png:
                key = re.sub(r'-\d+\.png$', '.png', base)  # '186.2_Depth-0.png'
                # print(key)
                
                # Поиск в CSV по нормализованному имени
                if key in df.index:
                    features = df.loc[key].values
                else:
                    # Fallback: если не найдено, используем нулевые признаки
                    print(f"Warning: {key} not found in CSV, using zero features")
                    features = np.zeros(25)
                
                manual_features.append(features)


            manual_features = torch.as_tensor(manual_features, dtype=torch.float32)
            # print(manual_features)
            manual_features = manual_features.cuda()
            # print(type(manual_features),manual_features.size())
            with torch.no_grad():
                outputs,auto_features = model(inputs, manual_features)

            # print(outputs,type(outputs))
            predict_weight = outputs.cpu().numpy()
            # print(predict_weight.shape)
            weight_pr.extend(predict_weight)
            # print(weight_gt,weight_pr)
            # print(type(weight_gt),type(weight_pr))
            # exit()
        log_str = phase +'：\n' \
                'Mean absolute error (MAE): {:.6f}\n' \
                'Mean squared error (MSE): {:.6f}\n' \
                'Root mean squared error (RMSE): {:.6f}\n' \
                'R2: {:.6f}\n'.format(mean_absolute_error(weight_gt,weight_pr),mean_squared_error(weight_gt, weight_pr),
                                      mean_squared_error(weight_gt, weight_pr) ** 0.5, r2_score(weight_gt,weight_pr))
        print(log_str),logger(log_str)
        pd.DataFrame({'gt':weight_gt,'pr':weight_pr}).to_csv(expPath+'/'+phase+'_weight_pr.csv')

    return

def test_fusonnet():
    expPath, model_name, num_classes, batch_size = makeEnv()
    # init_resnet_weightPath = "data/outputs/exps/resnet_fizze_train_fc/2021-12-12 06-36/epoch-148-0.10712745562195777-Weights.pth" #107g
    init_resnet_weightPath = "data/outputs/exps/train_fusonnet/2025-11-21 14-35/fianlEpochWeights.pth"


    logger("init_resnet_weightPath: "+init_resnet_weightPath+'\n')
    train_dataset = Chicken_200_trainset(transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((360,640))
    ]))
    val_dataset = Chicken_200_testset(transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((360,640))
    ]))

    image_datasets = {"train":train_dataset, "val":val_dataset}
    dataloaders_dict = {x: DataLoader(image_datasets[x],
        batch_size=batch_size, shuffle=False, num_workers=4) for x in ["train", "val"]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = next(iter(dataloaders_dict['train']))[0]
    print(img.shape)

    fusonnet = fusonnet50()
    fusonnet.load_state_dict(torch.load(init_resnet_weightPath, map_location='cpu'))
    print(fusonnet)

    fusonnet = fusonnet.to(device)

    # Сначала тестируем модель
    test_model(fusonnet, dataloaders_dict, expPath, device)
    
    # Затем сохраняем auto-features для обучения LightGBM
    print("Saving auto-features for LightGBM training...")
    save_auto_features(fusonnet, dataloaders_dict, expPath, device)
    print("Auto-features saved successfully!")


if __name__ == '__main__':
    test_fusonnet()



