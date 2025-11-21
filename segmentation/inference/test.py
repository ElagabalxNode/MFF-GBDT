# Run the test set of maskrcnn to obtain the instance segmentation image of the target

import sys
import os

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from utils import general as utils
import cv2
import numpy as np
import torchvision
import time
from PIL import Image
from segmentation.models.Mask_rcnn_Model import get_model_instance_segmentation
# from utils.transforms import  ToTensor,RandomHorizontalFlip,Compose
from torch.utils.data import DataLoader
from utils.engine import train_one_epoch, evaluate
from torchvision.transforms import transforms
import random


def toTensor(img):
    assert type(img) == np.ndarray, 'the img type is {}， but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2,0,1)))
    return img.float().div(255)


def PredictImg(resDir, path, imgName, model, device):

    global dst1

    # result = img.copy()
    # dst = img.copy()
    # img = transforms.ToTensor(img)
    # img = toTensor(img)

    img = Image.open(path).convert("RGB")
    result = cv2.imread(path)
    dst = result.copy()
    oImg = result.copy()
    img = transforms.Compose([transforms.ToTensor()])(img)

    names = {'0':'background', '1':'chicken'}

    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    masks = prediction[0]['masks']

    # print(masks)
    # print(prediction[0]['masks'].shape)
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],
             [255, 255, 0],[255, 0, 255],[80, 70, 180],
             [250, 80, 190],[245, 145, 50],[70, 150, 250],
             [50, 190, 190]]

    m_bOk = False
    for idx in range(boxes.shape[0]):
        # cv2.imshow('mask',masks[idx,0].mul(255).byte().cpu().numpy())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if scores[idx] >= 0.90:  # TODO Increase the threshold
            m_bOk = True

            color = colours[random.randrange(0, 10)]

            mask = masks[idx, 0].mul(255).byte().cpu().numpy()
            thresh = mask
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            cv2.drawContours(dst,contours, -1, color, -1)

            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            # print(str(labels[idx].item()))
            name = names.get(str(labels[idx].item()))+'-'+str(scores[idx].item())
            # print((int(x1.item()),int(y1.item())), (int(x2.item()),int(y2.item())))
            cv2.rectangle(result, (int(x1.item()),int(y1.item())), (int(x2.item()),int(y2.item())), (255,0,0), 3)

            cv2.putText(result,text=name,org=(int(x1.item()),int(y1.item())+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA,color=color)
            # print(type(masks))
            # print(type(masks[idx].cpu().numpy()))
            # resmask = masks[idx].cpu().numpy().reshape([720,1280])
            # print(resmask)

            dst1 = cv2.addWeighted(result, 0.7, dst, 0.3, 0)

            cv2.imwrite(os.path.join(resDir,'mask',imgName.split('.png')[0] + '-' + str(idx) + '.png'), mask)

            ret,mask= cv2.threshold(np.uint8(mask),100,255,0)
            mask = np.dstack((mask,mask,mask))
            # print(type(mask),mask.shape)
            maskImg = cv2.bitwise_and(oImg,mask)

            cv2.imwrite(os.path.join(resDir,'maskImg',imgName.split('.png')[0] + '-' + str(idx) + '.png'), maskImg)

            # print(masks[idx,0])

            # cv2.imshow('mask',mask)
            # # cv2.imshow('dst1',dst1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    # if m_bOk:
    #     cv2.imshow('result', dst1)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(resDir,'target',imgName), dst1)

    # return dst1
"""
More than PredictImg1: 1、Threshold processing of the result mask 2、Save one more image
"""
def PredictImg2(resDir, path, imgName, model, device):

    global dst1

    # result = img.copy()
    # dst = img.copy()
    # img = transforms.ToTensor(img)
    # img = toTensor(img)

    img = Image.open(path).convert("RGB")
    result = cv2.imread(path)
    dst = result.copy()
    oImg = result.copy()
    img = transforms.Compose([transforms.ToTensor()])(img)

    names = {'0':'background', '1':'chicken'}

    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    masks = prediction[0]['masks']

    # print(masks)
    # print(prediction[0]['masks'].shape)
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],
             [255, 255, 0],[255, 0, 255],[80, 70, 180],
             [250, 80, 190],[245, 145, 50],[70, 150, 250],
             [50, 190, 190]]

    multi_mask = np.zeros((720,1280),np.uint8)
    m_bOk = False
    for idx in range(boxes.shape[0]):
        # cv2.imshow('mask',masks[idx,0].mul(255).byte().cpu().numpy())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if scores[idx] >= 0.85:  # TODO Increase the threshold
            m_bOk = True

            color = colours[random.randrange(0, 10)]

            mask = masks[idx, 0].mul(255).byte().cpu().numpy()
            ret,mask= cv2.threshold(np.uint8(mask),100,255,0) # Threshold processing mask

            thresh = mask # findContours will consume mask
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # Find the contour of the mask
            cv2.drawContours(dst,contours, -1, color, -1) # Fill the contour
            cv2.drawContours(multi_mask,contours, -1, [255], -1) # Fill the contour

            # cv2.imshow('dst',dst)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names.get(str(labels[idx].item()))
            cv2.rectangle(result, (int(x1.item()),int(y1.item())), (int(x2.item()),int(y2.item())), (255,0,0), 3)
            cv2.putText(result,text=name,org=(int(x1.item()),int(y1.item())+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA,color=color)

            dst1 = cv2.addWeighted(result, 0.7, dst, 0.3, 0) # The mask, box and text are merged into the original image result

            cv2.imwrite(os.path.join(resDir,'mask',imgName.split('.png')[0] + '-' + str(idx) + '.png'), mask) # Save the source mask

            ret,mask= cv2.threshold(np.uint8(mask),100,255,0) # Threshold processing mask
            mask = np.dstack((mask,mask,mask))
            # print(type(mask),mask.shape)
            maskImg = cv2.bitwise_and(oImg,mask) # Mask processing the original image

            cv2.imwrite(os.path.join(resDir,'maskImg',imgName.split('.png')[0] + '-' + str(idx) + '.png'), maskImg)  # Save the original image

            # print(masks[idx,0])

            # cv2.imshow('mask',mask)
            # # cv2.imshow('dst1',dst1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    # if m_bOk:
    #     cv2.imshow('result', dst1)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(resDir,'target',imgName), dst1) # Save the fused image
    cv2.imwrite(os.path.join(resDir,imgName), multi_mask) # Save the fused image

    # return dst1
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2

    weightPath = "data/models/segmentation/weight/model-1121-100.pth"
    # weightPath = "data/models/segmentation/weight/82copymodel-10.pth"

    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(weightPath))

    start_time = time.time()

    # PredictImg( model, device)

    dir = 'data/raw/origin'

    weightName = weightPath.split("/")[-1]
    dataName = dir.split('/')[-2]
    resPath = 'data/outputs/exps/data_' + dataName + '_weight_' + weightName + '-result/'


    if not os.path.exists(resPath + 'mask'):
        os.makedirs(resPath + 'mask')
    if not os.path.exists(resPath + 'maskImg'):
        os.makedirs(resPath + 'maskImg')
    if not os.path.exists(resPath + 'target'):
        os.makedirs(resPath + 'target')


    imgList = []
    for x in os.listdir(dir):
        if x.endswith('png'):
            imgList.append(x)

    for name in imgList:
        print(name)
        path = os.path.join(dir,name)
        print(path)
        PredictImg(resPath,path,name, model, device)
        # PredictImg2(resPath,path,name, model, device)

    total_time = time.time() - start_time

    print(total_time/len(imgList))
