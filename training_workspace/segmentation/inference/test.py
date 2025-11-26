# Run the test set using YOLOv8-seg to obtain the instance segmentation image of the target

import sys
import os

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import cv2
import numpy as np
import time
from PIL import Image
from ultralytics import YOLO
import random


def PredictImg(resDir, path, imgName, model, device):
    """
    Predict instances using YOLOv8-seg and save results in the same format as Mask R-CNN.
    
    Args:
        resDir: Output directory
        path: Path to input image
        imgName: Image filename
        model: YOLOv8 model
        device: Device string ('cuda', 'cpu', etc.)
    """
    global dst1

    # Load image
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not read image {path}")
        return
    
    result = img.copy()
    dst = img.copy()
    oImg = img.copy()
    
    # Get image dimensions
    img_height, img_width = img.shape[:2]

    names = {'0': 'background', '1': 'chicken'}

    # Run YOLOv8 inference
    results = model.predict(
        path,
        device=device if device != 'cuda' else 0,  # YOLO uses "0" for cuda:0
        conf=0.90,  # Confidence threshold (matching original 0.90)
        verbose=False
    )
    
    # Extract results from first (and only) image
    result_obj = results[0]
    
    # Get boxes, masks, scores
    boxes = result_obj.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores = result_obj.boxes.conf.cpu().numpy()
    masks = result_obj.masks  # YOLO masks object
    
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255],
               [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250],
               [50, 190, 190]]

    m_bOk = False
    num_instances = len(boxes)
    
    for idx in range(num_instances):
        if scores[idx] >= 0.90:  # Confidence threshold
            m_bOk = True

            color = colours[random.randrange(0, 10)]

            # Get mask for this instance
            if masks is not None and masks.data is not None:
                # YOLO masks are in shape [N, H, W] where N is number of instances
                mask_tensor = masks.data[idx].cpu().numpy()  # Shape: [H, W]
                
                # Resize mask to original image size if needed
                if mask_tensor.shape != (img_height, img_width):
                    mask_tensor = cv2.resize(mask_tensor, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                
                # Convert to uint8 binary mask (0 or 255)
                mask = (mask_tensor * 255).astype(np.uint8)
            else:
                # Fallback: create mask from bounding box
                x1, y1, x2, y2 = boxes[idx]
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                mask[int(y1):int(y2), int(x1):int(x2)] = 255

            thresh = mask
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            cv2.drawContours(dst, contours, -1, color, -1)

            x1, y1, x2, y2 = boxes[idx]
            name = names.get('1', 'chicken') + '-' + str(scores[idx])
            cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

            cv2.putText(result, text=name, org=(int(x1), int(y1) + 10), 
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=color)

            dst1 = cv2.addWeighted(result, 0.7, dst, 0.3, 0)

            cv2.imwrite(os.path.join(resDir, 'mask', imgName.split('.png')[0] + '-' + str(idx) + '.png'), mask)

            ret, mask = cv2.threshold(np.uint8(mask), 100, 255, 0)
            mask = np.dstack((mask, mask, mask))
            maskImg = cv2.bitwise_and(oImg, mask)

            cv2.imwrite(os.path.join(resDir, 'maskImg', imgName.split('.png')[0] + '-' + str(idx) + '.png'), maskImg)

    if m_bOk:
        cv2.imwrite(os.path.join(resDir, 'target', imgName), dst1)
    else:
        # Save empty result if no detections
        cv2.imwrite(os.path.join(resDir, 'target', imgName), result)


"""
More than PredictImg: 1、Threshold processing of the result mask 2、Save one more image (multi_mask)
"""
def PredictImg2(resDir, path, imgName, model, device):
    """
    Predict instances using YOLOv8-seg with multi-mask output.
    """
    global dst1

    # Load image
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not read image {path}")
        return
    
    result = img.copy()
    dst = img.copy()
    oImg = img.copy()
    
    # Get image dimensions
    img_height, img_width = img.shape[:2]

    names = {'0': 'background', '1': 'chicken'}

    # Run YOLOv8 inference
    results = model.predict(
        path,
        device=device if device != 'cuda' else 0,
        conf=0.85,  # Lower threshold for PredictImg2
        verbose=False
    )
    
    result_obj = results[0]
    boxes = result_obj.boxes.xyxy.cpu().numpy()
    scores = result_obj.boxes.conf.cpu().numpy()
    masks = result_obj.masks

    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255],
               [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250],
               [50, 190, 190]]

    multi_mask = np.zeros((img_height, img_width), np.uint8)
    m_bOk = False
    num_instances = len(boxes)
    
    for idx in range(num_instances):
        if scores[idx] >= 0.85:  # Threshold processing mask
            m_bOk = True

            color = colours[random.randrange(0, 10)]

            # Get mask
            if masks is not None and masks.data is not None:
                mask_tensor = masks.data[idx].cpu().numpy()
                if mask_tensor.shape != (img_height, img_width):
                    mask_tensor = cv2.resize(mask_tensor, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                mask = (mask_tensor * 255).astype(np.uint8)
            else:
                x1, y1, x2, y2 = boxes[idx]
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                mask[int(y1):int(y2), int(x1):int(x2)] = 255

            ret, mask = cv2.threshold(np.uint8(mask), 100, 255, 0)  # Threshold processing mask

            thresh = mask  # findContours will consume mask
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(dst, contours, -1, color, -1)  # Fill the contour
            cv2.drawContours(multi_mask, contours, -1, [255], -1)  # Fill the contour

            x1, y1, x2, y2 = boxes[idx]
            name = names.get('1', 'chicken')
            cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
            cv2.putText(result, text=name, org=(int(x1), int(y1) + 10), 
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=color)

            dst1 = cv2.addWeighted(result, 0.7, dst, 0.3, 0)  # The mask, box and text are merged

            cv2.imwrite(os.path.join(resDir, 'mask', imgName.split('.png')[0] + '-' + str(idx) + '.png'), mask)

            ret, mask = cv2.threshold(np.uint8(mask), 100, 255, 0)  # Threshold processing mask
            mask = np.dstack((mask, mask, mask))
            maskImg = cv2.bitwise_and(oImg, mask)  # Mask processing the original image

            cv2.imwrite(os.path.join(resDir, 'maskImg', imgName.split('.png')[0] + '-' + str(idx) + '.png'), maskImg)

    if m_bOk:
        cv2.imwrite(os.path.join(resDir, 'target', imgName), dst1)  # Save the fused image
        cv2.imwrite(os.path.join(resDir, imgName), multi_mask)  # Save the multi-mask image
    else:
        cv2.imwrite(os.path.join(resDir, 'target', imgName), result)
        cv2.imwrite(os.path.join(resDir, imgName), multi_mask)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load YOLOv8-Nano model
    weightPath = "data/models/segmentation/yolo/best_n.pt"
    if not os.path.exists(weightPath):
        # Fallback to pretrained if trained model not found
        print(f"Warning: {weightPath} not found. Using pretrained yolov8n-seg.pt")
        weightPath = "yolov8n-seg.pt"
    
    model = YOLO(weightPath)
    print(f"Loaded YOLOv8 model from {weightPath}")

    start_time = time.time()

    dir = 'data/raw/origin'

    weightName = os.path.basename(weightPath).split('.')[0]  # Extract name without extension
    dataName = os.path.basename(dir) if os.path.basename(dir) else 'origin'
    resPath = f'data/outputs/exps/data_{dataName}_weight_{weightName}-result/'

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
        path = os.path.join(dir, name)
        print(path)
        PredictImg(resPath, path, name, model, device)
        # PredictImg2(resPath, path, name, model, device)

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/len(imgList):.4f}s")
