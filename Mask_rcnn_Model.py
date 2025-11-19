# 1. start from a pre-trained model, and just finetune the last layer.
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm
import urllib.request


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    # Перехватываем urllib.request.urlretrieve для показа прогресса загрузки
    original_urlretrieve = urllib.request.urlretrieve

    def urlretrieve_with_progress(url, filename, reporthook=None, data=None):
        """Wrapper for urlretrieve to show download progress bar"""
        if reporthook is None:
            pbar = None

            def hook(count, block_size, total_size):
                nonlocal pbar
                if pbar is None and total_size > 0:
                    pbar = tqdm(total=total_size, unit='B', unit_scale=True,
                                desc='Downloading model', ncols=80)
                if pbar is not None:
                    pbar.update(block_size)
                    if count * block_size >= total_size:
                        pbar.close()
            reporthook = hook
        return original_urlretrieve(url, filename, reporthook, data)

    # Временно заменяем urlretrieve для показа прогресса
    urllib.request.urlretrieve = urlretrieve_with_progress

    try:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True)
    finally:
        # Restore the original urlretrieve function to avoid side effects
        urllib.request.urlretrieve = original_urlretrieve

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    hidden_layer = 64
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
