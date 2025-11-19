from PIL import Image
import os
import numpy as np
import torch
import torch.utils.data


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transfroms):
        self.root = root
        self.transfroms = transfroms
        # Get all sorted file names in the current working directory and store them in a list
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'origin'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'mask'))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'origin', self.imgs[idx])
        mask_path = os.path.join(self.root, 'mask', self.masks[idx])
        # Ensure the image is in RGB mode, while the mask does not need to be converted to RGB mode, because the mask background is 0, and each color represents an instance
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        # Convert the PIL image to a numpy array, get the instance encoding in the mask and remove the background
        mask = np.array(mask)
        obj_id = np.unique(mask)
        obj_id = obj_id[1:]

        # None is newaxis, which is equivalent to adding a new dimension
        # Split the color-encoded mask into a set of binary masks
        # The explanation for the following code: For example, in FudanPed000012, there are two objects.
        # In FudanPed000012_mask, pixel value 0 represents the background,
        # pixel value 1 represents object 1, and pixel value 2 represents object 2.
        # These are only used to represent objects and are not for coloring; thus, 
        # when viewing the mask image by eye, everything looks black.
        # mask is a 559*536 2D matrix, obj_id=[0, 1, 2]
        # "obj_ids = obj_ids[1:]" removes the background pixel 0, so obj_id=[1, 2]
        # The line below creates masks (2 x 559 x 536), which contains two masks of size (559 x 536),
        # each corresponding to the first and second object, respectively.
        # In the first mask, the pixels covered by object 1 are True, all others are False.
        # In the second mask, the pixels covered by object 2 are True, all others are False.
        
        masks = mask == obj_id[:, None, None]  # Even if the image is 8-bit single-channel in "L" mode, PIL still loads it as three channels

        # Calculate bounding box coordinates for each mask
        num_objs = len(obj_id)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # Handle case when no objects are found (empty image)
        # Ensure boxes is always 2D tensor [N, 4] even when N=0
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]),
                                dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # The dataset has only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transfroms is not None:

            img ,target = self.transfroms(img, target)
            # target["masks"] = self.transfroms(target["masks"])

        return img, target

    def __len__(self):
        return len(self.imgs)

# Verify output
# dataset = PennFudanDataset('PennFudanPed/')
# print(dataset[0])
