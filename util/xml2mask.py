import os
import cv2
import argparse
import shutil
import numpy as np
from lxml import etree
from tqdm import tqdm

# Useful function to create a new directory and recursively delete the existing directory content:
def dir_create(path):
    # if (os.path.exists(path)) and (os.listdir(path) != []):
    #     shutil.rmtree(path)
    #     os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)

# The parameters of this script are the following data: input image directory, input file with CVAT annotations in XML format, output mask directory and image scale factor. Function to parse arguments from the command line:
def parse_args():
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Convert CVAT XML annotations to contours'
    )
    parser.add_argument(
        '--image-dir', metavar='DIRECTORY', required=True,
        help='directory with input images'
    )
    parser.add_argument(
        '--cvat-xml', metavar='FILE', required=True,
        help='input file with CVAT annotation in xml format'
    )
    parser.add_argument(
        '--output-dir', metavar='DIRECTORY', required=True,
        help='directory for output masks'
    )
    parser.add_argument(
        '--scale-factor', type=float, default=1.0,
        help='choose scale factor for images'
    )
    return parser.parse_args()



def parse_anno_file(cvat_xml, image_name):
    root = etree.parse(cvat_xml).getroot()
    anno = []
    image_name_attr = ".//image[@name='{}']".format(image_name)
    for image_tag in root.iterfind(image_name_attr):
        image = {}

        for key, value in image_tag.items():
            image[key] = value
        image['shapes'] = []

        for poly_tag in image_tag.iter('polygon'):
            polygon = {'type': 'polygon'}
            for key, value in poly_tag.items():
                polygon[key] = value
            image['shapes'].append(polygon)
        # for box_tag in image_tag.iter('box'):
        #     box = {'type': 'box'}
        #     for key, value in box_tag.items():
        #         box[key] = value
        #     box['points'] = "{0},{1};{2},{1};{2},{3};{0},{3}".format(
        #         box['xtl'], box['ytl'], box['xbr'], box['ybr'])
        #     image['shapes'].append(box)
        # image['shapes'].sort(key=lambda x: int(x.get('z_order', 0)))

        anno.append(image)
    return anno



def create_mask_file(width, height, bitness, background, shapes, scale_factor):
    mask = np.full((height, width), background, dtype=np.uint8)

    for i in range(len(shapes)):
        points = [tuple(map(float, p.split(','))) for p in shapes[i]['points'].split(';')]
        points = np.array([(int(p[0]), int(p[1])) for p in points])
        points = points*scale_factor
        points = points.astype(int)

        mask = cv2.fillPoly(mask, [points], color=i+1)

        # mask = cv2.drawContours(mask, [points], -1, color=(255, 255, 255), thickness=5)
        # mask = cv2.fillPoly(mask, [points], color=(0, 0, 255))
        # mask = cv2.fillPoly(mask, [points], color=(255, 255, 255))

    return mask

# Finally, the main function is:
def main():
    args = parse_args()
    dir_create(args.output_dir)
    img_list = [f for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))]
    mask_bitness = 24
    for img in tqdm(img_list, desc='Writing contours:'):
        img_path = os.path.join(args.image_dir, img)
        anno = parse_anno_file(args.cvat_xml, img)
        background = []
        is_first_image = True
        for image in anno:
            if is_first_image:
                current_image = cv2.imread(img_path)
                height, width, _ = current_image.shape
                background = np.zeros((height, width), np.uint8)
                is_first_image = False
            # output_path = os.path.join(args.output_dir, img.split('png')[0][:-1] + '.png')
            output_path = os.path.join(args.output_dir, img.split('JPG')[0][:-1] + '.png')
            # output_path = os.path.join(args.output_dir, args.image_dir.split("/")[len(args.image_dir.split("/"))-1] + '_' + img.split('.')[0] + '.png')
            background = create_mask_file(width,
                                          height,
                                          mask_bitness,
                                          background,
                                          image['shapes'],
                                          args.scale_factor)
            cv2.imwrite(output_path, background)


# When we execute the file as a command to the python interpreter, we must add the following construction:
if __name__ == "__main__":
    main()



# To run the script, you should run the following command (the default scale factor is 1 after marking without adjusting the image size):

# python xml2mask.py --image-dir original_images_dir --cvat-xml cvat.xml --output-dir masks_dir --scale-factor 0.4

# --image-dir ./data/105/origin --cvat-xml ./xml/105.xml --output-dir ./data/105/mask --scale-factor 1
# --image-dir ./data/82/origin --cvat-xml ./xml/82.xml --output-dir ./data/82/mask --scale-factor 1
# --image-dir ./data/100label/origin --cvat-xml ./xml/task_200只100张label-2021_07_14_14_15_10-cvat for images.xml --output-dir ./data/100label/mask --scale-factor 1
# --image-dir ./data/105origin --cvat-xml ./xml/105.xml --output-dir ./data/105mask --scale-factor 1
# --image-dir ./data/105origin --cvat-xml ./xml/105.xml --output-dir ./data/105mask --scale-factor 2
