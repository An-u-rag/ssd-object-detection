import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A

# generate default bounding boxes


def default_box_generator(layers, large_scale, small_scale):
    # input:
    # layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    # large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    # small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].

    # output:
    # boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.

    boxes = np.empty((10*10+5*5+3*3+1*1, 4, 8))

    # Default Bounding Box Generation
    cell_count = 0
    for i, layer in enumerate(layers):
        num_cells = layer * layer
        for cell in range(num_cells):
            x_center = round(((cell % layer)+0.5)/layer, 2)
            y_center = round(((cell // layer)+0.5)/layer, 2)
            boxes[cell_count, 0] = [x_center, y_center, small_scale[i], small_scale[i], x_center - (small_scale[i]/2),
                                    y_center -
                                    (small_scale[i]/2), x_center +
                                    (small_scale[i]/2),
                                    y_center+(small_scale[i]/2)]
            boxes[cell_count, 1] = [x_center, y_center, large_scale[i], large_scale[i], x_center - (large_scale[i] / 2),
                                    y_center -
                                    (large_scale[i] / 2), x_center +
                                    (large_scale[i] / 2),
                                    y_center + (large_scale[i] / 2)]
            lsize_1 = round(large_scale[i] * (np.sqrt(2)), 2)
            lsize_2 = round(large_scale[i] / (np.sqrt(2)), 2)
            boxes[cell_count, 2] = [x_center, y_center, lsize_1, lsize_2, x_center - (lsize_1 / 2),
                                    y_center -
                                    (lsize_2 / 2), x_center + (lsize_1 / 2),
                                    y_center + (lsize_2 / 2)]
            boxes[cell_count, 3] = [x_center, y_center, lsize_2, lsize_1, x_center - (lsize_2 / 2),
                                    y_center -
                                    (lsize_1 / 2), x_center + (lsize_2 / 2),
                                    y_center + (lsize_1 / 2)]
            cell_count += 1

    # Clipping bounding boxes more than the image boundaries
    boxes = np.where(boxes < 0, 0, boxes)
    boxes = np.where(boxes > 1, 1, boxes)

    # reshape to [135 * 4, 8]
    boxes = np.reshape(boxes, (540, 8))

    return boxes


# this is an example implementation of IOU.
# It is different from the one used in YOLO, please pay attention.
# you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min, y_min, x_max, y_max):
    # input:
    # boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    # x_min,y_min,x_max,y_max -- another box (box_r)

    # output:
    # ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]

    inter = np.maximum(np.minimum(boxs_default[:, 6], x_max)-np.maximum(boxs_default[:, 4], x_min), 0)*np.maximum(
        np.minimum(boxs_default[:, 7], y_max)-np.maximum(boxs_default[:, 5], y_min), 0)
    area_a = (boxs_default[:, 6]-boxs_default[:, 4]) * \
        (boxs_default[:, 7]-boxs_default[:, 5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union, 1e-8)


def match(ann_box, ann_confidence, boxs_default, threshold, cat_id, x_min, y_min, x_max, y_max):
    # input:
    # ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    # ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    # boxs_default            -- [num_of_boxes,8], default bounding boxes
    # threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    # cat_id                  -- class id, 0-cat, 1-dog, 2-person
    # x_min,y_min,x_max,y_max -- bounding box

    # compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min, y_min, x_max, y_max)

    ious_true = ious > threshold

    object_bbox_indices = [i for i, v in enumerate(ious_true) if v]
    if len(object_bbox_indices) == 0:
        object_bbox_indices.append(np.argmax(ious))

    for index in object_bbox_indices:
        gw = x_max - x_min
        gh = y_max - y_min
        gx = x_min + (gw/2)
        gy = y_min + (gh/2)
        px = boxs_default[index][0]
        py = boxs_default[index][1]
        pw = boxs_default[index][2]
        ph = boxs_default[index][3]
        tx = (gx - px)/pw
        ty = (gy - py)/ph
        tw = np.log(gw/pw)
        th = np.log(gh/ph)
        ann_box[index] = [tx, ty, tw, th]
        ann_confidence[index] = np.zeros(4)
        ann_confidence[index][cat_id] = 1


class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train=True, test=False, image_size=320):
        self.train = train
        self.test = test
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num

        # overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)

        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size

        # notice:
        # you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train
        if train:
            self.img_names = self.img_names[:int(len(self.img_names)*0.9)]
        elif test:
            self.img_names = self.img_names
        else:
            self.img_names = self.img_names[int(len(self.img_names)*0.9):]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num, 4], np.float32)  # bounding boxes
        ann_confidence = np.zeros(
            [self.box_num, self.class_num], np.float32)  # one-hot vectors
        # one-hot vectors with four classes
        # [1,0,0,0] -> cat
        # [0,1,0,0] -> dog
        # [0,0,1,0] -> person
        # [0,0,0,1] -> background

        # the default class for all cells is set to "background"
        ann_confidence[:, -1] = 1

        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"

        # TODO:
        # 1. prepare the image [3,320,320], by reading image "img_name" first.
        image = cv2.imread(img_name)
        h, w = image.shape[0], image.shape[1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train:

            with open(ann_name) as f:
                lines = f.readlines()
                lines = [line.split() for line in lines]
                lines = np.array(lines, dtype=np.float32)
                bbox = [[line[1], line[2], line[3], line[4], line[0]]
                        for line in lines]

            # Random Crop
            # Albumentations
            p = np.random.randint(0, 2)
            if p:
                h_crop = np.random.randint(int(h * 0.4), h)
                w_crop = np.random.randint(int(w * 0.4), w)
                albumentation_transform = A.Compose([
                    A.RandomCrop(h_crop, w_crop)
                ], bbox_params=A.BboxParams(format='coco'))

                transformed = albumentation_transform(image=image, bboxes=bbox)
                image = transformed['image']
                bboxs = transformed['bboxes']
                lines = np.array([[bbox[-1], bbox[0], bbox[1], bbox[2], bbox[3]]
                                 for bbox in bboxs], dtype=np.float32)

            # to use function "match":
            for line in lines:
                x_min = line[1] / w
                y_min = line[2] / h
                x_max = (line[1] + line[3]) / w
                y_max = (line[2] + line[4]) / h

                class_id = int(line[0])

                match(ann_box, ann_confidence, self.boxs_default,
                      0.5, class_id, x_min, y_min, x_max, y_max)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
        ])

        image_t = transform(image)
        image_t = image_t.to("cuda")

        if len(image_t.shape) != 3:
            image_t = image_t.unsqueeze(0)
            image_t = image_t.expand((3, image_t.shape[1:]))

        return image_t, ann_box, ann_confidence
