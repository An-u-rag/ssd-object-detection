import os
import random
import numpy as np

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


def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    # input:
    # pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    # output:
    # loss -- a single number for the value of the loss function, [1]

    # Reshape confidences and bboxes
    ann_confidence = ann_confidence.reshape(-1, 4)
    ann_box = ann_box.reshape(-1, 4)
    pred_confidence = pred_confidence.reshape(-1, 4)
    pred_box = pred_box.reshape(-1, 4)
    # object_indices = [i for i, v in enumerate(ann_box) if v.any()]
    # no_object_indices = [i for i, v in enumerate(ann_box) if not v.any()]

    object_indices = torch.where(ann_confidence[:, -1] == 0)
    no_object_indices = torch.where(ann_confidence[:, -1] == 1)

    # l1_loss = F.smooth_l1_loss(
    #     pred_box[object_indices], ann_box[object_indices])
    l1_loss = F.smooth_l1_loss(
        pred_box[object_indices], ann_box[object_indices])

    c_obj_loss = F.cross_entropy(
        pred_confidence[object_indices], ann_confidence[object_indices])
    c_noobj_loss = (
        3 * F.cross_entropy(pred_confidence[no_object_indices], ann_confidence[no_object_indices]))

    # loss = (l1_loss + c_loss) / len(object_indices)
    loss = (l1_loss + c_obj_loss + c_noobj_loss)

    return loss


class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()

        # num_of_classes, in this assignment, 4: cat, dog, person, background
        self.class_num = class_num
        self.box_num = 540

        # TODO: define layers
        self.CNNBlock1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.CNNBlock2 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.CNNBlock3 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.CNNBlock4 = nn.Conv2d(256, 16, 1, 1, 0)

        self.CNNBlock5 = nn.Conv2d(256, 16, 3, 1, 1)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # input:
        # x -- images, [batch_size, 3, 320, 320]

        # x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.

        # TODO: define forward

        # should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?

        # sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        # confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        # bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]

        batch_size = x.shape[0]
        x = self.CNNBlock1(x)
        x1 = self.CNNBlock2(x)
        x2 = self.CNNBlock3(x1)
        x3 = self.CNNBlock3(x2)
        x4 = self.CNNBlock4(x3)
        x4 = torch.reshape(x4, (batch_size, 16, 1))
        x5 = self.CNNBlock4(x3)
        x5 = torch.reshape(x5, (batch_size, 16, 1))

        x11 = self.CNNBlock5(x)
        x11 = torch.reshape(x11, (batch_size, 16, 100))
        x12 = self.CNNBlock5(x)
        x12 = torch.reshape(x12, (batch_size, 16, 100))

        x21 = self.CNNBlock5(x1)
        x21 = torch.reshape(x21, (batch_size, 16, 25))
        x22 = self.CNNBlock5(x1)
        x22 = torch.reshape(x22, (batch_size, 16, 25))

        x31 = self.CNNBlock5(x2)
        x31 = torch.reshape(x31, (batch_size, 16, 9))
        x32 = self.CNNBlock5(x2)
        x32 = torch.reshape(x32, (batch_size, 16, 9))

        bboxes = torch.cat((x11, x21, x31, x4), -1).to("cuda:0")
        confidence = torch.cat((x12, x22, x32, x5), -1).to("cuda:0")

        bboxes = torch.permute(bboxes, (0, 2, 1))
        confidence = torch.permute(confidence, (0, 2, 1))

        bboxes = torch.reshape(bboxes, (batch_size, self.box_num, 4))
        confidence = torch.reshape(
            confidence, (batch_size, self.box_num, self.class_num))
        # confidence = self.softmax(confidence)

        return confidence, bboxes
