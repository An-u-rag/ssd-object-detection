import argparse
import os
import numpy as np
import time
import cv2
import sys

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

from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
# please google how to use argparse
# a short intro:
# to train: python main.py
# to test:  python main.py --test


class_num = 4  # cat dog person background

num_epochs = 100
batch_size = 32


boxs_default = default_box_generator(
    [10, 5, 3, 1], [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7])


# Create network
device = "cuda:0" if torch.cuda.is_available() else "cpu"
network = SSD(class_num).to(device)
cudnn.benchmark = True

print(torch.cuda.get_device_name())

train_name = "train100_fixed_aug_nosoftmax"

if not args.test:
    dataset = COCO("data/train/images/", "data/train/annotations/",
                   class_num, boxs_default, train=True, image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/",
                        class_num, boxs_default, train=False, image_size=320)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    # feel free to try other optimizers and parameters.

    start_time = time.time()

    for epoch in range(num_epochs):
        # TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.to(device)
            ann_box = ann_box_.to(device)
            ann_confidence = ann_confidence_.to(device)

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)

            loss_net = SSD_loss(pred_confidence, pred_box,
                                ann_confidence, ann_box)

            loss_net.backward()
            optimizer.step()
            avg_loss += loss_net.data
            avg_count += 1

        print('[%d] time: %f train loss: %f' %
              (epoch, time.time()-start_time, avg_loss/avg_count))

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        # filtered_pred_confidence = np.array(filtered_pred_confidence)
        # filtered_pred_box = np.array(filtered_pred_box)

        # visualize
        visualize_pred(f"train_nonnms{epoch}_{train_name}", pred_confidence_, pred_box_, ann_confidence_[
                       0].numpy(), ann_box_[0].numpy(), images_[0].cpu().numpy(), boxs_default)

        # NMS
        filtered_pred_confidence, filtered_pred_box = non_maximum_suppression(
            pred_confidence_, pred_box_, boxs_default)

        # filtered_pred_confidence = np.array(filtered_pred_confidence)
        # filtered_pred_box = np.array(filtered_pred_box)

        # visualize
        visualize_pred(f"train{epoch}_{train_name}", filtered_pred_confidence, filtered_pred_box, ann_confidence_[
                       0].numpy(), ann_box_[0].numpy(), images_[0].cpu().numpy(), boxs_default)

        # VALIDATION
        network.eval()
        # fig, ax = plt.subplots()

        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.to(device)
            ann_box = ann_box_.to(device)
            ann_confidence = ann_confidence_.to(device)

            pred_confidence, pred_box = network(images)

            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()

            # optional: implement a function to accumulate precision and recall to compute mAP or F1.
            # update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
            # F1_score, precision, recall = generate_mAP(
            #     ann_confidence.detach().cpu().numpy(), pred_confidence_)
            # display plot
            # ax.plt(recall, precision)

            # ax.set_title("Precision-Recall-Curve")
            # ax.set_ylabel('Precision')
            # ax.set_xlabel('Recall')
            # plt.savefig("valprcurve{epoch}_{train_name}"+".png")

        # visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        # NMS
        filtered_pred_confidence, filtered_pred_box = non_maximum_suppression(
            pred_confidence_, pred_box_, boxs_default)

        visualize_pred(f"val{epoch}_{train_name}", filtered_pred_confidence, filtered_pred_box, ann_confidence_[
                       0].numpy(), ann_box_[0].numpy(), images_[0].cpu().numpy(), boxs_default)

        # optional: compute F1
        # F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        # print(F1score)

        # save weights
        if epoch % 10 == 9:
            # save last network
            print('saving net...')
            torch.save(network.state_dict(
            ), f'{train_name}.pth')

        # print(torch.cuda.memory_stats())


else:
    # TEST
    dataset_test = COCO("data/test/images/", "data/test/annotations/",
                        class_num, boxs_default, train=False, test=True, image_size=320)
    print("test dataset size: ", dataset_test.__len__(), " images")
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('train100_fixed_aug_nosoftmax.pth'))
    network.eval()

    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_ = data
        images = images_.to(device)
        ann_box = ann_box_.to(device)
        ann_confidence = ann_confidence_.to(device)

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        pred_confidence_, pred_box_ = non_maximum_suppression(
            pred_confidence_, pred_box_, boxs_default)

        visualize_pred(f"test{i}", pred_confidence_, pred_box_, ann_confidence_[
                       0].numpy(), ann_box_[0].numpy(), images_[0].cpu().numpy(), boxs_default, test=True)
