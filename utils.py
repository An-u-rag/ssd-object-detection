import copy

import numpy as np
import cv2
from dataset import iou
import matplotlib.pyplot as plt

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
# use [blue green red] to represent different classes


def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default, test=False):
    # input:
    # windowname      -- the name of the window to display the images
    # pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    # image_          -- the input image to the network
    # boxs_default    -- default bounding boxes, [num_of_boxes, 8]

    # print("shape of predicted confidence: ", pred_confidence.shape)
    # class_num = 4
    class_num = 3
    # class_num = 3 now, because we do not need the last class (background)

    image = np.transpose(image_, (1, 2, 0))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image1 = np.zeros(image.shape)
    image2 = np.zeros(image.shape)
    image3 = np.zeros(image.shape)
    image4 = np.zeros(image.shape)
    image1[:] = image[:]
    image2[:] = image[:]
    image3[:] = image[:]
    image4[:] = image[:]

    scale_size = image.shape[0]

    # draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            # if the network/ground_truth has high confidence on cell[i] with class[j]
            if ann_confidence[i, j] > 0.5:
                # print(ann_confidence[i, j])
                # print(ann_box[i])
                # TODO:
                # image1: draw ground truth bounding boxes on image1
                # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)

                # Image1 - recover ground truth bounding boxes
                px = boxs_default[i, 0]
                py = boxs_default[i, 1]
                pw = boxs_default[i, 2]
                ph = boxs_default[i, 3]
                tx = ann_box[i, 0]
                ty = ann_box[i, 1]
                tw = ann_box[i, 2]
                th = ann_box[i, 3]

                _gx = (pw * tx) + px
                _gy = (ph * ty) + py
                _gw = pw * np.exp(tw)
                _gh = ph * np.exp(th)
                x1 = _gx - (_gw/2)
                y1 = _gy - (_gh/2)
                x2 = _gx + (_gw/2)
                y2 = _gy + (_gh/2)
                # top left corner, x1<x2, y1<y2
                start_point = (int(x1 * scale_size), int(y1 * scale_size))
                # bottom right corner
                end_point = (int(x2 * scale_size), int(y2 * scale_size))
                # use red green blue to represent different classes
                color = colors[j]
                thickness = 2
                cv2.rectangle(image1, start_point, end_point, color, thickness)

                # Image2
                x1 = boxs_default[i, 4]
                y1 = boxs_default[i, 5]
                x2 = boxs_default[i, 6]
                y2 = boxs_default[i, 7]
                # top left corner, x1<x2, y1<y2
                start_point = (int(x1 * scale_size), int(y1 * scale_size))
                # bottom right corner
                end_point = (int(x2 * scale_size), int(y2 * scale_size))
                # use red green blue to represent different classes
                color = colors[j]
                thickness = 2
                cv2.rectangle(image2, start_point, end_point, color, thickness)

    if test == True:
        f = open(f'./outputs/test/Final3/{windowname}.txt', "w+")
    # predicted
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i, j] > 0.5:
                # TODO:
                # image3: draw network-predicted bounding boxes on image3
                # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)

                # Image3 - recover true bounding box from predicted bounding box
                px = boxs_default[i, 0] * scale_size
                py = boxs_default[i, 1] * scale_size
                pw = boxs_default[i, 2] * scale_size
                ph = boxs_default[i, 3] * scale_size
                dx = pred_box[i, 0]
                dy = pred_box[i, 1]
                dw = pred_box[i, 2]
                dh = pred_box[i, 3]

                _gx = (pw * dx) + px
                _gy = (ph * dy) + py
                _gw = pw * np.exp(dw)
                _gh = ph * np.exp(dh)
                x1 = _gx - (_gw/2)
                y1 = _gy - (_gh/2)
                x2 = _gx + (_gw/2)
                y2 = _gy + (_gh/2)
                # top left corner, x1<x2, y1<y2
                start_point = (int(x1), int(y1))
                # bottom right corner
                end_point = (int(x2), int(y2))
                # use red green blue to represent different classes
                color = colors[j]
                thickness = 2
                cv2.rectangle(image3, start_point, end_point, color, thickness)

                # Image4
                x1 = boxs_default[i, 4]
                y1 = boxs_default[i, 5]
                x2 = boxs_default[i, 6]
                y2 = boxs_default[i, 7]
                # top left corner, x1<x2, y1<y2
                start_point = (int(x1 * scale_size), int(y1 * scale_size))
                # bottom right corner
                end_point = (int(x2 * scale_size), int(y2 * scale_size))
                # use red green blue to represent different classes
                color = colors[j]
                thickness = 2
                cv2.rectangle(image4, start_point, end_point, color, thickness)

                if test == True:
                    f.write(f'{j} {_gx} {_gy} {_gw} {_gh}\n')

    if test == True:
        f.close()

    # combine four images into one
    h, w, _ = image1.shape
    image = np.zeros([h*2, w*2, 3])
    image[:h, :w] = image1
    image[:h, w:] = image2
    image[h:, :w] = image3
    image[h:, w:] = image4
    # cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]", image)
    if test == True:
        cv2.imwrite(
            f'./outputs/test/Final3_images/{windowname}.jpg', image * 255)
    else:
        cv2.imwrite(
            f'./outputs/train/train_new_nosoftmax/{windowname}.jpg', image * 255)
    # cv2.waitKey()
    # if you are using a server, you may not be able to display the image.
    # in that case, please save the image using cv2.imwrite and check the saved image for visualization.


def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.2/320, threshold=0.5):
    # print("reached NMS")
    # TODO: non maximum suppression
    # input:
    # confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # boxs_default -- default bounding boxes, [num_of_boxes, 8]
    # overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    # threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.

    # output:
    # depends on your implementation.
    # if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    # you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    A = box_.copy()
    C = confidence_[:, :-1].copy()
    thrown = []
    B = np.zeros([540, 4], dtype=np.float32)  # High probability boxes
    B_c = np.zeros([540, 4], dtype=np.float32)  # Corresponding confidences

    while len(thrown) < box_.shape[0]:

        max_prob = np.max(C)

        index = np.argmax(C) // 3

        if max_prob > threshold:
            x = A[index]
            B[index] = x
            B_c[index, :-1] = C[index]
            # A = np.delete(A, index, axis=0)
            # C = np.delete(C, index, axis=0)
            thrown.append(index)
            C[index] = np.array([0, 0, 0])

            # Higher box attributes
            px = boxs_default[index, 0]
            py = boxs_default[index, 1]
            pw = boxs_default[index, 2]
            ph = boxs_default[index, 3]
            tx1 = x[0]

            ty1 = x[1]
            tw1 = x[2]
            th1 = x[3]
            Gx = (pw * tx1) + px
            Gy = (ph * ty1) + py
            Gw = pw * np.exp(tw1)
            Gh = ph * np.exp(th1)
            Xmin = Gx - (Gw / 2) if Gx - (Gw / 2) > 0 else 0
            Ymin = Gy - (Gh / 2) if Gy - (Gh / 2) > 0 else 0
            Xmax = Gx + (Gw / 2) if Gx + (Gw / 2) < 1 else 1
            Ymax = Gy + (Gh / 2) if Gy + (Gh / 2) < 1 else 1

            for i, box in enumerate(A):
                if i in thrown:
                    continue
                else:
                    # Lower box attributes
                    px = boxs_default[i, 0]
                    py = boxs_default[i, 1]
                    pw = boxs_default[i, 2]
                    ph = boxs_default[i, 3]
                    tx2 = box[0]
                    ty2 = box[1]
                    tw2 = box[2]
                    th2 = box[3]
                    gx = (pw * tx2) + px
                    gy = (ph * ty2) + py
                    gw = pw * np.exp(tw2)
                    gh = ph * np.exp(th2)
                    xmin = gx - (gw / 2) if gx - (gw / 2) > 0 else 0
                    ymin = gy - (gh / 2) if gy - (gh / 2) > 0 else 0
                    xmax = gx + (gw / 2) if gx + (gw / 2) < 1 else 1
                    ymax = gy + (gh / 2) if gy + (gh / 2) < 1 else 1

                    # print("Xmin and xmin: ", Xmin, xmin)
                    # Find IOU between higher and lower box
                    higher_box = np.array(
                        [[Gx * 320, Gy * 320, Gw * 320, Gh * 320, Xmin * 320, Ymin * 320, Xmax * 320, Ymax * 320]])
                    iou_val = iou(higher_box, xmin * 320, ymin *
                                  320, xmax * 320, ymax * 320)[0]
                    # print(
                    #     f"iou between higher index {index} and lower index {i} is:  {iou_val}")
                    # input("Enter")

                    if iou_val >= overlap:
                        thrown.append(i)
                        C[i] = np.array([0, 0, 0])

        else:
            break

    return B_c, B


def generate_mAP(ann_conf, pred_conf):

    TP = 0
    FN = 0
    TN = 0
    FP = 0
    # Calculating positives in predictions
    for i, pred in enumerate(pred_conf):
        if np.argmax(pred_conf[i]) < 3 and np.argmax(pred_conf[i]) == np.argmax(ann_conf[i]):
            TP += 1
        elif np.argmax(pred_conf[i]) < 3 and np.argmax(pred_conf[i]) != np.argmax(ann_conf[i]):
            FP += 1
        elif np.argmax(pred_conf[i]) == 3 and np.argmax(ann_conf[i]) < 3:
            FN += 1
        else:
            TN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    F1_score = 2*precision*recall/np.maximum(precision+recall, 1e-8)
    print("F1 score:", F1_score)

    return F1_score, precision, recall
