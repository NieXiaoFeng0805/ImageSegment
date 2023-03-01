# author: Feng
# contact: 1245272985@qq.com
# datetime:2023/2/27 19:17
# software: PyCharm
"""
文件说明：Comparison index result

"""
import os

from medpy import metric
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import numpy as np


def data_loader(img_root, gt_root):
    """TO get the pred and mask"""
    images = [img_root + f for f in os.listdir(img_root) if f.endswith('png') or f.endswith('jpg')]
    gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('png') or f.endswith('jpg')]
    # print(images[0], gts[0])
    images = sorted(images)
    gts = sorted(gts)
    return images, gts


def GetConfusionMatrix(imgPredict, imgLabel, num_class):
    """To get the pred and mask confusion matrix"""
    # remove classes from unlabeled pixels in gt image and predict
    mask = (imgLabel >= 0) & (imgLabel < num_class)
    label = num_class * imgLabel[mask] + imgPredict[mask]
    count = int(np.bincount(label, minlength=num_class ** 2))  # 核心代码
    confusionMatrix = count.reshape(num_class, num_class)
    return confusionMatrix


def calculate_metric_percase(pred, gt):
    """To calculate the index()"""
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, jc, hd, asd


def meanIntersectionOverUnion(confusionMatrix):
    """To get the MIoU"""
    # Intersection = TP Union = TP + FP + FN
    # IoU = TP / (TP + FP + FN)
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return IoU, mIoU


# pred_root = './results/PraNet/CVC-ClinicDB/14.png'
# gt_root = './data/TestDataset/CVC-ClinicDB/masks/14.png'
# pred = mpimg.imread(pred_root)
# gt = mpimg.imread(gt_root)
# print(len(gt.shape))
# gt = cv.cvtColor(gt, cv.COLOR_RGB2GRAY)
# dice, jc, hd, asd = calculate_metric_percase(pred, gt)
# print(dice)

for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    pred_root = './results/PraNet/{}/'.format(_data_name)
    gt_root = './data/TestDataset/{}/masks/'.format(_data_name)
    preds, gts = data_loader(pred_root, gt_root)
    try:
        len(preds) != len(gts)
    except TypeError:
        raise 'The number of prediction charts is not equal to the number of masks'
    mean_dice = 0
    total_dice = 0
    IoU, MIoU = 0.0, 0.0
    for i in range(len(preds)):
        # To get the img and gt
        pred = mpimg.imread(preds[i])
        gt = mpimg.imread(gts[i])

        if len(gt.shape) >= 3:
            gt = cv.cvtColor(gt, cv.COLOR_RGB2GRAY)
        # To calculate four indexes
        dice, jc, hd, asd = calculate_metric_percase(pred, gt)
        total_dice += dice
        # To calculate IoU and MIoU

        confusion_matrix = GetConfusionMatrix(pred, gt, 2)
        print(confusion_matrix)
    mean_dice = total_dice / len(preds)
    print("The Mean_dice of {} ===>{:.4f}".format(_data_name, mean_dice))
