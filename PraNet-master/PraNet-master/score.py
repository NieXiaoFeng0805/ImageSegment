# author: Feng
# contact: 1245272985@qq.com
# datetime:2023/2/28 13:48
# software: PyCharm
"""
文件说明：To calculate confusion matrix

"""
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
import matplotlib.image as mpimg
import cv2 as cv

#
# def genConfusionMatrix(imgPredict, imgLabel):
#     """To get the pred and mask confusion matrix"""
#     # remove classes from unlabeled pixels in gt image and predict
#     mask = (imgLabel >= 0) & (imgLabel < self.numClass)
#     label = self.numClass * imgLabel[mask] + imgPredict[mask]
#     count = np.bincount(label, minlength=self.numClass ** 2)  # 核心代码
#     confusionMatrix = count.reshape(self.numClass, self.numClass)
#     return confusionMatrix
"""
confusionMetric
P\L     P    N
P      TP    FP
N      FN    TN
"""

pred_root = './results/PraNet/CVC-300/149.png'
gt_root = './data/TestDataset/CVC-300/masks/149.png'
# pred = mpimg.imread(pred_root)
# gt = mpimg.imread(gt_root)
# gt = cv.cvtColor(gt, cv.COLOR_RGB2GRAY)
pred = np.array(Image.open(pred_root))
gt = np.array(Image.open(gt_root))
gt = cv.cvtColor(gt, cv.COLOR_RGB2GRAY)
print(pred.shape)
print(gt.shape)


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':
    imgPredict = np.array(pred)
    imgLabel = np.array(gt)
    metric = SegmentationMetric(2)
    metric.addBatch(imgPredict, imgLabel)
    acc = metric.pixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    print(acc, mIoU)
