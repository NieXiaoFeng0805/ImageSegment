"""
@FileName：CalculateIndicators.py
@Description：指标的计算
@Author：Feng
@Time：2022/10/23 18:06
"""
import six.moves

import numpy as np

import FCN.cfg as cfg


def calc_semantic_segmentation_confusion(pred_labels, true_labels):
    '''
    计算混淆矩阵
    :param pred_labels: 预测标签
    :param gt_labels:真实标签
    :return:返回混淆矩阵
    '''
    pred_labels = iter(pred_labels)
    true_labels = iter(true_labels)
    # 总类别
    num_class = cfg.DATA_KINDS
    confusion = np.zeros((num_class, num_class), dtype=np.int64)
    for pre_label, true_label in six.moves.zip(pred_labels, true_labels):
        if pre_label.ndim != 2 or true_label.ndim != 2:
            raise ValueError('ndim of labels should be two')
        if pre_label.shape != true_label.shape:
            raise ValueError('Shape of ground truth and prediction should be same')
        pre_label = pre_label.flatten()
        true_label = true_label.flatten()

        lb_max = np.max((pre_label, true_label))
        # print(lb_max)
        if lb_max >= num_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:num_class, 0:num_class] = confusion

            num_class = lb_max + 1
            confusion = expanded_confusion

        mask = true_label >= 0
        # 将类别和每个像素点对应并计数
        confusion += np.bincount(
            num_class * true_label[mask].astype(int) + pre_label[mask],
            minlength=num_class ** 2).reshape((num_class, num_class))
    for iter_ in (pred_labels, true_labels):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
    return confusion


def calc_semantic_segmentation_iou(confusion):
    '''
    计算IoU
    :param confusion:
    :return:
    '''
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0)
                       - np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou[:-1]  # 背景类去除，若包含背景则直接返回iou即可


def eval_semantic_segmentation(pred_labels, true_labels):
    confusion = calc_semantic_segmentation_confusion(pred_labels, true_labels)
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_acc = np.diag(confusion).sum() / confusion.sum()
    classes_acc = np.diag(confusion) / (np.sum(confusion, axis=1) + 1e-10)
    return {
        'IoU': iou, 'MIoU': np.nanmean(iou),
        'pixel_acc': pixel_acc,
        'classes_acc': classes_acc,
        'mean_classes_acc': np.nanmean(classes_acc[:-1])
    }
