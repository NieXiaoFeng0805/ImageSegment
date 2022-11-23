"""
@FileName：dataset.py
@Description：数据预处理
@Author：Feng
@Time：2022/10/31 16:41
"""

import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image  # 图片处理库
import torchvision.transforms as transforms
from Size_change import keep_image_size_open


def Get_data(file_path=None):
    # 从文件中读取数据
    files_list = os.listdir(file_path)  # 得到所有图片的名字
    # print(files_list)
    # 将每张图对应的路径拼接并存储到新列表中（图片的完整路径）
    file_path_list = [os.path.join(file_path, img_name) for img_name in files_list]
    file_path_list.sort()  # 进行排序
    # print(file_path_list)
    return file_path_list


transform = transforms.Compose([
    transforms.ToTensor()
])


class TheDataset(Dataset):
    def __init__(self, file_path=None):
        '''
        初始化构造函数
        :param file_path: 数据路径
        :param transform: 需要更改数据格式
        :param target_transform:更改目标格式
        '''
        if file_path is None:
            raise ValueError("路径为空！请确定图片路径正确")
        if len(file_path) != 2:
            raise ValueError("路径错误(图片路径在前，标签路径在后)")
        # 读取数据路径
        self.imagePath = file_path[0]
        self.labelPath = file_path[1]
        # 将读取到的图片、标签路径存储到相应变量中
        self.imgs = Get_data(self.imagePath)  # 存放图片路径
        self.labels = Get_data(self.labelPath)  # 存放标签路径

    def __getitem__(self, index):
        # 读取存储到的图片、标签路径并打开
        # 按顺序读取图片和标签
        img = self.imgs[index]
        label = self.labels[index]
        img_i = keep_image_size_open(img)
        label_i = keep_image_size_open(label)
        return transform(img_i), transform(label_i)

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    T = TheDataset(['data/Unet_crop/training/images', 'data/Unet_crop/training/labels'])
    print(T[0][0].shape)
    print(T[0][1].shape)
