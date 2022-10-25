"""
@FileName：Dataset.py
@Description：数据预处理
@Author：Feng
@Time：2022/10/19 12:16
"""
import os

import FCN.cfg as cfg
import pandas as pd
import numpy as np
import torch
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image  # 图片处理库
import torchvision.transforms as transforms


class theDataset(Dataset):
    def __init__(self, file_path=None, crop_size=None):
        '''
        :param file_path: 列表类型，第一个元素为图片路径，第二个为标签路径
        :param crop_size: 图片尺寸呢大小
        '''
        if file_path is None:
            raise ValueError("路径为空！请确定图片路径正确")
        if len(file_path) != 2:
            raise ValueError("路径错误(图片路径在前，标签路径在后)")
        self.imagePath = file_path[0]
        self.labelPath = file_path[1]
        # 从路径中提取图片和标签数据
        self.imgs = self.readFile(self.imagePath)  # 存放图片路径
        self.labels = self.readFile(self.labelPath)  # 存放标签路径
        # 初始化处理数据需要尺寸
        self.crop_size = crop_size

    def __getitem__(self, index):
        '''
        对原图和标签进行处理
        :param index:
        :return:
        '''
        # 按顺序读取图片和标签
        img = self.imgs[index]
        label = self.labels[index]
        # 打开
        img = Image.open(img)
        label = Image.open(label).convert('RGB')  # 转成RGB避免不必要的错误
        # 进行中心裁剪
        img, label = self.center_crop(img, label, self.crop_size)
        img, label = self.img_transform(img, label)
        sample = {'img': img, 'label': label}  # 处理后的图像及标签大小
        return sample

    def __len__(self):  # 平常的len()函数不行，必须调用__len__()函数
        return len(self.imgs)

    def center_crop(self, data, label, crop_size):
        '''

        :param data: 需要中心裁剪的图片
        :param label: 需要中心裁剪的标签
        :param crop_size: 裁剪的尺寸
        :return: 裁剪后的数据及其标签
        '''
        data = ff.center_crop(data, crop_size)
        label = ff.center_crop(label, crop_size)
        return data, label

    def img_transform(self, img, label):
        '''
        对图片进行处理；其中图片是无需编码的，但标签需要
        :param img:
        :param label:
        :return:编码好的图片和标签，都是tensor格式
        '''
        # 将label转为numpy数组的格式并用uint8进行编码再转回来,这样能保证图片中的每个像素都是整型的
        label = np.array(label)
        label = Image.fromarray(label.astype('uint8'))
        # 将图片和标签转为张量并标准化
        transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.Normalizer_params[0], cfg.Normalizer_params[1])
        ])
        img = transforms_img(img)
        # 对标签进行编码，返回的是每个像素对应的类别
        label = label_processor.encode_label_img(label)
        # 将编码好的标签从numpy格式转为tensor格式
        label = torch.from_numpy(label)
        return img, label

    def readFile(self, path):
        '''
        读取路径
        :param path: 文件路径
        :return: 返回每张图片的完整路径
        '''
        # 从文件中读取数据
        files_list = os.listdir(path)  # 得到所有图片的名字
        # 将每张图对应的路径拼接并存储到新列表中（图片的完整路径）
        file_path_list = [os.path.join(path, img_name) for img_name in files_list]
        file_path_list.sort()  # 进行排序？
        return file_path_list


# 标签进行编码处理
class LabelProcessor:
    def __init__(self, file_path):
        # 从文件中读取每个类别对应的颜色像素值
        self.colormap = self.read_color_map(file_path)
        # 对标签做编码
        self.cm21b1 = self.encoder_label_pix(self.colormap)

    @staticmethod
    # 静态方法修饰器(定义在类中的普通函数)，可以用self.name的形式进行调用
    # 在静态方法内部不能调用 self.相关内容
    # 使用这个因为(简洁代码，封装功能等)
    def read_color_map(file_path):
        '''
        将标签对应的颜色进行存储
        :param :
        :return:返回一个列表(包含了所有类别的颜色对应的像素值)
        '''
        # 加载csv文件，以逗号隔开
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]  # 按行读取
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap

    @staticmethod
    def encoder_label_pix(colormap):  # 标签编码并返回哈希表
        '''
        用哈希算法加快查找和检索的效率;即完成了像素点到类别的一一映射关系
        :param colormap: 颜色列表
        :return:对应的哈希表
        '''
        # 类似于256进制使用哈希映射
        # (cm[0] * 256 + cm[1]) * 256 + cm[2] 希函数
        # cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i 哈希映射
        cm2lbl = np.zeros(256 ** 3)  # 哈希表
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):
        '''
        使用哈希表，将每个像素点对应的像素值做计算，再用获得的值去哈希表中进行查找其属于那种类别
        :param img:需要编码的标签
        :return:返回该标签每个像素点对应的类别
        '''
        data = np.array(img, dtype='int32')  # 转为numpy格式
        # idx即为哈希表每个像素的索引对应的类别
        idx = (data[:, :, 0] * 256 + data[:, :, 1] * 256 + data[:, :, 2])
        return np.array(self.cm21b1[idx], dtype='int64')


# 实例化标签编码对象
label_processor = LabelProcessor(cfg.class_dict_path)

if __name__ == '__main__':
    # 调用训练路径
    Train_root = cfg.TRAIN_ROOT
    Train_label = cfg.TRAIN_LABLE
    # 调用验证路径
    Val_root = cfg.VAL_ROOT
    Val_label = cfg.VAL_LABEL
    # 调用测试路径
    Test_root = cfg.TEST_ROOT
    Test_label = cfg.TEST_LABEL
    # 输入尺寸
    crop_size = cfg.crop_size

    # 调用预处理类(选择模式)
    Train = theDataset([Train_root, Train_label], crop_size)
    # Val = theDataset([Val_root, Val_label], crop_size)
    # Test = theDataset([Test_root, Test_label], crop_size)
