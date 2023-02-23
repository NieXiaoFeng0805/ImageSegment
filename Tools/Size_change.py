"""
@FileName：Size_change.py
@Description：更改图片尺寸
@Author：Feng
@Time：2022/11/1 16:58
"""
import os
import os.path
from PIL import Image
import torch
import matplotlib.pyplot as plt


def ResizeImage(filein, width, height, type):
    '''
    filein: 输入图片
    fileout: 输出图片
    width: 输出图片宽度
    height:输出图片高度
    type:输出图片类型（png, gif, jpeg...）
    '''

    for i, v in enumerate(filein):
        img = Image.open(v)
        out = img.resize((width, height), Image.ANTIALIAS)
        fileout = 'data/Unet_crop/label_{}.png'.format(i)
        out.save(fileout, type)


def Get_data(file_path=None):
    '''
    获取图片数据
    :param file_path:
    :return:
    '''
    # 从文件中读取数据
    files_list = os.listdir(file_path)  # 得到所有图片的名字
    # print('files_list:', files_list)
    # 将每张图对应的路径拼接并存储到新列表中（图片的完整路径）
    file_path_list = [os.path.join(file_path, img_name) for img_name in files_list]
    file_path_list.sort()  # 进行排序
    print('file_path_list: ', file_path_list)
    return file_path_list


def keep_image_size_open(path, size=(572, 572)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

