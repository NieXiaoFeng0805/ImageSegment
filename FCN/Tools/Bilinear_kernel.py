"""
@FileName：Bilinear_kernel.py
@Description：双线性插值初始化反卷积核
@Author：Feng
@Time：2022/10/22 17:22
"""
import numpy as np
import torch


def Bilinear_K(in_channels, out_channels, kernel_size):
    '''
    用双线性插值初始化卷积层中卷积核的权重参数
    :param in_channels:输入通道数
    :param out_channels:输出通道数
    :param kernel_size:核大小
    :return: 经过插值后的得到的卷积核(tensor格式，由numpy转化而来)
    '''
    factor = (kernel_size + 1) // 2  #
    center = kernel_size / 2
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)
