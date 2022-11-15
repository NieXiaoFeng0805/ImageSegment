# author: Feng
# contact: 1245272985@qq.com
# datetime:2022/11/14 15:27
# software: PyCharm
"""
文件说明：深度可分离卷积的实现（编码部分）

"""
import torch
import torchvision.models as models
from torch import nn
import math
import torch.nn.functional as F


# padding 的设置
def fixed_padding(inputs, kernel_size, dilation):
    '''
    padding 的设置,防止当卷积核是偶数的时候出现奇奇怪怪的bug
    :param inputs:
    :param kernel_size:要求的卷积核
    :param dilation:要求的膨胀率
    :return:
    '''
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)  # 有效的卷积核尺寸
    # padding 的计算
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    # 补完0之后的padding，（四周需要补的padding数量可能不一样
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


# 深度可分离卷积的定义
class SeparableConv2d_same(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d_same, self).__init__()

        # 逐通道卷积
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=0,
                               dilation=dilation,
                               groups=in_channel, bias=bias)

        # 逐点卷积
        self.pointwise = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, dilation=dilation,
                                   groups=1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
