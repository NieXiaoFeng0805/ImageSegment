# author: Feng
# contact: 1245272985@qq.com
# datetime:2022/11/14 15:56
# software: PyCharm
"""
文件说明：Xception的实现；由20个残差块和5个普通（深度）卷积组成

"""
import torch
import torchvision.models as models
from torch import nn
import math
import torch.nn.functional as F
from Net.DepthSeparable_conv import SeparableConv2d_same


# 残差块的定义
# 预先处理块：Entry Flow；更改尺寸，将原图缩放到1/16
# 中间重复块：Middle Flow；重复进行特征提取但不改变图像尺寸
# 最后输出块：Exit Flow；最后的融合输出流程

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, common_point, stride=1, dilation=1, start_with_relu=True,
                 is_last=False):
        '''
        :param in_channel: 输出通道
        :param out_channel: 输出通道
        :param common_point: 每个Block间小块的共同点，（比如有两个相同的卷积则值就为2）
        :param stride: 步长
        :param dilation: 膨胀率
        :param start_with_relu: 是否从Relu（）激活开始
        :param is_last:是否是最后一个小模块
        '''
        super(Block, self).__init__()

        # 针对中间的重复层进行判断，当输入通道数和输出通道数相同且步长为1时，短跳连接没有卷积操作，其他时候有
        if out_channel != in_channel or stride != 1:  # 进入短跳块
            self.skip = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channel)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)

        block_list = []

        block_list.append(self.relu)  # 先将Relu()添加到列表中，用不用由条件进行判断取值
        block_list.append(SeparableConv2d_same(in_channel, out_channel, 3, stride=1, dilation=dilation))
        block_list.append(nn.BatchNorm2d(out_channel))
        filters = out_channel  # 与通道相同的卷积核

        # 补充与之前一样的结构
        for i in range(common_point - 1):  # 减1是之前已经定义了一个一样的（前面通道数不一样）
            block_list.append(self.relu)
            block_list.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation))
            block_list.append(nn.BatchNorm2d(filters))

        if not start_with_relu:
            block_list = block_list[1:]
        if stride != 1:
            block_list.append(SeparableConv2d_same(out_channel, out_channel, 3, stride=2))
        if stride == 1 and is_last:  # 最后一层
            block_list.append(SeparableConv2d_same(out_channel, out_channel, 3, stride=1))
        # 将上述按顺序添加好的列表进行解压后再放到Sequential模块中
        self.block_list = nn.Sequential(*block_list)  # 一个小块中的全部内容

    # 前向传播
    def forward(self, input):
        x = self.block_list(input)
        if self.skip is not None:  # 进入短跳连接
            skip = self.skip(input)
            skip = self.skip_bn(skip)
        else:
            skip = input
        x += skip
        return x


class Xception(nn.Module):
    def __init__(self, inchannel=3, output_strid=16):
        super(Xception, self).__init__()
        if output_strid == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilation = (1, 2)
        elif output_strid == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilation = (2, 4)
        else:
            raise NotImplementedError

        # Entry Flow
        self.conv1 = nn.Conv2d(inchannel, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)  # 不改变尺寸
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, common_point=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, common_point=2, stride=2, start_with_relu=True)
        self.block3 = Block(256, 728, common_point=2, stride=entry_block3_stride, start_with_relu=True)

        # Middle Flow
        self.block4 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block5 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block6 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block7 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block8 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block9 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block10 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block11 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block12 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block13 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block14 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block15 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block16 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block17 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block18 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block19 = Block(728, 728, common_point=3, stride=1, dilation=middle_block_dilation, start_with_relu=True)

        # Exit Flow
        self.block20 = Block(728, 1024, common_point=2, stride=1, dilation=exit_block_dilation[0], start_with_relu=True,
                             is_last=True)

        self.conv3 = SeparableConv2d_same(1024, 1536, 3, stride=1, dilation=exit_block_dilation)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d_same(1536, 1536, 3, stride=1, dilation=exit_block_dilation)
        self.bn4 = nn.BatchNorm2d(1536)

        self.conv5 = SeparableConv2d_same(1536, 2048, 3, stride=1, dilation=exit_block_dilation)
        self.bn5 = nn.BatchNorm2d(2048)

        # 初始化权重
        self._init_wight()

    def forward(self, x):
        # Entry Flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        low_level_feature_map = x  # 保存之后需要融合的相同尺寸的特征图（1/4）
        x = self.block2(x)
        x = self.block3(x)

        # Middle Flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit Flow
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x, low_level_feature_map

    def _init_wight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
