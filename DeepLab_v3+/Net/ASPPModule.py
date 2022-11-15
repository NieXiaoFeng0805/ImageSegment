# author: Feng
# contact: 1245272985@qq.com
# datetime:2022/11/13 21:58
# software: PyCharm
"""
文件说明： 完全封装的 ASPP 模块

"""
import torch
import torchvision.models as models
from torch import nn
import math
import torch.nn.functional as F


class ASPP_Module(nn.Module):
    def __init__(self, in_channel, out_channel, output_strid):
        '''
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param output_strid: 控制不同的膨胀率
        '''
        super(ASPP_Module, self).__init__()
        dilations = []
        if output_strid == 16:
            dilations = [1, 6, 12, 18]
        elif output_strid == 8:
            dilations = [1, 12, 24, 36]

        # 因为是并行的，所以所有的块的输入通道和输出通道都是相同的
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, dilation=dilations[0], bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1],
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2],
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[3],
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 结果应该是[1,2048,1,1]
            # 全局平均池化更改通道数
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False),  # 更改通道数(-->256)方便连接
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # 将前面的输出进行concat 5*256；（concat在前向传播中进行实现）
        self.conv1 = nn.Conv2d(1280, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # 初始化卷积核(可有可无)
        # self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        # print('x1 size = ', x1.size())
        x2 = self.aspp2(x)
        # print('x2 size = ', x2.size())
        x3 = self.aspp3(x)
        # print('x3 size = ', x3.size())
        x4 = self.aspp4(x)
        # print('x4 size = ', x4.size())

        x5 = self.global_avg_pool(x)
        # 对x5进行处理，与上面的维度相同后才能进行拼接（这里选了x4的宽高）
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        # print('x5 size = ', x5.size())

        # 进行拼接
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # print('拼接后的维度', x.size())
        # 进行1×1 的卷积和归一化处理
        x = self.conv1(x)
        # print('经过1×1卷积后的维度', x.size())
        x = self.bn1(x)
        return x


# 测试
# model = ASPP_Module(2048, 256, 16)
# model.eval()
# image = torch.randn(1, 2048, 176, 240)
# output = model(image)
# print('result size = ', output.size())
