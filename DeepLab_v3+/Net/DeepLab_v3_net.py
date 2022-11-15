# author: Feng
# contact: 1245272985@qq.com
# datetime:2022/11/13 19:20
# software: PyCharm
"""
文件说明：DeepLab_v3+ 网络实现

"""
import torch
import torchvision.models as models
from torch import nn
import math
import torch.nn.functional as F

from Net.ASPPModule import ASPP_Module
from Net.DepthSeparable_conv import SeparableConv2d_same
from Net.Xception import Xception
import cfg


class DeepLab_v3_plus(nn.Module):
    def __init__(self, in_channel=3, n_classes=cfg.n_classes, os=cfg.out_stride, _print=True):
        if _print:
            print('Constructing DeepLab_v3+ model...')
            print('Backbone: Xception')
            print('Number of classes: {}'.format(n_classes))
            print('Output Stride: {}'.format(os))
            print('Number of Input Channels: {}'.format(in_channel))
        super(DeepLab_v3_plus, self).__init__()

        # 组合空洞卷积
        self.x_ception_features = Xception(in_channel, os)
        self.ASPP = ASPP_Module(2048, 256, 16)

        # ——————————————————————————解码部分——————————————————————————————————

        # ASPP 之后进入解码部分，先进入一个1×1的卷积
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        # 之前保存的特征图进入1×1的卷积以密集特征
        self.conv2 = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        # 解码器最后几层
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        )

    def forward(self, input):
        x, low_level_features = self.x_ception_features(input)
        x = self.ASPP(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 恢复得到原图尺寸1/4大小的上采样线性插值特征图（之前出来时是1/16）
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear')

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        # 恢复到原图大小
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def _init_wight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# 测试
# if __name__ == '__main__':
#     model = DeepLab_v3_plus(3, 12, 16, True)
#     model.eval()
#     # batch_size,channel,176×240
#     image = torch.randn(1, 3, 352, 480)
#     output = model(image)
#     print('输出维度：', output.size())
