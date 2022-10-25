"""
@FileName：FCNmodel.py
@Description：建立FCN模型
@Author：Feng
@Time：2022/10/22 17:36
"""
import numpy as np
import torch
from torchvision import models
from torch import nn
import FCN.cfg as cfg
from Tools import Bilinear_kernel as bi

pretrained_net = models.vgg16_bn()  # 获取带批量归一化的vgg16模型(不含FC层)


# 加载预训练权重
# pre_weight = torch.load(cfg.PRE_WEIGHT)
# pretrained_net.load_state_dict(pre_weight)


class FCN(nn.Module):
    def __init__(self, num_classes):  # 传入分类总数目
        super().__init__()  # 继承父类的init方法
        self.stage1 = pretrained_net.features[:7]
        self.stage2 = pretrained_net.features[7:14]
        self.stage3 = pretrained_net.features[14:24]
        self.stage4 = pretrained_net.features[24:34]
        self.stage5 = pretrained_net.features[34:]

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(512, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        # 过渡
        self.conv_trains1 = nn.Conv2d(512, 256, 1)
        self.conv_trains2 = nn.Conv2d(256, num_classes, 1)
        # FCN-8s
        # 八倍
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bi.Bilinear_K(num_classes, num_classes, 16)
        # 二倍1
        self.upsample_2x_1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        self.upsample_2x_1.weight.data = bi.Bilinear_K(512, 512, 4)  # 初始化反卷积核
        # 二倍2
        self.upsample_2x_2 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False)
        self.upsample_2x_2.weight.data = bi.Bilinear_K(256, 256, 4)

    def forward(self, x):  # 原图尺寸（352,480,3）
        s1 = self.stage1(x)  # s1特征图：176,240,64
        s2 = self.stage2(s1)  # s2特征图：88,120,128
        s3 = self.stage3(s2)  # s3特征图：44,60,256
        s4 = self.stage4(s3)  # s4特征图：22,30,512
        s5 = self.stage5(s4)  # s5特征图：11,15,512

        scores1 = self.scores1(s5)  # 11,15,12
        s5 = self.upsample_2x_1(s5)  # 22,30,512
        add1 = s4 + s5  # 第一次特征融合

        scores2 = self.scores2(add1)  # 22,30,12
        add1 = self.conv_trains1(add1)  # 22,30,256
        add1 = self.upsample_2x_2(add1)  # 44,60,256

        add2 = add1 + s3  # 第二次特征融合 # 44,60,256

        add2 = self.conv_trains2(add2)  # 44,60,12
        scores3 = self.upsample_8x(add2)  # 352,480,12
        return scores3
