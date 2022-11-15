# author: Feng
# contact: 1245272985@qq.com
# datetime:2022/11/9 10:33
# software: PyCharm
"""
文件说明：预测文件，以图片形式保存

"""
import numpy as np

from evaluation_segmentation import *
import torch
from dataset import LoadDataset
from Net.DeepLab_v3_net import DeepLab_v3_plus

from torch.autograd import Variable
import torch.nn.functional as F
import cfg
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from torchvision.utils import save_image

# 设置训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 制作数据集
make_test_data = LoadDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(make_test_data, batch_size=1, shuffle=False, num_workers=0)

net = DeepLab_v3_plus().to(device)  # 放置网络
net.load_state_dict(torch.load('Weights/SegNet_weight_150.pth'))  # 加载权重
net.eval()  # 验证模式

pd_label_color = pd.read_csv(cfg.class_dict_path, sep=',')  # 读取分类文件
name_value = pd_label_color['name'].values
# print(name_value)
num_class = len(name_value)
color_map = []

# 将对应的颜色分类加载到 color_map中
for i in range(num_class):
    tmp = pd_label_color.iloc[i]
    color = [tmp['r'], tmp['g'], tmp['b']]
    color_map.append(color)

# 转为numpy数组的个格式
cm = np.array(color_map).astype('uint8')

result_dir = 'Result/'
for i, sample in enumerate(test_data):
    valImg, valLabel = sample['img'].to(device), sample['label'].long().to(device)
    out = net(valImg)
    out = F.log_softmax(out, dim=1)

    pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
    pre = cm[pre_label]

    result = Image.fromarray(pre)
    result.save(result_dir + str(i) + '.png')
    print('{}--Done'.format(i))
    if i == 10:  # 前十张
        break

