"""
@FileName：predict.py
@Description：进行分割预测并分割保存
@Author：Feng
@Time：2022/10/22 12:17
"""
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from Models.FCNmodel import FCN
from Dataset import theDataset
import FCN.cfg as cfg

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
the_test = theDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(the_test, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)

net = FCN(cfg.DATA_KINDS).to(device)
net.load_state_dict(torch.load('xxx.pth'))
net.eval()

pd_label_color = pd.read_csv(cfg.class_dict_path, sep=',')
name_value = pd_label_color['name'].values
num_class = len(name_value)
colormap = []
for i in range(num_class):
    tmp = pd_label_color.iloc[i]
    color = [tmp['r'], tmp['g'], tmp['b']]
    colormap.append(color)

cm = np.array(colormap).astype('uint8')

# 保存预测图的路径
the_dir = cfg.RESULT_IMAGE

for i, sample in enumerate(test_data):
    valImage = sample['img'].to(device)
    valLabel = sample['lable'].to(device)
    out = net(valImage)
    out = F.log_softmax(out, dim=1)
    pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
    pre = cm[pre_label]
    pre1 = Image.fromarray(pre)
    pre1.save(the_dir + str(i) + '.png')
    print('Done')
