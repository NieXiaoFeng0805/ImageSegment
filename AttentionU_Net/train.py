"""
@FileName：train.py
@Description：
@Author：Feng
@Time：2022/11/7 16:08
"""
import os

from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from AttUnet import AttUNet
from dataset import TheDataset
import torch.nn.functional as F
from torchvision.utils import save_image
import cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# weight_path = 'weight/weight_30.pth'
weight_path = ''
# data_path = 'data/Unet_crop/training/images'

if __name__ == '__main__':
    data_loader = DataLoader(TheDataset([cfg.Train_data, cfg.Train_label]),
                             batch_size=1, shuffle=True)
    net = AttUNet(3, 3).to(device)  # 放置网络
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
    else:
        print('NO Pre_weight can used')
    opt = optim.Adam(net.parameters())
    loss_func = nn.BCELoss()

    epoch = 1
    while epoch <= 50:
        for i, (image, label) in enumerate(data_loader):
            image, label = image.to(device), label.to(device)
            out = net(image)
            train_loss = loss_func(out, label)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 5 == 0:
                print(f'{epoch}--{i}-train_loss==>{train_loss.item()}')

            _image = image[0]
            _seg_image = label[0]
            _out = out[0]
            last_img = torch.stack([_image, _seg_image, _out], dim=0)
            save_image(last_img, 'Result/result_{}.png'.format(i))
        if epoch % 10 == 0:
            torch.save(net.state_dict(), 'Weights/weight_{}.pth'.format(epoch))
        epoch += 1
