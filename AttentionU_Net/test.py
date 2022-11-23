# author: Feng
# contact: 1245272985@qq.com
# datetime:2022/11/21 16:30
# software: PyCharm
"""
文件说明：测试文件

"""
from AttUnet import AttUNet
from dataset import *
from Size_change import *
import torchvision.transforms as transforms
import cfg
from torchvision.utils import save_image

# 选择权重
weight = 'Weights/weight_100.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = AttUNet(3, 3).to(device)
net.load_state_dict(torch.load(weight))
net.eval()

test_data = 'data/Unet_crop/test/test_0.png'
img = keep_image_size_open(test_data)

transform = transforms.Compose([
    transforms.ToTensor()
])

img_data = transform(img).to(device)
# print(img_data.shape)

# 升维，将 bathSize 维度加入
img_data = torch.unsqueeze(img_data, dim=0)
# print(img_data.shape)

out = net(img_data)
save_image(out, 'predictResult/predict.png')

# test_pa
# for i in range(len())
