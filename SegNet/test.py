# author: Feng
# contact: 1245272985@qq.com
# datetime:2022/11/9 10:09
# software: PyCharm
"""
文件说明：测试文件

"""
from evaluation_segmentation import *
import torch
from dataset import TheDataset
from segnet import VGG16_SegNet
from torch.autograd import Variable
import torch.nn.functional as F
import cfg
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
miou_list = [0]

make_test_data = TheDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(make_test_data, batch_size=2, shuffle=True, num_workers=0)

net = VGG16_SegNet()
net.eval().to(device)
net.load_state_dict(torch.load('Weights/SegNet_50.pth'))

train_acc = 0
train_miou = 0
train_class_acc = 0
train_mpa = 0
error = 0

for i, sample in enumerate(test_data):
    data, label = Variable(sample['img']).to(device), Variable(sample['label']).to(device)
    out = net(data)
    out = F.log_softmax(out, dim=1)

    pre_label = out.max(dim=1)[1].data.cpu().numpy()
    pre_label = [i for i in pre_label]

    true_label = label.data.cpu().numpy()
    true_label = [i for i in true_label]

    eval_metrix = eval_semantic_segmentation(pre_label, true_label)
    train_acc = eval_metrix['mean_classes_acc'] + train_acc
    train_miou = eval_metrix['MIoU'] + train_miou
    train_mpa = eval_metrix['px_acc'] + train_mpa

    if len(eval_metrix['classes_acc']) < 12:
        eval_metrix['classes_acc'] = 0
        train_class_acc = train_class_acc + eval_metrix['classes_acc']
        error += 1
    else:
        train_class_acc = train_class_acc + eval_metrix['classes_acc']

    print(eval_metrix['classes_acc'], '=========', i)

print('test_acc:{:.5f} , test_miou:{:.5f} '.format(train_acc / (len(test_data) - error),
                                                   train_miou / (len(test_data) - error)))
print('test_mpa :{:.5f} , test_class_acc:{:.5f}'.format(train_mpa / (len(test_data) - error),
                                                        train_class_acc / (len(test_data) - error)))

# 更新指标
if train_miou / (len(test_data) - error) > max(miou_list):
    miou_list.append(train_miou / (len(test_data) - error))
    print('==========last')
