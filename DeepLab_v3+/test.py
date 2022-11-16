# author: Feng
# contact: 1245272985@qq.com
# datetime:2022/11/9 10:09
# software: PyCharm
"""
文件说明：测试文件

"""
from evaluation_segmentation import *
import torch
from dataset import LoadDataset
from Net.DeepLab_v3_net import DeepLab_v3_plus
from torch.autograd import Variable
import torch.nn.functional as F
import cfg
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
miou_list = [0]

make_test_data = LoadDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(make_test_data, batch_size=1, shuffle=True, num_workers=0)

net = DeepLab_v3_plus().eval().to(device)
net.load_state_dict(torch.load('Weights/DeepLab_v3+_60.pth'))

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
    train_acc = eval_metrix['mean_class_accuracy'] + train_acc
    train_miou = eval_metrix['miou'] + train_miou
    train_mpa = eval_metrix['pixel_accuracy'] + train_mpa

    if len(eval_metrix['class_accuracy']) < 12:
        eval_metrix['class_accuracy'] = 0
        train_class_acc = train_class_acc + eval_metrix['class_accuracy']
        error += 1
    else:
        train_class_acc = train_class_acc + eval_metrix['class_accuracy']

    print(eval_metrix['class_accuracy'], '=========', i)

print('test_acc:{:.8f} , test_miou:{:.8f} '.format(train_acc / (len(test_data) - error),
                                                   train_miou / (len(test_data) - error)))
print('test_mpa :{:.8f} , test_class_acc:{:.8f}'.format(train_mpa / (len(test_data) - error),
                                                        train_class_acc / (len(test_data) - error)))

# 更新指标
if train_miou / (len(test_data) - error) > max(miou_list):
    miou_list.append(train_miou / (len(test_data) - error))
    print('==========last')
