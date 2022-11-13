"""
@FileName：test.py
@Description：测试
@Author：Feng
@Time：2022/10/24 16:18
"""
import torch
from torch.utils.data import DataLoader
from Dataset import theDataset
from Models.FCNmodel import FCN
from torch.autograd import Variable
import torch.nn.functional as F
from Tools import CalculateIndicators
import FCN.cfg as cfg

# 设置测试设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
miou_list = [0]

# 导入测试集
the_test = theDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(the_test, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)

net = FCN(cfg.DATA_KINDS)  # 网络模型选择
net.eval()  # 验证模式
net.to(device)  # 模型放置
net.load_state_dict(torch.load('Models/Weights/FirstTest/183.pth'))  # 权重导入

train_acc = 0
train_miou = 0
train_classes_acc = 0  # 类像素精度
train_mpa = 0  # 均像素精度
error = 0

for i, sample in enumerate(test_data):
    img_data = Variable(sample['img']).to(device)
    img_label = Variable(sample['label']).to(device)
    out = net(img_data)
    out = F.log_softmax(out, dim=1)

    pre_label = out.max(dim=1)[1].data.cpu().numpy()
    pre_label = [i for i in pre_label]

    true_label = img_label.data.cpu().numpy()
    true_label = [i for i in true_label]

    eval_matrix = CalculateIndicators.eval_semantic_segmentation(pre_label, true_label)
    train_acc = eval_matrix['mean_classes_acc'] + train_acc
    train_miou = eval_matrix['MIoU'] + train_miou
    train_mpa = eval_matrix['pixel_acc'] + train_mpa
    if len(eval_matrix['classes_acc']) < 12:
        eval_matrix['classes_acc'] = 0
        train_classes_acc = train_classes_acc + eval_matrix['classes_acc']
        error += 1
    else:
        train_classes_acc = train_classes_acc + eval_matrix['classes_acc']
    print(eval_matrix['classes_acc'], '=================', i)
epoch_str = (
    'Test_acc:{:.5f} , Test_MIou:{:.5f} , Test_mpa:{:.5f} , Test_classes-acc:{:}'
        .format(train_acc / (len(test_data) - error),
                train_miou / (len(test_data) - error),
                train_mpa / (len(test_data) - error),
                train_classes_acc / (len(test_data) - error)),
)

if train_miou / (len(test_data) - error) > max(miou_list):
    miou_list.append(train_miou / (len(test_data) - error))
    print(epoch_str)
    print('===========last')
