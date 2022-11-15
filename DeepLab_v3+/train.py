# author: Feng
# contact: 1245272985@qq.com
# datetime:2022/11/15 10:58
# software: PyCharm
"""
文件说明：训练文件

"""
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import LoadDataset
from evaluation_segmentation import *
from Net import DeepLab_v3_net
import cfg

# 训练设备
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
num_class = cfg.DATA_KINDS  # 分类总数

# 制作数据集
Load_train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
Load_val = LoadDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

# 读取数据
train_data = DataLoader(Load_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=1)
val_data = DataLoader(Load_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=1)
# 将模型放置设备
deeplab_v3plus = DeepLab_v3_net.DeepLab_v3_plus().to(device)

# 损失函数
criterion = nn.NLLLoss().to(device)
# 优化器
optimizer = optim.Adam(deeplab_v3plus.parameters(), lr=cfg.LEARNING_RATE)


# 进行训练
def train(model):
    best = [0]  # 保存最好指标
    net = model.train()
    # 训练轮次
    for epoch in range(cfg.EPOCH):
        print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH))  # 打印轮次及进度
        # 学习率变化策略
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
        # 指标初始化
        train_loss = 0  # 损失值
        train_acc = 0  # 准确率
        train_miou = 0  # MIoU
        train_class_acc = 0  # 类准确率
        # 训练批次
        for i, sample in enumerate(train_data):
            # 载入数据
            img_data = Variable(sample['img'].to(device))
            img_label = Variable(sample['label'].to(device))
            # 训练
            out = net(img_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            train_loss += loss.item()  # 损失叠加

            # 评估
            # 预测标签
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]
            # 真实标签
            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metrix['mean_class_accuracy']  # 准确率叠加
            train_miou += eval_metrix['miou']  # MIoU叠加
            train_class_acc += eval_metrix['class_accuracy']  # 类准确度叠加
            if i % 2 == 0:
                print('|batch[{}/{}]|batch_loss {: .5f}|'.format(i + 1, len(train_data), loss.item()))

        metric_description = '|Train Acc|: {:.5f}|Train Mean IU|: {:.5f}\n|Train_class_acc|:{:}'.format(
            train_acc / len(train_data),  # 每轮平均准确率
            train_miou / len(train_data),  # 每轮平均IoU
            train_class_acc / len(train_data),  # 每轮平均类准确率
        )

        print(metric_description)
        # 保存最好结果
        if max(best) <= train_miou / len(train_data):
            best.append(train_miou / len(train_data))
            t.save(net.state_dict(), 'Weights/DeepLab_v3+_{}.pth'.format(epoch))


if __name__ == "__main__":
    train(deeplab_v3plus)
