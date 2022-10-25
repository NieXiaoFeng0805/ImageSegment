"""
@FileName：train.py
@Description：训练模型
@Author：Feng
@Time：2022/10/22 12:16
"""
import torch
from torch.utils.data import DataLoader
from Dataset import theDataset
import FCN.cfg as cfg
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from datetime import datetime
from Models.FCNmodel import FCN
from Tools import CalculateIndicators

# 判断是否采用gpu进行训练
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 调用预处理
the_train = theDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABLE], cfg.crop_size)
the_val = theDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

# 加载数据
# 数据加载器DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None)
#     # 构建可迭代的数据装载器
#     # dataset: Dataset 类，决定数据从哪里读取以及如何读取
#     # batchsize: 批大小
#     # num_works:num_works: 是否多进程读取数据,gpu可以选择4或更高
#     # sheuffle: 每个 epoch 是否乱序
#     # drop_last: 当样本数不能被 batchsize 整除时，是否舍弃最后一批数据
train_data = DataLoader(the_train,
                        batch_size=cfg.BATCH_SIZE,
                        shuffle=True,
                        num_workers=1)
val_data = DataLoader(the_val,
                      batch_size=cfg.BATCH_SIZE,
                      shuffle=True,
                      num_workers=1)

fcn_8s = FCN(cfg.DATA_KINDS)  # 调用模型
fcn_8s = fcn_8s.to(device)  # 放置模型
lossFunc = nn.NLLLoss().to(device)  # 使用NLL损失函数，可以自定义最后一层激活函数和分类
# 优化函数；RGB数据集一般Adam比较好；RGBD数据集SGD比较好
optimizer = optim.Adam(fcn_8s.parameters(), lr=cfg.LEARNING_RATE)


# 训练模式
def train(model):
    '''
    定义训练函数
    :param model:使用的模型
    :return:
    '''
    best = [0]  # 保存最好权重
    net = model.train()  # 表明正在训练模式
    for epoch in range(cfg.EPOCH):  # 大循环
        print("Epoch is [{}/{}]".format(epoch + 1, cfg.EPOCH))  # 打印消息
        if epoch % 50 == 0 and epoch != 0:  # 更改学习策略
            for group in optimizer.param_groups:
                group['lr'] *= 0.5  # 这里选择的策略是每隔50轮将学习率降低一半
        # 各项指标初始化
        train_loss = 0  # 失误
        train_acc = 0  # 准确率
        train_miou = 0  # 平均IoU
        train_classes_acc = 0  # 分类准确率
        # 开始训练
        for i, sample in enumerate(train_data):  # 小循环，计算批次
            img_data = Variable(sample['img'].to(device))
            img_label = Variable(sample['label'].to(device))
            out = net(img_data)  # 通过模型训练得到的输出
            out = F.log_softmax(out, dim=1)  # 手动在维度为1上分类(因为使用了NLL)
            loss = lossFunc(out, img_label)  # 计算损失(一次小循环的loss)

            optimizer.zero_grad()  # 梯度清零，因为每次都是一个新过程

            loss.backward()  # 反向传播以学习权重
            optimizer.step()  # 进行权重更新

            train_loss += loss.item()  # 累加一次大循环中的loss

            pre_label = out.max(dim=1)[1].data.cpu().numpy()  # 去预测结果中最好的索引
            pre_label = [i for i in pre_label]  # numpy转列表

            true_label = img_label.data.cpu().numpy()  # 真实标签
            true_label = [i for i in true_label]  # numpy转列表
            # 计算混淆矩阵
            eval_matrix = CalculateIndicators.eval_semantic_segmentation(pre_label, true_label)
            # 计算各个指标，从混淆矩阵中对应的键去找
            train_acc += eval_matrix["mean_classes_acc"]
            train_miou += eval_matrix["MIoU"]
            train_classes_acc += eval_matrix["classes_acc"]
            # 打印
            print('|batch[{}/{}]| batch_loss {:.8f}|'
                  .format(i + 1, len(train_data), loss.item()))
        # 一次大循环的平均的各项指标值
        matrix_description = "|Train Acc|: {:.5f}\n|Train MIou|: {:.5f}\n|Train_classes_acc|:{:}".format(
            train_acc / len(train_data),
            train_miou / len(train_data),
            train_classes_acc / len(train_data))
        print(matrix_description)
        # 保存最好权重
        if max(best) <= train_miou / len(train_data):
            best.append(train_miou / len(train_data))
            torch.save(net.state_dict(), 'Models/Weights/FirstTest/{}.pth'.format(epoch))


# 验证模式,没有学习的过程，即无反向传播过程
def evaluate(model):
    net = model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_miou = 0
    eval_classes_acc = 0

    prec_time = datetime.now()
    for j, sample in enumerate(val_data):
        valImg = Variable(sample['img'].to(device))
        valLabel = Variable(sample['label'].long().to(device))

        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        loss = lossFunc(out, valLabel)
        eval_loss = loss.item() + eval_loss

        pre_label = out.max(dim=1)[1].data.cup().numpy()
        pre_label = [i for i in pre_label]

        true_label = valLabel.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metrices = CalculateIndicators.eval_semantic_segmentation(pre_label, true_label)
        eval_acc = eval_metrices['mean_classes_acc'] + eval_acc
        eval_miou = eval_metrices['MIoU'] + eval_miou

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{.0f}'.format(h, m, s)
    val_str = ('|Valid Loss|: {:.5f} \n|Valid Acc|: {:.5f}\n|Valid MIoU|: {:.5f}\n|Valid_Classes_acc|: {:}'
               .format(eval_loss / len(train_data),
                       eval_acc / len(val_data),
                       eval_miou / len(val_data),
                       eval_classes_acc / len(val_data)))
    print(val_str)
    print(time_str)


if __name__ == '__main__':
    train(fcn_8s)
