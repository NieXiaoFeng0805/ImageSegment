"""
@FileName：cfg.py
@Description：配置文件
@Author：Feng
@Time：2022/10/22 10:53
"""
BATCH_SIZE = 4  # 训练批次
EPOCH = 500  # 迭代次数
LEARNING_RATE = 0.01  # 学习率
CHANNELS = 3  # 图像通道数
DATA_KINDS = 12  # 数据集的类别
# 训练集及其标签
TRAIN_ROOT = 'Data/CamVid/train'
TRAIN_LABLE = 'Data/CamVid/train_labels'
# 验证集及其标签
VAL_ROOT = 'Data/CamVid/val'
VAL_LABEL = 'Data/CamVid/val_labels'
# 测试集及其标签
TEST_ROOT = 'Data/CamVid/test'
TEST_LABEL = 'Data/CamVid/test_labels'

# 预测分割保存图片路径
RESULT_IMAGE = 'ResultImage/FirstTest/'

# 预训练权重路径
PRE_WEIGHT = 'Models/Weights/xxx.pth'

# 分类标签及其对应颜色
class_dict_path = 'Data/CamVid/class_dict.csv'
# 数据集图片尺寸处理
crop_size = (352, 480)
# 标签处理编码格式,可以添加
label_coder = ['uint8']
# 标准化参数
Normalizer_params = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

