"""
@FileName：cfg.py
@Description：配置文件
@Author：Feng
@Time：2022/10/31 16:38
"""
BATCH_SIZE = 2  # 训练批次
EPOCH = 50  # 迭代次数
LEARNING_RATE = 1e-4  # 学习率
CHANNELS = 3  # 图像通道数
DATA_KINDS = 12  # 数据集的类别

# CamVid数据集
TRAIN_ROOT = 'data/CamVid/train'
TRAIN_LABEL = 'data/CamVid/train_labels'

VAL_ROOT = 'data/CamVid/val'
VAL_LABEL = 'data/CamVid/val_labels'

TEST_ROOT = 'data/CamVid/test'
TEST_LABEL = 'data/CamVid/test_labels'

# UNet数据集路径
UNet_TRAIN_DATA = 'Data/Unet_crop/training/images'
UNet_TRAIN_LABEL = 'Data/Unet_crop/training/labels'
UNet_TEST_DATA = 'Data/Unet_crop/test/images'
UNet_TEST_LABEL = 'Data/Unet_crop/test/mask'

# 预测分割保存图片路径
RESULT_IMAGE = 'Result/xxx'

# 分类标签及其对应颜色
class_dict_path = 'data/CamVid/class_dict.csv'

# 预训练权重路径
PRE_WEIGHT = 'Models/Weights/weights_19.pth'

# 标准化参数
Normalizer_params = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
# Normalizer_params = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]

# 数据集图片尺寸处理
crop_size = (352, 480)
