# author: Feng
# contact: 1245272985@qq.com
# datetime:2022/11/14 19:03
# software: PyCharm
"""
文件说明：配置文件

"""
In_channel = 3
n_classes = 12
out_stride = 16

BATCH_SIZE = 2  # 训练批次
EPOCH = 250  # 迭代次数
LEARNING_RATE = 1e-4  # 学习率
CHANNELS = 3  # 图像通道数
DATA_KINDS = 12  # 数据集的类别

# 眼球毛细血管数据集
Train_data = 'data/Unet_crop/training/images'
Train_label = 'data/Unet_crop/training/labels'

Test_data = 'data/test/images'



# 数据集图片尺寸处理
crop_size = (352, 480)

# 标准化参数
Normalizer_params = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

# 分类标签及其对应颜色
class_dict_path = 'data/CamVid/class_dict.csv'
