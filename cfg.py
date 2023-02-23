# author: Feng
# contact: 1245272985@qq.com
# datetime:2022/11/27 14:06
# software: PyCharm
"""
文件说明：通用配置文件

"""

# 数据集路径

### CamVid数据集 ###
# 训练集及其标签
CamVid_train_img = '../Data/CamVid/train'
CamVid_train_label = '../Data/CamVid/train_labels'
# 验证集及其标签
CamVid_val_img = '../Data/CamVid/val'
CamVid_val_label = '../Data/CamVid/val_labels'
# 测试集及其标签
CamVid_test_img = '../Data/CamVid/test'
CamVid_test_labels = '../Data/CamVid/test_labels'
# 分类标签及其对应颜色
class_dict_path = '../Data/CamVid/class_dict.csv'

### 眼球毛细血管数据集 ###
Eye_train_img = '../Data/Unet_crop/training/images'
Eye_train_label = '../Data/Unet_crop/training/labels'
Eye_test_img = '../Data/Unet_crop/test'

# 训练批次设定
TOTAL_EPOCH = 200

# 超参数设定
LEARNING_RATE = 1e-4  # 学习率
Normalizer_params = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]  # 标准化参数
CROP_SIZE = (384, 384)  # 裁剪尺寸
BATCH_SIZE = 2  # 每次读入的数据量
