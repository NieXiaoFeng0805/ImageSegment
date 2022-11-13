# Data

存放数据集(训练、验证、测试)

# ResultImage存储结果

# Models：存放模型和权重

## 	Weights：存放训练的权重文件

## 		FCNmodel：训练模型

# cfg.py：配置文件

包含训练批次、迭代次数、学习率、数据路径、分类标签及其颜色路径、图片处理尺寸等

# dataset.py：数据预处理

## 		数据处理类：theDataset(Dataset)

构造函数 ：初始化方法，获取数据的路径(图像和标签)和尺寸大小

```python
def __init__(self, file_path=None, crop_size=None):
```

获取图片及其相对应的标签并进行处理

```python
def __getitem__(self, index):
```

​	**其中**包含了对图片及标签的裁剪和对标签的编码操作

```python
def center_crop(self, data, label, crop_size):
def img_transform(self, img, label):
```



## 	标签编码类：LabelProcessor

处理标签，对标签进行编码(cm2lbl)；其中包含两个静态方法**read_color_map**和**encoder_label_pix**。

**read_color_map**：加载标签分类及其对应的颜色文件，一般是csv文件；返回颜色表

**encoder_label_pix**：对标签进行编码并返回一个哈希表

**encode_label_img**：调用哈希表，将每个像素点对应的像素值做计算，再用获得的值去哈希表中进行查找并返回每个像素点对应的类别



# train.py：训练模型

# test.py：测试模型

# predict.py：预测模型