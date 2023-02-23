# author: Feng
# contact: 1245272985@qq.com
# datetime:2023/2/22 21:43
# software: PyCharm
"""
文件说明：查看数据集图像(2D)

"""
import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

# nii或者nii.gz文件路径
file = 'E:\ImageSegment\MyNetModel\Data\Task08_HepaticVessel\Task08_HepaticVessel\imagesTr\hepaticvessel_150.nii.gz'
img = nib.load(file)

print(img)
print(img.header['db_name'])  # 输出nii的头文件
width, height, queue = img.dataobj.shape
OrthoSlicer3D(img.dataobj).show()

num = 1
for i in range(0, queue, 10):
    img_arr = img.dataobj[:, :, i]
    plt.subplot(5, 4, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1

plt.show()
