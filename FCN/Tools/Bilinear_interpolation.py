"""
@FileName：Bilinear_interpolation.py
@Description：双线性插值进行图像的放大
@Author：Feng
@Time：2022/10/22 16:25
"""
import cv2
import numpy as np
import FCN.cfg as cfg


def Bilinear_insert(srcImg, new_size):
    '''
    使用双线性插值将图像放大
    :param srcImg: 输入图像
    :param new_size: 目标尺寸
    :return: 目标图像
    '''
    src_h, src_w = srcImg.shape[:2]  # 原始图像宽高
    ob_h, ob_w = new_size  # 目标图像的宽高
    if ob_h == src_h and src_w == ob_w:
        return srcImg.copy()

    scale_x = float(src_w) / ob_w  # 缩放比例
    scale_y = float(src_h) / ob_h

    # 遍历目标图上的每个像素，由原图的点插入数值
    # 生成一张目标尺寸大小的空白图，遍历插值
    ob = np.zeros((ob_h, ob_w, cfg.CHANNELS), dtype=np.uint8)
    for n in range(cfg.CHANNELS):
        for ob_y in range(ob_h):
            for ob_x in range(ob_w):
                # 目标像素在原图上的坐标
                src_x = (ob_x + 0.5) * scale_x - 0.5
                src_y = (ob_y + 0.5) * scale_y - 0.5

                # 计算其在原图上四个临近点的位置
                src_x_0 = int(float(src_x))  # 向下取整
                src_y_0 = int(float(src_y))
                # 防止出界
                src_x_1 = min(src_x_0 + 1, src_w - 1)
                src_y_1 = min(src_y_0 + 1, src_h - 1)
                # 双线性插值
                value0 = (src_x_1 - src_x) * srcImg[src_y_0, src_x_0, n] + (src_x - src_x_0) * srcImg[
                    src_y_0, src_x_1, n]
                value1 = (src_x_1 - src_x) * srcImg[src_y_1, src_x_0, n] + (src_x - src_x_0) * srcImg[
                    src_y_1, src_x_1, n]
                ob[ob_y, ob_x, n] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)
    return ob

# if __name__ == '__main__':
#     img_in = cv2.imread('../Data/CamVid/train/0001TP_006690.png')
#     img_out = Bilinear_insert(img_in, (500, 500))
#     cv2.imshow('src', img_in)
#     cv2.imshow('object', img_out)
#     cv2.waitKey(0)
