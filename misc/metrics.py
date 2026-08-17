from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]

    # 将热图重塑为 (batch_size, num_joints, height * width) --> (8,14,4096)
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))

    # 获取每个关键点对应位置的索引和最大概率值
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    # 将最大概率值和索引重塑为 (batch_size, num_joints, 1)
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    # 创建预测结果张量，并复制索引值，得到形状为 (batch_size, num_joints, 2)
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    # 对预测结果进行调整，将最大值索引映射回（x,y）坐标，并确保 x 坐标在图像宽度范围内，y 坐标在图像高度范围内
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    # 创建掩码，用于过滤掉概率为零的预测值
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    # 将预测结果与掩码相乘，过滤掉概率为零的预测值
    preds *= pred_mask
    return preds, maxvals