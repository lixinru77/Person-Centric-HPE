import cv2
import torch
import numpy as np
import torch.nn as nn
from math import sqrt, pi, log
import torch.nn.functional as F
from misc import get_max_preds, generateTarget
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from uuid import uuid4


#######260316xinru doft-argmax亚像素损失
def soft_argmax_2d(heatmaps, temperature=50.0):
    """
    将 (B, C, H, W) 的热图转化为 (B, C, 2) 的连续坐标
    """
    B, C, H, W = heatmaps.shape
    heatmaps = heatmaps.view(B, C, H * W)

    # 【修复1】：乘以 temperature，让原本 0~1 的特征值放大，使得 softmax 能够产生尖锐的峰值！
    probs = F.softmax(heatmaps * temperature, dim=2)

    #热格子生成坐标编号
    x_grid = torch.arange(W, device=heatmaps.device, dtype=torch.float32)
    y_grid = torch.arange(H, device=heatmaps.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_grid, x_grid, indexing='ij')

    grid_x = grid_x.contiguous().view(1, 1, H * W)
    grid_y = grid_y.contiguous().view(1, 1, H * W)

    #根据概率算坐标的“加权平均
    expected_x = torch.sum(probs * grid_x, dim=2)
    expected_y = torch.sum(probs * grid_y, dim=2)

    return torch.stack([expected_x, expected_y], dim=-1)



class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.alpha = 0.25
        self.gamma = 2.0

    def forward(self, pred, target):

        # 计算交叉熵损失
        cross_entropy = F.cross_entropy(pred, target, reduction='none')

        # 计算Focal Loss
        focal_weight = self.alpha * (1 - torch.exp(-cross_entropy)) ** self.gamma
        loss = torch.mean(focal_weight * cross_entropy)

        return loss

class LossComputer():
    def __init__(self, cfg, device):
        self.device = device
        self.cfg = cfg
        self.numFrames = self.cfg.DATASET.numFrames
        self.numGroupFrames = self.cfg.DATASET.numGroupFrames
        self.numKeypoints = self.cfg.DATASET.numKeypoints
        self.heatmapSize = self.width = self.height = self.cfg.DATASET.heatmapSize
        self.imgSize = self.imgWidth = self.imgHeight = self.cfg.DATASET.imgSize
        self.lossDecay = self.cfg.TRAINING.lossDecay
        self.alpha = 0.9
        self.beta = 1.0
        self.bce = nn.BCELoss()
        self.numObjs = self.cfg.DATASET.numObjs



    # def computeLoss(self, preds_keypoints, gt):
    def computeLoss(self, preds_keypoints, gt, return_heatmaps=False):
        b = gt.size(0)
        heatmaps = torch.zeros((b, self.numKeypoints * self.numObjs, self.height, self.width))
        gtKpts = torch.zeros((b, self.numKeypoints * self.numObjs, 2))

        for i in range(len(gt)):
            #heatmap 是 GT 热图，形状大概是：(batch, keypoints, height, width)
            #gtKpts GT 坐标，形状是：B × 14 × 2, 每个关键点有一个 (x,y) 坐标。
            #generateTarget() 把真实关键点坐标转换成两种监督信号：一种是高斯热图 target，用于 BCE 热图监督；另一种是关键点坐标 targetKpts，用于 L1 坐标监督。
            heatmap, gtKpt = generateTarget(gt[i], self.numKeypoints, self.heatmapSize, self.imgSize)
            heatmaps[i, :] = torch.tensor(heatmap)
            gtKpts[i] = torch.tensor(gtKpt)
        #raw_preds：网络较早阶段/原始分支预测出的热图；
        #gcn_preds：经过后续结构优化后的关键点热图。
        raw_preds, gcn_preds = preds_keypoints

        # ---------------- 1. 恢复原本的 BCE Loss (强迫网络画出正确的火柴人热图) ----------------
        #把网络预测出来的热图，和 GT 生成的标准热图进行比较，计算二者差距
        raw_gt_loss = self.computeBCESingleFrame(
            raw_preds.view(-1, self.numKeypoints * self.numObjs, self.height, self.width), heatmaps)
        gcn_gt_loss_bce = self.computeBCESingleFrame(gcn_preds.view(-1, self.numKeypoints, self.height, self.width),
                                                     heatmaps)

        # ---------------- 2. 创新点 A：Soft-argmax 亚像素坐标辅助 Loss (抠细节) ----------------
        #将模型预测的热图转换为坐标
        pred_coords_gcn = soft_argmax_2d(gcn_preds.view(-1, self.numKeypoints, self.height, self.width))
        #gtKpts GT 坐标，
        target_coords = gtKpts.to(self.device)
        
        l1_loss_fn = nn.L1Loss()
        # 维度对齐处理
        pred_coords_gcn_aligned = pred_coords_gcn.view(b, -1, self.numKeypoints, 2).mean(dim=1)
        # 计算亚像素级偏移误差
        gcn_gt_loss_l1 = l1_loss_fn(pred_coords_gcn_aligned, target_coords)

        # 【核心融合】：BCE负责粗定位，L1负责亚像素微调 (权重设为0.1防破坏收敛)
        gcn_gt_loss = gcn_gt_loss_bce + 0.1 * gcn_gt_loss_l1
        # -----------------------------------------------------------------------------------

        # 形状还原
        raw_preds = raw_preds.permute(0, 2, 1, 3, 4).view(-1, self.numKeypoints * self.numObjs, self.height, self.width)
        gcn_preds = gcn_preds.permute(0, 2, 1, 3, 4).view(-1, self.numKeypoints, self.height, self.width)

        if self.alpha < 1.0:
            self.alpha += self.lossDecay
            self.beta -= self.lossDecay
        if self.lossDecay != -1:
            total_loss_keypoints = self.alpha * raw_gt_loss + self.beta * gcn_gt_loss
        else:
            total_loss_keypoints = raw_gt_loss + gcn_gt_loss

        # ---------------- 3. 【致命 Bug 修复】：输出亚像素坐标给评估脚本 ----------------
        # 坚决弃用粗糙的 get_max_preds，直接输出我们的高精度积分坐标！
        pred2d = pred_coords_gcn_aligned.detach().cpu().numpy()

        # Ground Truth 的坐标还是用老方法提取没关系
        gt2d, _ = get_max_preds(heatmaps.detach().cpu().numpy())
        giou_loss = torch.tensor(float('inf'))

        if return_heatmaps:
            return total_loss_keypoints, giou_loss, gcn_gt_loss, pred2d, gt2d, heatmaps


        return total_loss_keypoints, giou_loss, gcn_gt_loss, pred2d, gt2d
   

    def computeBCESingleFrame(self, preds, gt):
        loss = self.bce(preds, gt.to(self.device))

        return loss

    def box_iou(self, box1, box2):
        # 计算两个框的交集部分的坐标
        inter_x1 = torch.max(box1[0], box2[0])
        inter_y1 = torch.max(box1[1], box2[1])
        inter_x2 = torch.min(box1[2], box2[2])
        inter_y2 = torch.min(box1[3], box2[3])

        # 计算交集面积
        inter_area = torch.clamp(inter_x2 - inter_x1 + 1, min=0) * torch.clamp(inter_y2 - inter_y1 + 1, min=0)

        # 计算两个框的面积
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # 计算并集面积
        union_area = box1_area + box2_area - inter_area

        # 计算IoU
        iou = inter_area / union_area

        return iou, union_area

    def box_giou(self, box1, box2):
        # 计算IoU和并集面积
        iou, union_area = self.box_iou(box1, box2)

        # 计算外接矩形的坐标
        enclose_x1 = torch.min(box1[0], box2[0])
        enclose_y1 = torch.min(box1[1], box2[1])
        enclose_x2 = torch.max(box1[2], box2[2])
        enclose_y2 = torch.max(box1[3], box2[3])

        # 计算外接矩形的面积
        enclose_area = (enclose_x2 - enclose_x1 + 1) * (enclose_y2 - enclose_y1 + 1)

        # 计算GIOU
        giou = 1-(iou - (enclose_area - union_area) / enclose_area)


        return giou.mean()

