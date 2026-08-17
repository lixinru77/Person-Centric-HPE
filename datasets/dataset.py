import os
import random
import json
import torch
import numpy as np
from PIL import Image
from random import sample
import torch.nn.functional as F
import torch.utils.data as data
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datasets.base import BaseDataset, generateGTAnnot
import openpyxl
from openpyxl import Workbook

def getDataset(phase, cfg, args, random=True):
    return HuPR3D_horivert(phase, cfg, args, random)

class HuPR3D_horivert(BaseDataset):
    
    def __init__(self, phase, cfg, args, random=True):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        super(HuPR3D_horivert, self).__init__(phase)
        self.duration = cfg.DATASET.duration # 10 FPS * 60 seconds
        self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numChirps = cfg.DATASET.numChirps
        self.r = cfg.DATASET.rangeSize #64
        self.w = cfg.DATASET.azimuthSize #64
        self.h = cfg.DATASET.elevationSize #8
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.sampling_ratio = args.sampling_ratio

        # self.dirRoot = '/mnt/datastore/RVHAD'
        self.dirRoot = '/mnt/datastore/RADAR_NEW'

        self.idxToJoints = cfg.DATASET.idxToJoints
        self.random = random

        self.dt = 0
        self.objNums = 1

        #暂时注释生成标准的coco标签，xinru
        generateGTAnnot(cfg, phase)
        ##########

        self.gtFile = os.path.join(self.dirRoot, '%s_gt.json' % phase)
        self.coco = COCO(self.gtFile)
        self.imageIds = self.coco.getImgIds()
        self.VRDAEPaths_hori = []
        self.VRDAEPaths_vert = []
        for name in self.imageIds:
            namestr = '%09d' % name
            self.VRDAEPaths_hori.append(
                os.path.join(self.dirRoot, 'rrv_pairs_%d/hori/%09d.npy' % (int(namestr[:4]), int(namestr[-4:]))))
            self.VRDAEPaths_vert.append(
                os.path.join(self.dirRoot, 'rrv_pairs_%d/vert/%09d.npy' % (int(namestr[:4]), int(namestr[-4:]))))

            # 模拟延时数据生成
            # self.VRDAEPaths_hori.append(
            #     os.path.join(self.dirRoot, 'rrv_pairs_%d/hori/%09d.npy' % (int(namestr[:4]), int(namestr[-4:]))))
            # if (int(namestr[-4:]) + self.dt) < 600:
            #     self.VRDAEPaths_vert.append(
            #         os.path.join(self.dirRoot, 'rrv_pairs_%d/vert/%09d.npy' % (int(namestr[:4]), int(namestr[-4:]) + self.dt)))
            # else:
            #     self.VRDAEPaths_vert.append(self.VRDAEPaths_vert[-1])

        self.annots = self._load_coco_keypoint_annotations()
        self.transformFunc = self.getTransformFunc(cfg)

    def evaluateEach(self, loadDir, f1_score_avg): # <--- 去掉硬编码的路径
        res_file = os.path.join(loadDir, "%s_results.json"% self.phase)
        anns = json.load(open(res_file))
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        keypoint_list = []
        for i in range(self.numKeypoints):
            coco_eval.evaluate(i)
            coco_eval.accumulate()
            coco_eval.summarize()
            
            info_str = [(name, coco_eval.stats[ind]) for ind, name in enumerate(stats_names)]
        
            keypoint_list.append(info_str[1][1])

        # === 动态写入 TXT 文件的逻辑 ===
        # 自动把 txt 文件保存在你当前运行的模型输出文件夹里 (比如 output/mutli+mnet/)
        txtPath = os.path.join(loadDir, "keypoint_results.txt")
        print(f"\n==========> 准备将14个关键点精度写入: {txtPath} <==========")
        
        # 使用 'a' 模式追加写入
        with open(txtPath, 'a') as f:
            f.write(f"\n========== Evaluation Results for: {loadDir} ==========\n")
            for i in range(self.numKeypoints):
                result_str = '%s: %.3f' % (self.idxToJoints[i], keypoint_list[i])
                print(result_str)  # 打印到控制台让你看见
                f.write(result_str + '\n')  # 写入 TXT 文件
            
            f1_str = f"Behavior F1_score: {float(f1_score_avg):.3f}"
            print(f1_str)
            f.write(f1_str + '\n')
            f.write("========================================================\n")

        return info_str[0][1]  # return the value of AP

    def evaluate(self, loadDir, logger=None):
        res_file = os.path.join(loadDir, "%s_results.json"% self.phase)
        anns = json.load(open(res_file))
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        ###########加入logger日志调用######
        if logger is not None:
            logger.logger.info("=" * 60)
            logger.logger.info("COCO Evaluation Results:")
            for name, value in info_str:
                logger.logger.info(f"{name}: {value:.3f}")
            logger.logger.info("=" * 60)
        # === 新添加结束 ===

        for idx_metric in range(10):
            print("%s:\t%.3f\t"%(info_str[idx_metric][0], info_str[idx_metric][1]), end='')
            if (idx_metric+1) % 5 == 0:
                print()
        return info_str[0][1] # return the value of AP

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.imageIds:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        im_ann = self.coco.loadImgs(index)[0]
        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)
        rec = []
        for obj in objs:
            joints_2d = np.zeros((self.numKeypoints, 2), dtype=np.float64)
            joints_2d_vis = np.zeros((self.numKeypoints, 2), dtype=np.float64)
            for ipt in range(self.numKeypoints):
                joints_2d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_2d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_2d_vis[ipt, 0] = t_vis
                joints_2d_vis[ipt, 1] = t_vis
            rec.append({
                'joints': joints_2d,
                'joints_vis': joints_2d_vis,
                'bbox': obj['bbox'], # x, y, w, h
                'imageId': obj['image_id'],
                'behaviorID': obj['behavior_id']
            })
        return rec

    def __getitem__(self, index):
        if self.random:
            index = index * random.randint(1, self.sampling_ratio)
        else:
            index = index * self.sampling_ratio

        #xinru
        index = min(index, len(self.VRDAEPaths_hori)-1)

        # collect past frames and furture frames for the center target frame
        # TODO:以index为中心，左右各取self.numGroupFrames//2帧，共self.numGroupFrames帧,同时chirps只取8个处理，然后打包，shape-->(8,8,2,64,64,8)
        padSize = index % self.duration
        idx = index - self.numGroupFrames//2 - 1

        VRDAEmaps_hori = torch.zeros((self.numGroupFrames, self.numFrames, 2, self.r, self.w, self.h))
        VRDAEmaps_vert = torch.zeros((self.numGroupFrames, self.numFrames, 2, self.r, self.w, self.h))


        for j in range(self.numGroupFrames):
            # 防止越界
            if (j + padSize) <= self.numGroupFrames//2:
                idx = index - padSize
            elif j > (self.duration - 1 - padSize) + self.numGroupFrames//2:
                idx = index + (self.duration - 1 - padSize)
            else:
                idx += 1

            ####xinru
            idx = max(0, min(idx, len(self.VRDAEPaths_hori) -1))

            VRDAEPath_hori = self.VRDAEPaths_hori[idx]
            VRDAEPath_vert = self.VRDAEPaths_vert[idx]

            VRDAERealImag_hori = np.load(VRDAEPath_hori)
            VRDAERealImag_vert = np.load(VRDAEPath_vert)

            idxSampleChirps = 0

            for idxChirps in range(self.numChirps//2 - self.numFrames//2, self.numChirps//2 + self.numFrames//2):
                VRDAEmaps_hori[j, idxSampleChirps, 0, :, :, :] = self.transformFunc(VRDAERealImag_hori[idxChirps].real).permute(1, 2, 0)
                VRDAEmaps_hori[j, idxSampleChirps, 1, :, :, :] = self.transformFunc(VRDAERealImag_hori[idxChirps].imag).permute(1, 2, 0)
                VRDAEmaps_vert[j, idxSampleChirps, 0, :, :, :] = self.transformFunc(VRDAERealImag_vert[idxChirps].real).permute(1, 2, 0)
                VRDAEmaps_vert[j, idxSampleChirps, 1, :, :, :] = self.transformFunc(VRDAERealImag_vert[idxChirps].imag).permute(1, 2, 0)
                idxSampleChirps += 1


        # 1. 找到当前 index 对应的 imageId（帧ID）
        # current_image_id = self.annots[index]['imageId']
        current_image_id = self.imageIds[index]
        # 2. 从所有标签中，过滤出属于同一帧（同一个 imageId）的所有人
        current_frame_annots = [annot for annot in self.annots if annot['imageId'] == current_image_id]
        # 3. 设定最大人数（根据你的数据集修改，比如 2）
        # 这样可以保证每次返回的 Tensor 形状固定，避免 DataLoader 组装 Batch 时报错
        MAX_PERSONS = 3
        # 初始化固定形状的空 Tensor
        joints_padded = torch.zeros((MAX_PERSONS, self.numKeypoints, 2), dtype=torch.float32)
        bbox_padded = torch.zeros((MAX_PERSONS, 4), dtype=torch.float32)
        behavior_padded = torch.zeros((MAX_PERSONS,), dtype=torch.long)
        # 记录这一帧实际上有几个人（非常重要，后面算 Loss 时要用它来过滤假人）
        actual_num_persons = min(len(current_frame_annots), MAX_PERSONS)
        # 4. 把真实的人的数据填进去
        for i in range(actual_num_persons):
            annot = current_frame_annots[i]
            # 注意：强烈建议 joints 也用 FloatTensor，因为坐标回归通常是浮点数
            joints_padded[i] = torch.FloatTensor(annot['joints'])
            bbox_padded[i] = torch.FloatTensor(annot['bbox'])
            behavior_padded[i] = annot['behaviorID']

        return {'VRDAEmap_hori': VRDAEmaps_hori,
                'VRDAEmap_vert': VRDAEmaps_vert,
                'jointsGroup': joints_padded,
                'bbox': bbox_padded,
                'behavior': behavior_padded,
                'num_persons': actual_num_persons,
                'imageId': current_image_id
                }
    
    def __len__(self):
        return len(self.VRDAEPaths_hori)//self.sampling_ratio
