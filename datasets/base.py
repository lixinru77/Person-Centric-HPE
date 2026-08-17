import os
import json
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy', '.txt']

class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, radarData):
        c = radarData.size(0)
        minValues = torch.min(radarData.view(c, -1), 1)[0].view(c, 1, 1)
        radarDataZero = radarData - minValues
        maxValues = torch.max(radarDataZero.view(c, -1), 1)[0].view(c, 1, 1)
        radarDataNorm = radarDataZero / (maxValues + 1e-8)
        std, mean = torch.std_mean(radarDataNorm.view(c, -1), 1)
        return (radarDataNorm - mean.view(c, 1, 1)) / (std.view(c, 1, 1) + 1e-8)


def generateGTAnnot(cfg, phase='train'):
        annot = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        annot["info"] = {
            "description": "RVHAD dataset",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "Jiang.Z.Y.",
            "date_created": "2024/11/05"
        }
        
        annot["categories"] = [{
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [
                "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee",
                "L_Ankle", "Neck", "Head", "L_Shoulder", "L_Elbow",
                "L_Wrist", "R_Shoulder", "R_Elbow", "R_Wrist"
            ],
            "skeleton": [
                [14, 13], [13, 12], [11, 10], [10, 9], [9, 7], [12, 9], [8, 7], [7, 1], [7, 4], [6, 5], [5, 4], [3, 2], [2, 1]
            ],
            "behavior": {
                0: "Standing",
                1: "Sitting",
                2: "Falling"
            },
        }]
        
        group_idx = eval('cfg.DATASET.' + phase + 'Name')
        print(f"Processing sequences for {phase}: {group_idx}")

        with open(os.path.join(cfg.DATASET.dataDir, 'hrnet_annot_%s.json' % phase)) as fp:
            annot_files = json.load(fp)
            num_sequences = min(len(annot_files), len(group_idx))
            
            annot_id = 0  # 【防错核心 1】全局唯一的标注ID计数器
            
            for i in range(num_sequences):
                current_seq_id = group_idx[i]
                for block in annot_files[i]:
                    image_id = int(block['image'][:-4]) + current_seq_id * 100000
                    
                    # 读取原始数据
                    joints_multi = np.array(block["joints"])
                    bbox_multi = np.array(block["boxes"])
                    beh_raw = block.get("behavior", 0) 
                    
                    # 【防错核心 2】兼容单人和多人的维度，统一升维到 (N人, ...)
                    if len(joints_multi.shape) == 2:
                        joints_multi = np.expand_dims(joints_multi, axis=0)
                        bbox_multi = np.expand_dims(bbox_multi, axis=0)
                        beh_multi = [beh_raw] # 单人情况，转成列表
                    else:
                        # 如果本来就是多人，判断 behavior 是一整个列表还是单个值
                        beh_multi = beh_raw if isinstance(beh_raw, list) else [beh_raw] * joints_multi.shape[0]

                    num_persons = joints_multi.shape[0]  # 获取这张图里有几个人

                    # 【防错核心 3】遍历这张图里的每一个人，单独生成标注
                    for p in range(num_persons):
                        joints = joints_multi[p]
                        bbox = bbox_multi[p]
                        behavior_id = int(beh_multi[p])
                        
                        if behavior_id == 3:
                            behavior_id -= 1
                            
                        visIdx = np.ones((14, 1)) + 1.0
                        joints_concat = np.concatenate((joints, visIdx), axis=1).reshape(-1).tolist()
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / 2
                        
                        annot["annotations"].append({
                            "num_keypoints": 14,
                            "area": float(area),
                            "iscrowd": 0,
                            "keypoints": joints_concat,
                            "image_id": image_id,
                            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])],
                            "category_id": 1,
                            "id": annot_id,  # <--- 使用全局唯一标注ID
                            "behavior_id": behavior_id,  # <--- 写入动作标签
                        })
                        annot_id += 1  # 每写完一个人，ID加1，绝不重复
                    
                    # 图片信息只需要存一次（不管图里有几个人）
                    annot["images"].append({
                        "license": -1,
                        "file_name": block['image'],
                        "coco_url": "None",
                        "height": 256,
                        "width": 256,
                        "date_captured": "None",
                        "flickr_url": "None",
                        "id": image_id
                    })
                    
        with open(os.path.join(cfg.DATASET.dataDir, '%s_gt.json' % phase), 'w') as fp:
            json.dump(annot, fp)


class BaseDataset(data.Dataset):
    def __init__(self, phase):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        super(BaseDataset, self).__init__()
        self.phase = phase

    def getTransformFunc(self, cfg):
        if self.phase == 'train':
            transformFunc = transforms.Compose([
                transforms.ToTensor(),
                Normalize()
            ])
        else:
            transformFunc = transforms.Compose([
                transforms.ToTensor(),
                Normalize()
            ])
        return transformFunc
    
    def isImageFile(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def getPaths(self, dataDirGroup, dirGroup, mode, frameGroup):
        num = len(dataDirGroup)
        images = []
        for i in range(num):
            for dirName in dirGroup[i]:
                for frame in frameGroup:
                    path = os.path.join(dataDirGroup[i], dirName, mode, frame + '.npy')
                    images.append(path)
        return images

    def getAnnots(self, dataDirGroup, dirGroup, mode, fileName):
        num = len(dataDirGroup)
        annots = []
        for i in range(num):
            for dirName in dirGroup[i]:
                path = os.path.join(dataDirGroup[i], dirName, mode, fileName)
                with open(path, 'r') as fp:
                    annot = json.load(fp)
                annots.extend(annot)
        return annots

    def __getitem__(self, idx):
        raise NotImplementedError('Subclass of BaseDataset must implement __getitem__')

    def __len__(self):
        raise NotImplementedError('Subclass of BaseDataset must implement __len__')