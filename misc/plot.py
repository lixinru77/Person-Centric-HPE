
import os
import cv2
import math
import torch
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from preprocessing.vitalSignsDet import VitalSIgnsProcessor

dict_behaviors = {
    0: 'Standing',
    1: 'Sitting',
    2: 'Stepping'
}

# 定义颜色列表，可以根据需要添加更多颜色
colors = [[255, 0, 0],  # 红色
          [0, 255, 0],  # 绿色
          [0, 0, 255],  # 蓝色
          [255, 255, 0],  # 黄色
          [255, 0, 255],  # 紫色
          [0, 255, 255],  # 青色
          [128, 0, 128],  # 紫红色
          [255, 165, 0],  # 橙色
          [0, 128, 0],  # 深绿色
          [0, 0, 128],  # 深蓝色
          ]



def plotPredGTCanvasPaper(
    pred_joints,
    gt_joints,
    imageIds,
    visDir,
    image_root='/mnt/datastore/RADAR-OLD',
    sample_stride=10,
    crop_margin=22,
    canvas_size=256
):
    """
    Minimal paper-style prediction visualization:
    - Prediction only
    - Slim colorful skeleton
    - Small joints
    - Light white outline
    - No GT panel, no text
    - Style closer to common pose-estimation papers / OpenPose demos

    pred_joints: [B, 14, 2], full-image 256-coordinate prediction
    gt_joints:   [B, 14, 2], only used for stable crop box (not drawn)
    imageIds:    [B]
    """

    import os
    import cv2
    import torch
    import numpy as np

    # 14-keypoint skeleton
    joints_edges = [
        (0, 1), (1, 2),              # right leg
        (0, 3), (3, 4), (4, 5),      # left leg
        (0, 6), (3, 6),              # hip to neck
        (6, 7),                      # neck to head
        (6, 8), (8, 9), (9, 10),     # left arm
        (6, 11), (11, 12), (12, 13)  # right arm
    ]

    # 更接近 OpenPose 的轻量彩色风格（BGR）
    edge_colors = [
        (255, 170,   0),   # right leg
        (255, 210,   0),

        (  0, 220, 255),   # pelvis / left leg
        (  0, 255, 255),
        (  0, 255, 190),

        ( 80, 230,  80),   # torso
        (120, 255, 120),
        (180, 255, 100),   # head

        (255,  80, 180),   # left arm
        (255,  60, 255),
        (220,  70, 255),

        (255, 150,  60),   # right arm
        (255, 100, 100),
        (255,  60,  60),
    ]

    joint_colors = [
        (255, 200,   0),   # 0
        (255, 185,   0),   # 1
        (255, 170,   0),   # 2
        (  0, 255, 255),   # 3
        (  0, 255, 225),   # 4
        (  0, 255, 190),   # 5
        (100, 255, 100),   # 6
        (180, 255, 100),   # 7
        (255,  80, 180),   # 8
        (255,  60, 255),   # 9
        (220,  70, 255),   # 10
        (255, 150,  60),   # 11
        (255, 100, 100),   # 12
        (255,  60,  60),   # 13
    ]

    shadow_color = (255, 255, 255)

    def to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def clip_joints(joints, w, h):
        joints = joints.copy()
        joints[:, 0] = np.clip(joints[:, 0], 0, w - 1)
        joints[:, 1] = np.clip(joints[:, 1], 0, h - 1)
        return joints

    def get_crop_box(gt, pred, img_w, img_h, margin=22):
        """
        用 GT + Pred 共同确定稳定 crop 区域，但最终只画 pred。
        如果你以后不想依赖 gt，也可以只用 pred。
        """
        if gt is None:
            pts = pred[:, :2].copy()
        else:
            pts = np.concatenate([gt[:, :2], pred[:, :2]], axis=0)

        pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)

        x1 = int(np.floor(np.min(pts[:, 0]) - margin))
        y1 = int(np.floor(np.min(pts[:, 1]) - margin))
        x2 = int(np.ceil(np.max(pts[:, 0]) + margin))
        y2 = int(np.ceil(np.max(pts[:, 1]) + margin))

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w - 1, x2)
        y2 = min(img_h - 1, y2)

        bw = x2 - x1 + 1
        bh = y2 - y1 + 1
        side = max(bw, bh)

        # 不要太小，避免人被裁得太紧
        side = max(side, 120)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        x1 = cx - side // 2
        y1 = cy - side // 2
        x2 = x1 + side
        y2 = y1 + side

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 >= img_w:
            shift = x2 - img_w + 1
            x1 -= shift
            x2 -= shift
        if y2 >= img_h:
            shift = y2 - img_h + 1
            y1 -= shift
            y2 -= shift

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w - 1, x2)
        y2 = min(img_h - 1, y2)

        return x1, y1, x2, y2

    def crop_and_remap(img, joints, crop_box, out_size):
        x1, y1, x2, y2 = crop_box
        crop = img[y1:y2 + 1, x1:x2 + 1].copy()

        crop_h, crop_w = crop.shape[:2]
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

        out_joints = joints.copy()
        out_joints[:, 0] = (out_joints[:, 0] - x1) * (out_size / max(crop_w, 1))
        out_joints[:, 1] = (out_joints[:, 1] - y1) * (out_size / max(crop_h, 1))
        out_joints = clip_joints(out_joints, out_size, out_size)

        return crop, out_joints

    def draw_skeleton(img, joints):
        out = img.copy()

        # 轻量白色底边
        for a, b in joints_edges:
            p1 = tuple(joints[a].astype(int))
            p2 = tuple(joints[b].astype(int))
            cv2.line(out, p1, p2, shadow_color, 3, lineType=cv2.LINE_AA)

        # 彩色主骨架（细）
        for idx, (a, b) in enumerate(joints_edges):
            p1 = tuple(joints[a].astype(int))
            p2 = tuple(joints[b].astype(int))
            cv2.line(out, p1, p2, edge_colors[idx], 2, lineType=cv2.LINE_AA)

        # 小关节点
        for j, p in enumerate(joints):
            p = tuple(p.astype(int))
            cv2.circle(out, p, 3, shadow_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, p, 2, joint_colors[j], -1, lineType=cv2.LINE_AA)

        return out

    pred_joints = to_numpy(pred_joints)
    gt_joints = to_numpy(gt_joints)
    imageIds = to_numpy(imageIds)

    os.makedirs(visDir, exist_ok=True)

    for b in range(pred_joints.shape[0]):
        image_id = int(imageIds[b])
        namestr = f'{image_id:09d}'
        seq_id = int(namestr[:4])
        frame_id = int(namestr[-4:])

        if frame_id % sample_stride != 0:
            continue

        rgb_path = os.path.join(
            image_root,
            f'rrv_pairs_{seq_id}',
            'frame',
            f'{frame_id:09d}.jpg'
        )

        img = cv2.imread(rgb_path)
        if img is None:
            print(f'[VIS WARNING] image not found: {rgb_path}')
            continue

        if img.shape[0] != 256 or img.shape[1] != 256:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        pred = pred_joints[b].copy()
        pred = clip_joints(pred, 256, 256)

        gt = None
        if gt_joints is not None:
            gt = gt_joints[b].copy()
            gt = clip_joints(gt, 256, 256)

        crop_box = get_crop_box(gt, pred, 256, 256, margin=crop_margin)
        pred_img, pred_crop_joints = crop_and_remap(img, pred, crop_box, canvas_size)

        pred_panel = draw_skeleton(pred_img, pred_crop_joints)

        # 很轻的白边，便于论文排版
        pred_panel = cv2.copyMakeBorder(
            pred_panel,
            top=4,
            bottom=4,
            left=4,
            right=4,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )

        save_dir = os.path.join(visDir, f'rrv_pairs_{seq_id}')
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f'{frame_id:09d}_paper.png')
        cv2.imwrite(save_path, pred_panel)