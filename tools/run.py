import os
import time
import torch
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F

from models import HuPRNet
from datasets import getDataset
from tools.base import BaseRunner
from sklearn.metrics import f1_score

from misc.plot import plotPredGTCanvas
from misc.plot import plotPredGTCanvasPaper


# ============================================================
# CCTD: bbox-guided radar crop and resize
# ============================================================
def crop_and_resize_radar(radar_map, bbox, original_img_size=256, target_size=64):
    """
    根据 bbox 裁剪雷达信号，并 resize 回 64×64。

    radar_map: GPU tensor, shape usually:
        [numGroupFrames, 2, numFrames, range, azimuth, elevation]
    bbox: CPU numpy array, [x, y, w, h] in 256×256 image coordinate.
    """
    scale = target_size / original_img_size

    x = int(bbox[0] * scale)
    y = int(bbox[1] * scale)
    w = int(bbox[2] * scale)
    h = int(bbox[3] * scale)

    # Boundary protection
    x = max(0, min(x, target_size - 1))
    y = max(0, min(y, target_size - 1))
    w = max(1, min(w, target_size - x))
    h = max(1, min(h, target_size - y))

    # Crop spatial region: range/azimuth dimensions
    cropped_map = radar_map[:, :, :, y:y + h, x:x + w, :]

    # Move elevation before spatial dims, then resize spatial dims
    permuted_map = cropped_map.permute(0, 1, 2, 5, 3, 4).contiguous()
    original_shape = permuted_map.shape

    flattened = permuted_map.reshape(-1, 1, h, w)

    resized = F.interpolate(
        flattened,
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    )

    final_map = resized.view(*original_shape[:-2], target_size, target_size)
    final_map = final_map.permute(0, 1, 2, 4, 5, 3).contiguous()

    return final_map


class Runner(BaseRunner):
    def __init__(self, args, cfg):
        super(Runner, self).__init__(args, cfg)

        if not args.eval:
            self.trainSet = getDataset('train', cfg, args)
            self.trainLoader = data.DataLoader(
                self.trainSet,
                self.cfg.TRAINING.batchSize,
                shuffle=True,
                num_workers=cfg.SETUP.numWorkers
            )
        else:
            self.trainLoader = [0]

        self.testSet = getDataset('test' if args.eval else 'val', cfg, args)
        self.testLoader = data.DataLoader(
            self.testSet,
            self.cfg.TEST.batchSize,
            shuffle=False,
            num_workers=cfg.SETUP.numWorkers
        )

        self.model = HuPRNet(self.cfg).to(self.device)
        print(self.model)

        self.stepSize = len(self.trainLoader) * self.cfg.TRAINING.warmupEpoch
        LR = self.cfg.TRAINING.lr if self.cfg.TRAINING.warmupEpoch == -1 else \
            self.cfg.TRAINING.lr / (self.cfg.TRAINING.warmupGrowth ** self.stepSize)

        self.initialize(LR)
        self.beta = 0.0

    def load_medium_weight(self):
        checkpoint = os.path.join(self.dir, '%s.pth' % "model_best")
        print('==========>Loading the medium model weight from %s' % checkpoint)

        checkpoint = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if self.cfg.TRAINING.freeze_epoch == 0:
            self.start_epoch = 0
            print('==========>微调模式：强行重置起始轮数为 0')
        else:
            self.start_epoch = checkpoint['epoch']
            print('==========>续训模式：从存档进度 Epoch %d 开始' % checkpoint['epoch'])

    # ============================================================
    # Eval with runtime profiling
    # ============================================================
    def eval(self, visualization=True, epoch=-1):
        self.model.eval()

        # 只在单独 --eval 时统计 runtime。
        # 训练过程中每个 epoch 末尾调用 eval(epoch>=0) 时不统计，避免拖慢训练。
        profile_runtime = (epoch == -1)
        warmup_batches = 5

        cctd_time_total = 0.0
        model_time_total = 0.0
        pipeline_time_total = 0.0

        profile_batches = 0
        profile_frames = 0
        profile_rois = 0

        if profile_runtime and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()

        f1_behavior_list = []
        self.logger.clear(len(self.testLoader.dataset))
        savePreds = []

        with torch.no_grad():
            for idx, batch in enumerate(self.testLoader):
                VRDAEmaps_hori_full = batch['VRDAEmap_hori'].to(self.device).float()
                VRDAEmaps_vert_full = batch['VRDAEmap_vert'].to(self.device).float()

                keypoints_all = batch['jointsGroup'].to(self.device).float()
                imageIds_batch = batch['imageId']

                # behavior 是可选的：如果没有行为分类需求，不会影响姿态评估
                has_behavior = ('behavior' in batch) and hasattr(self.model, 'BehaviorClassifier')
                if has_behavior:
                    behaviors_all = batch['behavior'].to(self.device).long()
                else:
                    behaviors_all = None

                bboxes_all_cpu = batch['bbox'].numpy()
                num_persons_batch_cpu = batch['num_persons'].numpy()

                batch_size = VRDAEmaps_hori_full.size(0)

                list_hori, list_vert, list_keypoints = [], [], []
                list_gt_original = []
                list_imageIds_aligned, list_bboxes_aligned = [], []
                list_behaviors = []

                # ---------------- CCTD timing starts ----------------
                if profile_runtime and torch.cuda.is_available():
                    torch.cuda.synchronize()
                pipeline_start = time.perf_counter()
                cctd_start = time.perf_counter()

                # 1. CCTD crop + local coordinate transform
                for b in range(batch_size):
                    num_persons = int(num_persons_batch_cpu[b])

                    for p in range(num_persons):
                        person_bbox_cpu = bboxes_all_cpu[b, p]

                        # crop radar ROI
                        list_hori.append(
                            crop_and_resize_radar(VRDAEmaps_hori_full[b], person_bbox_cpu)
                        )
                        list_vert.append(
                            crop_and_resize_radar(VRDAEmaps_vert_full[b], person_bbox_cpu)
                        )

                        if has_behavior:
                            list_behaviors.append(behaviors_all[b, p])

                        # GT original keypoints for visualization
                        original_keypoints = keypoints_all[b, p]
                        list_gt_original.append(original_keypoints.clone())

                        # map GT keypoints to bbox-local 256×256 coordinate
                        x, y, w, h = person_bbox_cpu[0], person_bbox_cpu[1], person_bbox_cpu[2], person_bbox_cpu[3]
                        w_safe = max(float(w), 1.0)
                        h_safe = max(float(h), 1.0)

                        transformed_keypoints = original_keypoints.clone()
                        transformed_keypoints[:, 0] = (transformed_keypoints[:, 0] - float(x)) * (256.0 / w_safe)
                        transformed_keypoints[:, 1] = (transformed_keypoints[:, 1] - float(y)) * (256.0 / h_safe)
                        list_keypoints.append(transformed_keypoints)

                        list_imageIds_aligned.append(imageIds_batch[b])
                        list_bboxes_aligned.append(torch.tensor(person_bbox_cpu, dtype=torch.float32))

                if len(list_hori) == 0:
                    continue

                # 2. Stack CCTD outputs
                batch_hori_input = torch.stack(list_hori, dim=0)
                batch_vert_input = torch.stack(list_vert, dim=0)
                batch_keypoints_input = torch.stack(list_keypoints, dim=0)

                batch_imageIds_input = torch.stack(list_imageIds_aligned).to(self.device)
                batch_bboxes_input = torch.stack(list_bboxes_aligned, dim=0).to(self.device)
                batch_gt_original_input = torch.stack(list_gt_original, dim=0)

                if has_behavior:
                    batch_behaviors_input = torch.stack(list_behaviors, dim=0)
                else:
                    batch_behaviors_input = None

                if profile_runtime and torch.cuda.is_available():
                    torch.cuda.synchronize()
                cctd_end = time.perf_counter()
                # ---------------- CCTD timing ends ----------------

                # ---------------- Model forward timing starts ----------------
                if profile_runtime and torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_start = time.perf_counter()

                heatmap_pred, gcn_heatmap = self.model(batch_hori_input, batch_vert_input)

                if profile_runtime and torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_end = time.perf_counter()
                # ---------------- Model forward timing ends ----------------

                pipeline_end = model_end

                # Runtime accumulation after warmup
                if profile_runtime and idx >= warmup_batches:
                    num_rois = batch_hori_input.size(0)

                    cctd_time_total += (cctd_end - cctd_start)
                    model_time_total += (model_end - model_start)
                    pipeline_time_total += (pipeline_end - pipeline_start)

                    profile_batches += 1
                    profile_frames += batch_size
                    profile_rois += num_rois

                preds_keypoints = (heatmap_pred, gcn_heatmap)

                # behavior classifier is not included in model forward timing
                if has_behavior:
                    preds_behaviors = self.model.BehaviorClassifier(gcn_heatmap)
                else:
                    preds_behaviors = None

                total_loss, giou_loss, gcn_gt_loss, preds_keypoints, gts_keypoints = \
                    self.lossComputer.computeLoss(preds_keypoints, batch_keypoints_input)

                if has_behavior and (epoch == -1 or epoch >= self.cfg.TRAINING.freeze_epoch):
                    behavior_loss = self.FocalLoss(preds_behaviors, batch_behaviors_input)
                    f1_avg_behavior = self.f1_accuracy(preds_behaviors, batch_behaviors_input)
                    f1_behavior_list.append(f1_avg_behavior)
                else:
                    behavior_loss = torch.tensor(float('inf'))
                    f1_avg_behavior = torch.zeros(1,)

                self.logger.display(
                    total_loss,
                    gcn_gt_loss,
                    giou_loss,
                    behavior_loss,
                    f1_avg_behavior,
                    len(batch_hori_input),
                    epoch
                )

                # 3. Restore local prediction to global 256×256 image coordinate
                final_preds = preds_keypoints.copy() * self.imgHeatmapRatio

                for i in range(len(batch_bboxes_input)):
                    bbox = batch_bboxes_input[i]
                    x = bbox[0].item()
                    y = bbox[1].item()
                    w = max(bbox[2].item(), 1.0)
                    h = max(bbox[3].item(), 1.0)

                    final_preds[i, :, 0] = final_preds[i, :, 0] * (w / 256.0) + x
                    final_preds[i, :, 1] = final_preds[i, :, 1] * (h / 256.0) + y

                # Optional visualization
                # if visualization:
                #     plotPredGTCanvasPaper(
                #         pred_joints=final_preds,
                #         gt_joints=batch_gt_original_input,
                #         imageIds=batch_imageIds_input,
                #         visDir=self.visDir,
                #         image_root='/mnt/datastore/RADAR-DATA',
                #         sample_stride=10
                #     )

                self.saveKeypoints(
                    savePreds,
                    final_preds,
                    batch_bboxes_input.cpu().numpy(),
                    batch_imageIds_input.cpu().numpy()
                )

        # behavior F1 summary
        if len(f1_behavior_list) > 0:
            avg_tensor = sum(f1_behavior_list) / len(f1_behavior_list)
            f1_score_avg = avg_tensor.item() if isinstance(avg_tensor, torch.Tensor) else avg_tensor
        else:
            f1_score_avg = 0.0

        self.writeKeypoints(savePreds)

        if self.args.keypoints:
            self.testSet.evaluateEach(self.dir, f1_score_avg)

        accAP = self.testSet.evaluate(self.dir, logger=self.logger)
        print(f"Finally average F1_score is {f1_score_avg}")

        # ============================================================
        # Runtime profiling summary
        # ============================================================
        if profile_runtime and profile_batches > 0:
            model_fps_roi = profile_rois / model_time_total
            cctd_fps_frame = profile_frames / cctd_time_total
            pipeline_fps_frame = profile_frames / pipeline_time_total

            avg_cctd_ms_per_frame = cctd_time_total / profile_frames * 1000.0
            avg_cctd_ms_per_roi = cctd_time_total / profile_rois * 1000.0

            avg_model_ms_per_roi = model_time_total / profile_rois * 1000.0
            avg_pipeline_ms_per_frame = pipeline_time_total / profile_frames * 1000.0

            if torch.cuda.is_available():
                peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                peak_reserved_gb = torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
            else:
                peak_mem_gb = 0.0
                peak_reserved_gb = 0.0

            print("\n========== Runtime Profiling ==========")
            print(f"Profiled batches: {profile_batches}")
            print(f"Profiled frames: {profile_frames}")
            print(f"Profiled person ROIs: {profile_rois}")
            print(f"CCTD crop+resize time: {avg_cctd_ms_per_frame:.3f} ms/frame")
            print(f"CCTD crop+resize time: {avg_cctd_ms_per_roi:.3f} ms/ROI")
            print(f"Model forward time: {avg_model_ms_per_roi:.3f} ms/ROI")
            print(f"Model inference FPS: {model_fps_roi:.2f} ROI/s")
            print(f"Complete CCTD+model pipeline: {avg_pipeline_ms_per_frame:.3f} ms/frame")
            print(f"Complete CCTD+model FPS: {pipeline_fps_frame:.2f} frame/s")
            print(f"Peak GPU memory allocated: {peak_mem_gb:.3f} GB")
            print(f"Peak GPU memory reserved: {peak_reserved_gb:.3f} GB")
            print("=======================================\n")

            profile_path = os.path.join(self.dir, "runtime_profile.txt")
            with open(profile_path, "w") as f:
                f.write("========== Runtime Profiling ==========\n")
                f.write(f"Profiled batches: {profile_batches}\n")
                f.write(f"Profiled frames: {profile_frames}\n")
                f.write(f"Profiled person ROIs: {profile_rois}\n")
                f.write(f"CCTD crop+resize time: {avg_cctd_ms_per_frame:.3f} ms/frame\n")
                f.write(f"CCTD crop+resize time: {avg_cctd_ms_per_roi:.3f} ms/ROI\n")
                f.write(f"Model forward time: {avg_model_ms_per_roi:.3f} ms/ROI\n")
                f.write(f"Model inference FPS: {model_fps_roi:.2f} ROI/s\n")
                f.write(f"Complete CCTD+model pipeline: {avg_pipeline_ms_per_frame:.3f} ms/frame\n")
                f.write(f"Complete CCTD+model FPS: {pipeline_fps_frame:.2f} frame/s\n")
                f.write(f"Peak GPU memory allocated: {peak_mem_gb:.3f} GB\n")
                f.write(f"Peak GPU memory reserved: {peak_reserved_gb:.3f} GB\n")
                f.write("=======================================\n")

            print(f"Runtime profile saved to: {profile_path}")

        return accAP, f1_score_avg

    # ============================================================
    # Train: no runtime profiling here
    # ============================================================
    def train(self):
        self.stepSize = len(self.trainLoader) * self.cfg.TRAINING.warmupEpoch
        LR = self.cfg.TRAINING.lr if self.cfg.TRAINING.warmupEpoch == -1 else \
            self.cfg.TRAINING.lr / (self.cfg.TRAINING.warmupGrowth ** self.stepSize)

        self.beta = 0.0

        for epoch in range(self.start_epoch, self.cfg.TRAINING.epochs):
            if epoch == self.cfg.TRAINING.freeze_epoch:
                self.load_medium_weight()
                self.freeze_grad_pose(self.model, False)
                self.freeze_grad_behavior(self.model, True)
                self.initialize(LR)

            if epoch >= self.cfg.TRAINING.freeze_epoch:
                self.model.eval()
                if hasattr(self.model, 'BehaviorClassifier'):
                    self.model.BehaviorClassifier.train()
            else:
                self.model.train()

            loss_list = []
            self.logger.clear(len(self.trainLoader.dataset))

            for idxBatch, batch in enumerate(self.trainLoader):
                self.optimizer.zero_grad()

                VRDAEmaps_hori_full = batch['VRDAEmap_hori'].to(self.device).float()
                VRDAEmaps_vert_full = batch['VRDAEmap_vert'].to(self.device).float()
                keypoints_all = batch['jointsGroup'].to(self.device).float()

                has_behavior = ('behavior' in batch) and hasattr(self.model, 'BehaviorClassifier')
                if has_behavior:
                    behaviors_all = batch['behavior'].to(self.device).long()
                else:
                    behaviors_all = None

                bboxes_all_cpu = batch['bbox'].numpy()
                num_persons_batch_cpu = batch['num_persons'].numpy()

                batch_size = VRDAEmaps_hori_full.size(0)

                list_hori, list_vert, list_keypoints = [], [], []
                list_behaviors = []

                for b in range(batch_size):
                    num_persons = int(num_persons_batch_cpu[b])

                    for p in range(num_persons):
                        person_bbox_cpu = bboxes_all_cpu[b, p]

                        list_hori.append(
                            crop_and_resize_radar(VRDAEmaps_hori_full[b], person_bbox_cpu)
                        )
                        list_vert.append(
                            crop_and_resize_radar(VRDAEmaps_vert_full[b], person_bbox_cpu)
                        )

                        if has_behavior:
                            list_behaviors.append(behaviors_all[b, p])

                        original_keypoints = keypoints_all[b, p]
                        x, y, w, h = person_bbox_cpu[0], person_bbox_cpu[1], person_bbox_cpu[2], person_bbox_cpu[3]

                        w_safe = max(float(w), 1.0)
                        h_safe = max(float(h), 1.0)

                        transformed_keypoints = original_keypoints.clone()
                        transformed_keypoints[:, 0] = (transformed_keypoints[:, 0] - float(x)) * (256.0 / w_safe)
                        transformed_keypoints[:, 1] = (transformed_keypoints[:, 1] - float(y)) * (256.0 / h_safe)

                        list_keypoints.append(transformed_keypoints)

                if len(list_hori) == 0:
                    continue

                batch_hori_input = torch.stack(list_hori, dim=0)
                batch_vert_input = torch.stack(list_vert, dim=0)
                batch_keypoints_input = torch.stack(list_keypoints, dim=0)

                if has_behavior:
                    batch_behaviors_input = torch.stack(list_behaviors, dim=0)
                else:
                    batch_behaviors_input = None

                heatmap_pred, gcn_heatmap = self.model(batch_hori_input, batch_vert_input)
                preds_keypoints = (heatmap_pred, gcn_heatmap)

                if has_behavior:
                    preds_behaviors = self.model.BehaviorClassifier(gcn_heatmap)
                else:
                    preds_behaviors = None

                total_loss, giou_loss, gcn_gt_loss, _, _ = \
                    self.lossComputer.computeLoss(preds_keypoints, batch_keypoints_input)

                if has_behavior and epoch >= self.cfg.TRAINING.freeze_epoch:
                    behavior_loss = self.FocalLoss(preds_behaviors, batch_behaviors_input)
                    f1_avg_behavior = self.f1_accuracy(preds_behaviors, batch_behaviors_input)

                    total_loss = total_loss + behavior_loss
                    behavior_loss.backward()
                else:
                    behavior_loss = torch.tensor(float('inf'))
                    f1_avg_behavior = torch.zeros(1,)

                    total_loss.backward()

                self.optimizer.step()

                self.logger.display(
                    total_loss,
                    gcn_gt_loss,
                    giou_loss,
                    behavior_loss,
                    f1_avg_behavior,
                    len(batch_hori_input),
                    epoch
                )

                if idxBatch % self.cfg.TRAINING.lrDecayIter == 0:
                    self.adjustLR(epoch)

                loss_list.append(total_loss.item())

            accAP, f1_score_avg = self.eval(visualization=False, epoch=epoch)

            if epoch >= self.cfg.TRAINING.freeze_epoch and hasattr(self.model, 'BehaviorClassifier'):
                self.saveModelWeight(epoch, f1_score_avg, f1_score_avg=f1_score_avg)
            else:
                self.saveModelWeight(epoch, accAP, f1_score_avg=f1_score_avg)

            self.saveLosslist(epoch, loss_list, 'train')

    # ============================================================
    # Behavior classification utilities
    # ============================================================
    def f1_accuracy(self, pred, target):
        pred = torch.argmax(pred, dim=1)
        f1 = f1_score(
            pred.cpu().detach().numpy(),
            target.cpu().detach().numpy(),
            average='weighted'
        )
        return torch.tensor(f1)

    def f1_score_cm(self, cm):
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        f1 = 2 * (precision * recall) / (precision + recall)
        overall_f1 = np.mean(f1)
        return overall_f1

    def freeze_grad_behavior(self, model, requires_grad):
        [
            setattr(param, 'requires_grad', requires_grad)
            for name, param in model.named_parameters()
            if 'BehaviorClassifier' in name
        ]

    def freeze_grad_pose(self, model, requires_grad):
        [
            setattr(param, 'requires_grad', requires_grad)
            for name, param in model.named_parameters()
            if 'BehaviorClassifier' not in name
        ]