import os

import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime


class Logger():
    def __init__(self, log_name="mutli+mnet+behavior"):
        self.bestAP = -1
        self.best_f1_score_avg = 0
        self.progressBar = None
        np.set_printoptions(precision=3)

        # xinru
        log_dir = "./logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 按照时间戳来添加文件路径
        timestamp = datetime.now().strftime("%m%d")
        self.log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")

        self.logger = logging.getLogger('TrainingLogger')  # 创建一个logger
        self.logger.setLevel(logging.INFO)  # 只记录重要的信息

        # 清除已有的handler，避免重复
        if self.logger.handlers:
            self.logger.handlers.clear()

        # 创建文件handler（使用'w'模式，不是追加）
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 设置日志格式
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)  # 添加文件记录方式

        self.logger.info("=" * 50)
        self.logger.info(f"Training started at {timestamp}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info("=" * 50)
        # 设置日志级别
        # logging.basicConfig(filename=self.log_file, level=logging.INFO, format=log_format)

    def clear(self, loaderSize):
        self.progressBar = tqdm(total=loaderSize)

    # xinru loss只显示到进度条但不记录
    def display(self, total_loss, gcn_gt_loss, giou_loss, behavior_loss, acc_avg_behavior, updateSize, epoch):
        if gcn_gt_loss is not None:
            self.progressBar.set_postfix(EP=epoch, Total_Loss=total_loss.item(),
                                         gcn_gt_loss=gcn_gt_loss.item(), giou_loss=giou_loss.item(),
                                         behavior_loss=behavior_loss.item(), f1_avg_behavior=acc_avg_behavior.item(), )
        else:
            self.progressBar.set_postfix(EP=epoch, Total_Loss=total_loss.item(), behavior_loss=behavior_loss.item())
        self.progressBar.update(updateSize)

        # 将日志信息保存到文本文件中
        # if gcn_gt_loss is not None:
        #     self.logger.info(
        #     f"Epoch: {epoch}, Total Loss: {total_loss.item()},")
        #     f" Keypoints Loss: {gcn_gt_loss.item()}, "
        #     f"gIOU Loss: {giou_loss.item()}, "
        #     f"Behavior Loss: {behavior_loss.item()}, "
        #     f"Average Behavior Accuracy: {acc_avg_behavior.item()}")
        # else:
        #     self.logger.info(
        #         f"Epoch: {epoch}, Total Loss: {total_loss.item():.4f}, "
        #         f"Behavior Loss: {behavior_loss.item():.4f}")

    def showBestAP(self):
        return self.bestAP

    def showBestF1Score(self):
        return self.best_f1_score_avg

    def isBestAccAP(self, acc):
        if acc > self.bestAP or self.bestAP == -1:
            self.bestAP = acc
            self.logger.info("Best AP: {:.4f}".format(self.bestAP))
            return True
        else:
            self.logger.info(f"Current AP: {acc:.4f} (Best AP: {self.bestAP:.4f})")
            return False

    def isBestF1Score(self, acc):
        if acc > self.best_f1_score_avg:
            self.best_f1_score_avg = acc
            self.logger.info("Best f1_score_avg: {:.4f}".format(self.best_f1_score_avg))
            return True
        else:
            self.logger.info(f"Current F1 Score: {acc:.4f} (Best F1: {self.best_f1_score_avg:.4f})")
            return False

    def close(self):
        """训练结束时调用"""
        if self.progressBar:
            self.progressBar.close()
        self.logger.info("=" * 50)
        self.logger.info(f"Training completed. Best AP: {self.bestAP:.4f}, Best F1: {self.best_f1_score_avg:.4f}")
        self.logger.info("=" * 50)