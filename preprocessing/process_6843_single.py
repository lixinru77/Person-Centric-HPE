import os
import glob
import csv
import json
import time
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from plot_utils import PlotMaps, PlotHeatmaps


# ============================================================
# 运行开关：像原始 1843 脚本一样，直接在这里选择功能
# ============================================================
# VISUALIZATION = False   # False: 生成 hori/vert .npy；True: 读取 .npy 并生成可视化图片
VISUALIZATION = True 
USE_OFFSET = False      # 默认不用 offset。若发现雷达/视频明显错位，再改成 True
CLAMP_RADAR_INDEX = True  
# CLAMP_RADAR_INDEX = False  


class RadarObject:
    def __init__(self):
        numGroup = 276

        # ================= 路径配置 =================
        self.base_path = '/mnt/datastore/'
        self.root = 'RADAR_NEW'
        self.saveRoot = 'RADAR_NEW'

        self.radarDataFileNameGroup = []
        self.saveDirNameGroup = []
        self.rgbFileNameGroup = []
        self.jointsFileNameGroup = []

        # ================= 6843 雷达配置 =================
        self.numADCSamples = 256
        self.adcRatio = 4
        self.numAngleBins = self.numADCSamples // self.adcRatio  # 64
        self.numEleBins = 8
        self.numRX = 4
        self.numLanes = 2
        self.framePerSecond = 10
        self.duration = 30
        self.numFrame = self.framePerSecond * self.duration

        # 6843: 3 TX，每帧 255 组 TDM chirp
        self.numChirp = 255 * 3
        self.idxProcChirp = 255
        self.numGroupChirp = 4

        # ================= 单人数据集配置 =================
        self.numKeypoints = 14
        self.objNums = 1  # 关键：这里必须是单人，不按多人画图/组织标签

        self.xIndices = [-45, -30, -15, 0, 15, 30, 45]
        self.yIndices = [i * 10 for i in range(10)]

        # 处理 rrv_pairs_7 和 rrv_pairs_8；range 右边界不包含 9
        self.start_id = 1
        self.end_id = 36

        self.initialize(numGroup)

    def initialize(self, numGroup):
        for i in range(self.start_id, self.end_id):
            pair_path = os.path.join(self.base_path, self.root, f'rrv_pairs_{i}')

            radarDataFileName = [
                os.path.join(pair_path, 'raw_radar', 'hori'),
                os.path.join(pair_path, 'raw_radar', 'vert')
            ]

            saveDirName = os.path.join(self.base_path, self.saveRoot, f'rrv_pairs_{i}')
            rgbFileName = os.path.join(saveDirName, 'frame')
            jointsFileName = os.path.join(saveDirName, 'annot', 'hrnet_annot.json')

            self.radarDataFileNameGroup.append(radarDataFileName)
            self.saveDirNameGroup.append(saveDirName)
            self.rgbFileNameGroup.append(rgbFileName)
            self.jointsFileNameGroup.append(jointsFileName)

    def postProcessFFT3D(self, dataFFT):
        dataFFT = np.fft.fftshift(dataFFT, axes=(0, 1,))
        dataFFT = np.transpose(dataFFT, (2, 0, 1))
        dataFFT = np.flip(dataFFT, axis=(1, 2))
        return dataFFT

    def getadcDataFromDCA1000(self, fileName):
        adcDataName = sorted(os.listdir(fileName))
        if len(adcDataName) == 0:
            raise FileNotFoundError(f'No raw radar file found in: {fileName}')

        adcData = np.fromfile(os.path.join(fileName, adcDataName[0]), dtype=np.int16)
        fileSize = adcData.shape[0]

        adcData = adcData.reshape(-1, self.numLanes * 2).transpose()
        fileSize = int(fileSize / 2)
        LVDS = np.zeros((2, fileSize))

        temp = np.empty((adcData[0].size + adcData[1].size), dtype=adcData[0].dtype)
        temp[0::2] = adcData[0]
        temp[1::2] = adcData[1]
        LVDS[0] = temp

        temp = np.empty((adcData[2].size + adcData[3].size), dtype=adcData[2].dtype)
        temp[0::2] = adcData[2]
        temp[1::2] = adcData[3]
        LVDS[1] = temp

        adcData = np.zeros((self.numRX, int(fileSize / self.numRX)), dtype='complex_')
        iter_idx = 0
        for i in range(0, fileSize, self.numADCSamples * 4):
            if iter_idx + self.numADCSamples > adcData.shape[1]:
                break
            adcData[0][iter_idx:iter_idx + self.numADCSamples] = (
                LVDS[0][i:i + self.numADCSamples] + 1j * LVDS[1][i:i + self.numADCSamples]
            )
            adcData[1][iter_idx:iter_idx + self.numADCSamples] = (
                LVDS[0][i + self.numADCSamples:i + self.numADCSamples * 2]
                + 1j * LVDS[1][i + self.numADCSamples:i + self.numADCSamples * 2]
            )
            adcData[2][iter_idx:iter_idx + self.numADCSamples] = (
                LVDS[0][i + self.numADCSamples * 2:i + self.numADCSamples * 3]
                + 1j * LVDS[1][i + self.numADCSamples * 2:i + self.numADCSamples * 3]
            )
            adcData[3][iter_idx:iter_idx + self.numADCSamples] = (
                LVDS[0][i + self.numADCSamples * 3:i + self.numADCSamples * 4]
                + 1j * LVDS[1][i + self.numADCSamples * 3:i + self.numADCSamples * 4]
            )
            iter_idx += self.numADCSamples

        adcDataReshape = adcData.reshape(self.numRX, -1, self.numADCSamples)
        print(f'Shape of radar data from {fileName}: {adcDataReshape.shape}')
        return adcDataReshape

    def clutterRemoval(self, input_val, axis=0):
        reordering = np.arange(len(input_val.shape))
        reordering[0] = axis
        reordering[axis] = 0
        input_val = input_val.transpose(reordering)
        mean = input_val.transpose(reordering).mean(0)
        output_val = input_val - np.expand_dims(mean, axis=0)
        out = output_val.transpose(reordering)
        return out

    def generateHeatmap(self, frame):
        if frame.shape[1] < self.numChirp:
            raise ValueError(f'Frame chirp number is insufficient: {frame.shape}, expected chirps={self.numChirp}')

        # TX1 + TX3: 8 virtual antennas；TX2: 4 virtual antennas
        dataRadar = np.zeros((self.numRX * 2, self.idxProcChirp, self.numADCSamples), dtype='complex_')
        dataRadar2 = np.zeros((self.numRX, self.idxProcChirp, self.numADCSamples), dtype='complex_')

        for idxRX in range(self.numRX):
            for idxChirp in range(self.numChirp):
                if idxChirp % 3 == 0:
                    dataRadar[idxRX, idxChirp // 3] = frame[idxRX, idxChirp]
                elif idxChirp % 3 == 1:
                    dataRadar2[idxRX, idxChirp // 3] = frame[idxRX, idxChirp]
                else:
                    dataRadar[idxRX + 4, idxChirp // 3] = frame[idxRX, idxChirp]

        # static clutter removal along slow-time/chirp dimension
        dataRadar = np.transpose(dataRadar, (1, 0, 2))
        dataRadar = self.clutterRemoval(dataRadar, axis=0)
        dataRadar = np.transpose(dataRadar, (1, 0, 2))

        dataRadar2 = np.transpose(dataRadar2, (1, 0, 2))
        dataRadar2 = self.clutterRemoval(dataRadar2, axis=0)
        dataRadar2 = np.transpose(dataRadar2, (1, 0, 2))

        # range-doppler FFT
        for idxRX in range(self.numRX * 2):
            dataRadar[idxRX, :, :] = np.fft.fft2(dataRadar[idxRX, :, :])
        for idxRX in range(self.numRX):
            dataRadar2[idxRX, :, :] = np.fft.fft2(dataRadar2[idxRX, :, :])

        # zero padding for azimuth/elevation FFT
        dataRadar = np.pad(dataRadar, ((0, self.numAngleBins - dataRadar.shape[0]), (0, 0), (0, 0)), mode='constant')
        dataRadar2 = np.pad(dataRadar2, ((2, self.numAngleBins - 4 - 2), (0, 0), (0, 0)), mode='constant')

        dataMerge = np.stack((dataRadar, dataRadar2))
        dataMerge = np.pad(dataMerge, ((0, self.numEleBins - dataMerge.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')

        for idxChirp in range(self.idxProcChirp):
            for idxADC in range(self.numADCSamples):
                dataMerge[:, 2, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 2, idxChirp, idxADC])
                dataMerge[:, 3, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 3, idxChirp, idxADC])
                dataMerge[:, 4, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 4, idxChirp, idxADC])
                dataMerge[:, 5, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 5, idxChirp, idxADC])
                for idxEle in range(self.numEleBins):
                    dataMerge[idxEle, :, idxChirp, idxADC] = np.fft.fft(dataMerge[idxEle, :, idxChirp, idxADC])

        idxADCSpecific = [i for i in range(124, 60, -1)]
        rate = self.adcRatio

        dataTemp = np.zeros(
            (self.idxProcChirp, self.numADCSamples // rate, self.numAngleBins, self.numEleBins),
            dtype='complex_'
        )
        dataFFTGroup = np.zeros(
            (self.idxProcChirp // self.numGroupChirp, self.numADCSamples // rate, self.numAngleBins, self.numEleBins),
            dtype='complex_'
        )

        for idxEle in range(self.numEleBins):
            for idxRX in range(self.numAngleBins):
                for idxADC in range(self.numADCSamples // rate):
                    dataTemp[:, idxADC, idxRX, idxEle] = dataMerge[idxEle, idxRX, :, idxADCSpecific[idxADC]]
                    dataTemp[:, idxADC, idxRX, idxEle] = np.fft.fftshift(dataTemp[:, idxADC, idxRX, idxEle], axes=0)

        # idxProcChirp=255, numGroupChirp=4 -> 63。这里严格取 63 个 Doppler slice，避免 62/63 不一致。
        chirpPad = self.idxProcChirp // self.numGroupChirp
        start = self.idxProcChirp // 2 - chirpPad // 2
        end = start + chirpPad
        for out_i, idxChirp in enumerate(range(start, end)):
            dataFFTGroup[out_i, :, :, :] = self.postProcessFFT3D(
                np.transpose(dataTemp[idxChirp, :, :, :], (1, 2, 0))
            )

        return dataFFTGroup

    def saveDataAsFigure(self, img, joints, output, visDirName, idxFrame, output2=None):
        heatmap = PlotHeatmaps(joints, self.numKeypoints, self.objNums)
        PlotMaps(visDirName, self.xIndices, self.yIndices, idxFrame, output, img, heatmap, output2)

    def saveRadarData(self, matrix, dirName, idxFrame):
        os.makedirs(dirName, exist_ok=True)
        dirSave = os.path.join(dirName, f'{idxFrame:09d}.npy')
        np.save(dirSave, matrix)

    def parse_synchronization_info(self, video_csv_path, radar_log_path):
        video_start_time = None
        radar_start_dt = None

        try:
            with open(video_csv_path, 'r') as f:
                reader = csv.reader(f)
                _ = next(reader, None)
                first_row = next(reader, None)
                if first_row:
                    time_str = first_row[1].strip()
                    video_start_time = datetime.strptime(time_str, '%H:%M:%S.%f').time()
        except Exception as e:
            print(f'[WARN] Failed to read video timestamp {video_csv_path}: {e}')
            return 0

        try:
            with open(radar_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'Capture start time' in line:
                        date_str = line.split(' - ')[1].strip()
                        radar_start_dt = datetime.strptime(date_str, '%a %b %d %H:%M:%S %Y')
                        break
        except Exception as e:
            print(f'[WARN] Failed to read radar log {radar_log_path}: {e}')
            return 0

        if video_start_time is None or radar_start_dt is None:
            return 0

        video_dt_full = datetime.combine(radar_start_dt.date(), video_start_time)
        diff_seconds = (radar_start_dt - video_dt_full).total_seconds()
        offset_frames = int(round(diff_seconds * self.framePerSecond))
        return offset_frames

    def get_offsets(self, hori_radar_path, vert_radar_path):
        if not USE_OFFSET:
            return 0, 0

        raw_radar_dir = os.path.dirname(hori_radar_path)
        pair_root_dir = os.path.dirname(raw_radar_dir)
        video_csv = os.path.join(pair_root_dir, 'timestamps.csv')
        if not os.path.exists(video_csv):
            video_csv = os.path.join(raw_radar_dir, 'timestamps.csv')

        if not os.path.exists(video_csv):
            print('[WARN] USE_OFFSET=True but timestamps.csv not found. Use offsets 0, 0.')
            return 0, 0

        hori_logs = glob.glob(os.path.join(raw_radar_dir, '*azimuth*Raw_LogFile.csv'))
        vert_logs = glob.glob(os.path.join(raw_radar_dir, '*elevation*Raw_LogFile.csv'))

        offset_hori = self.parse_synchronization_info(video_csv, hori_logs[0]) if hori_logs else 0
        offset_vert = self.parse_synchronization_info(video_csv, vert_logs[0]) if vert_logs else 0
        return offset_hori, offset_vert

    def _safe_radar_frame_index(self, target_video_frame, offset, total_radar_frames):
        idx_read = target_video_frame - offset
        if CLAMP_RADAR_INDEX:
            return min(max(idx_read, 0), total_radar_frames - 1)
        return idx_read

    def processRadarDataHoriVert(self):
        _ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for idxName in tqdm(range(len(self.radarDataFileNameGroup))):
            hori_radar_path = self.radarDataFileNameGroup[idxName][0]
            vert_radar_path = self.radarDataFileNameGroup[idxName][1]

            offset_hori, offset_vert = self.get_offsets(hori_radar_path, vert_radar_path)
            print(f'Offsets -> Hori: {offset_hori}, Vert: {offset_vert}')

            adcDataHori = self.getadcDataFromDCA1000(hori_radar_path)
            adcDataVert = self.getadcDataFromDCA1000(vert_radar_path)

            total_hori_frames = adcDataHori.shape[1] // self.numChirp
            total_vert_frames = adcDataVert.shape[1] // self.numChirp
            print(f'Total radar frames -> Hori: {total_hori_frames}, Vert: {total_vert_frames}')

            if total_hori_frames <= 0 or total_vert_frames <= 0:
                raise RuntimeError('Radar raw data is too short. Check numChirp or raw file.')

            for idxFrame in tqdm(range(0, self.numFrame)):
                idx_hori_read = self._safe_radar_frame_index(idxFrame, offset_hori, total_hori_frames)
                idx_vert_read = self._safe_radar_frame_index(idxFrame, offset_vert, total_vert_frames)

                if idx_hori_read < 0 or idx_vert_read < 0:
                    print(f'[SKIP] Frame {idxFrame}: radar has not started yet.')
                    continue
                if idx_hori_read >= total_hori_frames or idx_vert_read >= total_vert_frames:
                    print(f'[STOP] Frame {idxFrame}: radar data exhausted.')
                    break

                frameHori = adcDataHori[:, self.numChirp * idx_hori_read:self.numChirp * (idx_hori_read + 1), 0:self.numADCSamples]
                frameVert = adcDataVert[:, self.numChirp * idx_vert_read:self.numChirp * (idx_vert_read + 1), 0:self.numADCSamples]

                time1 = time.time()
                outputHori = self.generateHeatmap(frameHori)
                time2 = time.time()
                outputVert = self.generateHeatmap(frameVert)
                time3 = time.time()

                self.saveRadarData(outputHori, os.path.join(self.saveDirNameGroup[idxName], 'hori'), idxFrame)
                self.saveRadarData(outputVert, os.path.join(self.saveDirNameGroup[idxName], 'vert'), idxFrame)

                print(
                    f'{hori_radar_path}, finished frame {idxFrame}, '
                    f'avg single radar FFT time={((time2 - time1) + (time3 - time2)) / 2:.4f}s, '
                    f'two radar FFT time={time3 - time1:.4f}s',
                    end='\r'
                )
            print()

    def _load_single_person_joints_list(self, annotGroup, idxFrame, img_shape=None):
        """
        兼容单人 JSON：
        annotGroup = [[frame0, frame1, ...]]

        注意：plot_utils.PlotHeatmaps 的输入不是裸的 (14, 2)，
        而是按 person 维度包一层，即 [(14, 2)]。
        如果直接传 (14, 2)，GT Heatmap 很容易为空或错位。
        """
        if not isinstance(annotGroup, list) or len(annotGroup) == 0:
            raise ValueError('Invalid annotation json structure. Expected [[frame_dict, ...]].')
        if idxFrame >= len(annotGroup[0]):
            raise IndexError(f'idxFrame={idxFrame} exceeds annotation length={len(annotGroup[0])}')

        joints = np.array(annotGroup[0][idxFrame]['joints'], dtype=np.float32)

        # 脱壳：兼容 joints = [[14,2]] 或 [[[14,2]]] 的情况
        while joints.ndim > 2:
            joints = joints[0]

        if joints.shape != (self.numKeypoints, 2):
            raise ValueError(f'Invalid joints shape at frame {idxFrame}: {joints.shape}, expected (14, 2)')


        # 坐标尺度兼容，仅用于可视化：
        # hrnet_annot.json 中的 joints 默认是 256×256 坐标；
        # 如果 RGB 图像不是 256×256，则临时缩放到 RGB 图像坐标系。
        # if img_shape is not None:
        #     img_h, img_w = img_shape[:2]

        #     label_w = 256.0
        #     label_h = 256.0

        #     joints[:, 0] = joints[:, 0] * (img_w / label_w)
        #     joints[:, 1] = joints[:, 1] * (img_h / label_h)

        # 关键：单人也要包成 [person_joints]，不要直接返回 (14,2)
        return [joints.tolist()]

    def loadDataPlot(self):
        for idxName in tqdm(range(len(self.radarDataFileNameGroup))):
            with open(self.jointsFileNameGroup[idxName], 'r') as fp:
                annotGroup = json.load(fp)

            for idxFrame in tqdm(range(0, self.numFrame)):
                hori_path = os.path.join(self.saveDirNameGroup[idxName], 'hori', f'{idxFrame:09d}.npy')
                vert_path = os.path.join(self.saveDirNameGroup[idxName], 'vert', f'{idxFrame:09d}.npy')
                img_path = os.path.join(self.rgbFileNameGroup[idxName], f'{idxFrame:09d}.jpg')

                if not os.path.exists(hori_path) or not os.path.exists(vert_path):
                    print(f'[SKIP] Missing radar npy at frame {idxFrame}')
                    continue
                if not os.path.exists(img_path):
                    print(f'[SKIP] Missing RGB image: {img_path}')
                    continue

                outputHori = np.load(hori_path)
                outputVert = np.load(vert_path)
                outputHori = np.mean(np.abs(outputHori), axis=(0, 3))
                outputVert = np.mean(np.abs(outputVert), axis=(0, 3))

                img = np.array(Image.open(img_path).convert('RGB'))
                joints_list = self._load_single_person_joints_list(annotGroup, idxFrame, img_shape=img.shape)

                visDirName = os.path.join(self.saveDirNameGroup[idxName], 'visualization', f'{idxFrame:09d}.png')
                os.makedirs(os.path.dirname(visDirName), exist_ok=True)

                # 单人逻辑：仍然按 1 个人处理，但传给 PlotHeatmaps 时需要 [person_joints]
                self.saveDataAsFigure(img, joints_list, outputHori, visDirName, idxFrame, outputVert)
                print(f'{self.radarDataFileNameGroup[idxName][0]}, finished visualization frame {idxFrame}', end='\r')
            print()


if __name__ == '__main__':
    radarObject = RadarObject()
    if VISUALIZATION:
        radarObject.loadDataPlot()
    else:
        radarObject.processRadarDataHoriVert()
