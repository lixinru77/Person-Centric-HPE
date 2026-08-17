import os
import cv2
import shutil
from tqdm import tqdm


def video2frame(base_root, target_frame_count, start_single_id, end_single_id, target_frame_rate):

    file_path = base_root

    for j in tqdm(range(start_single_id, end_single_id + 1)):
        videoDir = file_path + "rrv_pairs_{}/".format(j)

        output_folder = os.path.join(videoDir, "frame")
        os.makedirs(output_folder, exist_ok=True)

        video_path = os.path.join(videoDir, "gt_video", os.listdir(os.path.join(videoDir, "gt_video"))[0])

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"fps: {fps}")

        if not cap.isOpened():
            print("无法打开视频文件!")
            exit()

        # 计算帧抽取的间隔
        frame_interval = fps / target_frame_rate
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算实际抽帧间隔和抽取总帧数的范围
        frame_indices = [int(i * frame_interval) for i in range(target_frame_count)]

        frame_num = 0
        saved_frames = 0

        while saved_frames < target_frame_count:
            ret, frame = cap.read()

            if not ret or frame_num >= total_frames:
                break

            # 只保存映射到30FPS对应的帧
            if frame_num in frame_indices:
                frame_filename = os.path.join(output_folder, f'{saved_frames:09d}.jpg')
                cv2.imwrite(frame_filename, frame)
                saved_frames += 1

            frame_num += 1

        cap.release()


    # cv2.destroyAllWindows()


if __name__ == '__main__':
    base_root = "/mnt/newmy/RADAR_DATA/"
    target_frame_count = 300 # 设置目标帧数
    target_frame_rate = 10.0
    start_single_id = 3
    end_single_id = 4
    video2frame(base_root, target_frame_count, start_single_id, end_single_id, target_frame_rate)
