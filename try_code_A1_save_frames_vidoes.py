
import cv2
import os
from tqdm import tqdm

# 定义要读取的文件夹路径
folder_path =  'E:/Datasets/Reflection//ref//'
#'D://Datasets//Reflection//Check_SIRR//'  reflection vivo1


# 获取文件夹中的所有视频文件
video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4') or f.endswith('.mov')]

# 循环处理每个视频文件
for video_file in tqdm(video_files):
    # 读取视频文件
    video_path = os.path.join(folder_path, video_file)
    video = cv2.VideoCapture(video_path)

    # 获取视频帧率和总帧数
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 定义保存图像的文件夹路径
    save_folder_path = os.path.join(folder_path, os.path.splitext(video_file)[0])
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # 循环读取每一帧图像，并保存前30帧和每隔25帧的帧为PNG格式文件
    for i in tqdm(range(total_frames)):
        ret, frame = video.read()
        if not ret:
            break
        if i < 30 or i % 25 == 0:
            save_path = os.path.join(save_folder_path, f'frame_{i:06d}.jpg')
            cv2.imwrite(save_path, frame)

    video.release()