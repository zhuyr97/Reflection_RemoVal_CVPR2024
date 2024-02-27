import os
import numpy as np
from PIL import Image

unified_path = 'E:/Datasets/Reflection//reflection vivo1//'   #"D:/Datasets/Reflection/Check_SIRR/"
# 指定文件夹列表、输出路径和 x 的数目
input_folders_dict = {'indoor_meetingroom': 37, 'indoor_night1': 36, 'indoor_retail1': 33, 'indoor_table_merge': 38,
                      'indoor_retail2': 37, 'indoor_retail3': 35, 'outdoor_afternoon_park1': 30,
                      'outdoor_chair': 38, 'outdoor_floor_plants': 32, 'outdoor_night1': 37,
                      'outdoor_night_park1': 33, 'outdoor_night_park2': 37, 'outdoor_night_park3': 37,
                      'outdoor_park1': 34, 'outdoor_plants': 39, 'outdoor_trees1': 35, 'outdoor_trees2': 37}

    # {'USTC_center': 30, 'USTC_center1': 29, 'indoor_center1': 34, 'indoor_center2': 29,
    #              'indoor_center3': 38, 'indoor_center4': 37, 'indoor_center5': 34, 'indoor_classroom': 33,
    #              'indoor_classroom1': 0, 'indoor_exhibition1': 34, 'indoor_exhibition2': 35,
    #              'indoor_exhibition3': 37, 'indoor_exhibition4': 38, 'indoor_exhibition5': 36,
    #              'indoor_exhibition6': 35, 'indoor_lab1': 23, 'indoor_lab2': 10, 'indoor_lab3': 34,
    #              'indoor_library1': 31, 'indoor_library2': 0, 'outdoor_floor': 42, 'outdoor_floor1': 34,
    #              'outdoor_floor2': 31, 'outdoor_floor3': 42, 'outdoor_stone': 32, 'painting1': 36,
    #              'painting2': 35, 'painting3': 39, 'painting4': 37}

output_folder = 'E:/Datasets/Reflection//reflection vivo1//GT_HZ1/'  #"D:/Datasets/Reflection/Check_SIRR/GT/"
output_txt = 'E:/Datasets/Reflection//reflection vivo1//Ref_HZ1.txt'  #"D:/Datasets/Reflection/Check_SIRR/Ref_USTC.txt"

# x = 0

# 假设之前已经将 .mov 文件名和数字存储为了一个字典 mov_dict

# 按键排序后遍历字典的键和值
for key in sorted(input_folders_dict.keys()):
    value = input_folders_dict[key]
    x = value
    # print(key, value)
    input_folder = unified_path + key
    # for input_folder in input_folders:
        # 获取文件夹名称并创建输出文件夹
    folder_name = os.path.basename(input_folder)
    print('folder_name: {} || value: {}'.format(folder_name, value))


    output_folder_path = output_folder  #os.path.join(output_folder, folder_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 读取文件夹下的所有图片并计算 ground truth 图片的大小
    image_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".jpg") or f.endswith(".png")])
    ground_truth_size = None
    if x > 0:
        for i in range(x):
            img = Image.open(image_files[i])
            if ground_truth_size is None:
                ground_truth_size = img.size
            else:
                assert img.size == ground_truth_size, "All ground truth images must have the same size"
        ground_truth = np.zeros((ground_truth_size[1], ground_truth_size[0], 3))
        for i in range(x):
            img = np.array(Image.open(image_files[i]))
            ground_truth += img
        ground_truth /= x
        ground_truth_path = output_folder_path + '/' + str(folder_name) + "_GT.png"
        #os.path.join(output_folder_path, f"{folder_name}_GT.png")
        ground_truth = Image.fromarray(np.uint8(ground_truth))
        ground_truth.save(ground_truth_path)
    else:
        ground_truth_path = None

    # 保存降质图片和 txt 文件
    with open(output_txt, "a") as f:
        for i in range(x, len(image_files)):
            img_path = image_files[i]
            degraded_path = input_folder + '/' + os.path.basename(img_path)
            #print('degraded_path:',degraded_path)
            #  os.path.join(output_folder_path, os.path.basename(img_path))
            img = Image.open(img_path)
            # 在这里对图像进行降质处理
            # img.save(degraded_path)
            # 将降质图片和 ground truth 图片的路径保存到 txt 文件中
            save_degraded_path  = degraded_path.split('Reflection//')[1]
            save_ground_truth_path = ground_truth_path.split('Reflection//')[1]

            #print('save_degraded_path:',save_degraded_path)
            # f.write(f"{degraded_path} {ground_truth_path}\n")
            #print(save_degraded_path, '||', save_ground_truth_path)
            f.write(f"{save_degraded_path} {save_ground_truth_path}\n")