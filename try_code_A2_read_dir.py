import os

folder_path =  'E:/Datasets/Reflection//reflection vivo1//'
#   'D://Datasets//Reflection//Check_SIRR//'

files = os.listdir(folder_path)

print(sorted(files))


import os
import glob

# 指定文件夹路径
folder_path = folder_path

# 使用 glob 模块获取所有 .mov 格式文件的路径
mov_files = glob.glob(os.path.join(folder_path, "*.mov"))

# 提取文件名并保存到列表中
mov_filenames = []
for mov_file in mov_files:
    filename = os.path.basename(mov_file.split('.mov')[0])
    mov_filenames.append(filename)

# 打印输出文件名列表
print(mov_filenames)


import os
import glob

# 指定文件夹路径
folder_path = folder_path

# 使用 glob 模块获取所有 .mov 格式文件的路径，并按文件名排序
mov_files = sorted(glob.glob(os.path.join(folder_path, "*.mov")))

# 将文件名存储为字典格式
mov_dict = {}
for i, mov_file in enumerate(mov_files):
    filename = os.path.basename(mov_file.split('.mov')[0])
    mov_dict[filename] = i

# 打印输出字典
print(len(mov_dict))
print(mov_dict)