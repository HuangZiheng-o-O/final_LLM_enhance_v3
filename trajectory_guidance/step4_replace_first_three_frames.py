import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

def step4_replace_first_three_frames_in_directory(output_dir, raw_path, new_output_dir):
    # 确保新输出目录存在
    os.makedirs(new_output_dir, exist_ok=True)

    # 读取原始的前三帧数据
    raw_data = np.load(raw_path)[:3]

    # 定义替换前三帧的处理函数
    def replace_first_three_frames(file_path):
        output_data = np.load(file_path)
        output_data[:3] = raw_data  # 替换前三帧
        # 保存到新的文件路径
        new_file_path = os.path.join(new_output_dir, os.path.basename(file_path))
        np.save(new_file_path, output_data)

    # 遍历目录中的所有 .npy 文件并进行并行处理
    npy_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.npy')]

    with ThreadPoolExecutor() as executor:
        executor.map(replace_first_three_frames, npy_files)


# step4_replace_first_three_frames_in_directory(
#     output_dir="./output/263final_correct",
#     raw_path="./S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy",
#     new_output_dir="./output/263final_correct_replace3frames"
# )
