import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def process_motion_file(original_motion_path, raw_motion_path, output_dir_choice1, output_dir_choice2):
    """
    处理单个原始运动文件，生成两个修改后的.npy文件并保存。

    参数：
    - original_motion_path: 原始运动文件的路径（如 S_shape.npy）
    - raw_motion_path: 原始数据结构文件的路径（如 raw_sample0_repeat0_len128.npy）
    - output_dir_choice1: 存储 choice1 结果的目录
    - output_dir_choice2: 存储 choice2 结果的目录
    """
    # 获取原始运动名称
    original_motion_name = os.path.basename(original_motion_path).split('.')[0]

    # 加载原始数据结构
    data_structure = np.load(raw_motion_path, allow_pickle=True)
    # 提取 root_linear_velocity
    root_linear_velocity = data_structure[:, 1:3]  # Shape should be (seq_len, 2)

    # 加载 velocity_change_npy
    velocity_change_npy = np.load(original_motion_path, allow_pickle=True)
    # 提取 velocity_data
    velocity_data = velocity_change_npy[0, :, :]  # 根据实际数据调整索引
    original_length = velocity_data.shape[0]
    seq_len = root_linear_velocity.shape[0]

    # 插值到新的长度
    original_indices = np.linspace(0, 1, num=original_length)
    new_indices = np.linspace(0, 1, num=seq_len)

    # 插值 x 和 y
    interp_func_x = interp1d(original_indices, velocity_data[:, 0], kind='linear')
    interp_func_y = interp1d(original_indices, velocity_data[:, 1], kind='linear')

    # 创建新的速度数组
    velocity_change_npy_new = np.zeros((seq_len, 2))
    velocity_change_npy_new[:, 0] = interp_func_x(new_indices)
    velocity_change_npy_new[:, 1] = interp_func_y(new_indices)

    # 获取 root_linear_velocity 的最小值和最大值
    min_root_x, max_root_x = np.min(root_linear_velocity[:, 0]), np.max(root_linear_velocity[:, 0])
    min_root_y, max_root_y = np.min(root_linear_velocity[:, 1]), np.max(root_linear_velocity[:, 1])

    # --- Choice 1 ---
    choice1 = velocity_change_npy_new.copy()
    # 缩放 x
    min_choice1_x, max_choice1_x = np.min(choice1[:, 0]), np.max(choice1[:, 0])
    choice1[:, 0] = (choice1[:, 0] - min_choice1_x) / (max_choice1_x - min_choice1_x)
    choice1[:, 0] = choice1[:, 0] * (max_root_x - min_root_x) + min_root_x
    # 缩放 y
    min_choice1_y, max_choice1_y = np.min(choice1[:, 1]), np.max(choice1[:, 1])
    choice1[:, 1] = (choice1[:, 1] - min_choice1_y) / (max_choice1_y - min_choice1_y)
    choice1[:, 1] = choice1[:, 1] * (max_root_y - min_root_y) + min_root_y

    # --- Choice 2 ---
    choice2 = velocity_change_npy_new.copy()
    choice2 = choice2[:, [1, 0]]  # 交换 x 和 y
    # 缩放 x
    min_choice2_x, max_choice2_x = np.min(choice2[:, 0]), np.max(choice2[:, 0])
    choice2[:, 0] = (choice2[:, 0] - min_choice2_x) / (max_choice2_x - min_choice2_x)
    choice2[:, 0] = choice2[:, 0] * (max_root_x - min_root_x) + min_root_x
    # 缩放 y
    min_choice2_y, max_choice2_y = np.min(choice2[:, 1]), np.max(choice2[:, 1])
    choice2[:, 1] = (choice2[:, 1] - min_choice2_y) / (max_choice2_y - min_choice2_y)
    choice2[:, 1] = choice2[:, 1] * (max_root_y - min_root_y) + min_root_y

    # 替换 data_structure 中的数据并保存
    # Choice 1
    data_structure_choice1 = data_structure.copy()
    data_structure_choice1[:, 1:3] = choice1
    output_path_choice1 = os.path.join(output_dir_choice1, f"{original_motion_name}_change1.npy")
    np.save(output_path_choice1, data_structure_choice1)

    # Choice 2
    data_structure_choice2 = data_structure.copy()
    data_structure_choice2[:, 1:3] = choice2
    output_path_choice2 = os.path.join(output_dir_choice2, f"{original_motion_name}_change2.npy")
    np.save(output_path_choice2, data_structure_choice2)

    print(f"已处理 {original_motion_name} 并保存结果。")


def process_all_motion_files(motion_dir, raw_motion_path, output_dir_choice1, output_dir_choice2):
    """
    处理指定目录下的所有 {original_motion_name}.npy 文件。

    参数：
    - motion_dir: 存放 {original_motion_name}.npy 文件的目录
    - raw_motion_path: 原始数据结构文件的路径
    - output_dir_choice1: 存储 choice1 结果的目录
    - output_dir_choice2: 存储 choice2 结果的目录
    """
    # 创建输出目录
    os.makedirs(output_dir_choice1, exist_ok=True)
    os.makedirs(output_dir_choice2, exist_ok=True)

    # 遍历所有 .npy 文件
    for filename in os.listdir(motion_dir):
        if filename.endswith('.npy'):
            original_motion_path = os.path.join(motion_dir, filename)
            process_motion_file(original_motion_path, raw_motion_path, output_dir_choice1, output_dir_choice2)


# 主程序
if __name__ == "__main__":
    # 输入目录，包含 {original_motion_name}.npy 文件
    motion_dir = "../trajectory_guidance/npysave/"
    # 原始数据结构文件的路径（需要根据实际情况修改）
    raw_motion_path = "../S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy"
    # 输出目录
    output_dir_choice1 = "../trajectory_guidance/motion_change_npy/1"
    output_dir_choice2 = "../trajectory_guidance/motion_change_npy/2"

    process_all_motion_files(motion_dir, raw_motion_path, output_dir_choice1, output_dir_choice2)
