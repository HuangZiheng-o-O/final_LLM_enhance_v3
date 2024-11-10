# import os
# import numpy as np
# import torch
# import math
# from tqdm import tqdm
#
#
# # 定义2D旋转函数，接受有符号角度
#
# def rotate_2d(points, angle_rad):
#     """
#     Rotate 2D points by a given angle in radians.
#     Positive angle: counter-clockwise rotation
#     Negative angle: clockwise rotation
#     """
#     angle_rad = torch.tensor(angle_rad, dtype=points.dtype, device=points.device)  # 确保 angle_rad 是 Tensor
#     cos_theta = torch.cos(angle_rad)
#     sin_theta = torch.sin(angle_rad)
#     rotation_matrix = torch.tensor([[cos_theta, -sin_theta],
#                                     [sin_theta, cos_theta]], dtype=points.dtype, device=points.device)
#     return torch.matmul(points, rotation_matrix.T)
#
#
#
# # 计算预期的朝向方向（标准）
# def compute_standard_facing_direction(original_positions):
#     """
#     Compute the standard facing direction from the first four frames of original_positions.
#     original_positions: Tensor of shape (frames, joints, coordinates)
#     Returns: 单一的标准运动方向 2D 向量
#     """
#     extracted_2Dorigin = original_positions[:, 1:2, [0, 2]].squeeze(1)  # Shape: (frames, 2)
#     delta_positions = extracted_2Dorigin[1:5] - extracted_2Dorigin[:4]  # Shape: (4, 2)
#     avg_delta = delta_positions.mean(dim=0)  # Shape: (2,)
#     norm = torch.norm(avg_delta)
#     if norm < 1e-6:
#         raise ValueError("标准运动方向的平均移动向量接近于零，无法计算方向。")
#     standard_motion_direction = avg_delta / norm  # Shape: (2,)
#     return standard_motion_direction
#
#
# # 计算当前轨迹的运动方向
# def compute_motion_direction(positions_2D):
#     delta_positions = positions_2D[1:5] - positions_2D[:4]  # Shape: (4, 2)
#     avg_delta = delta_positions.mean(dim=0)  # Shape: (2,)
#     norm = torch.norm(avg_delta)
#     if norm < 1e-6:
#         raise ValueError("当前轨迹的平均移动向量接近于零，无法计算方向。")
#     motion_direction = avg_delta / norm  # Shape: (2,)
#     return motion_direction
#
#
# # 计算有符号的旋转角度，从 source_dir 旋转到 target_dir
# def compute_signed_angle(source_dir, target_dir):
#     source = source_dir / (torch.norm(source_dir) + 1e-8)
#     target = target_dir / (torch.norm(target_dir) + 1e-8)
#     cross = source[0] * target[1] - source[1] * target[0]
#     dot = source[0] * target[0] + source[1] * target[1]
#     angle = torch.atan2(cross, dot)
#     return angle.item()
#
#
# # 定义路径
# input_dir = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/interpolated_sampled'
# output_dir = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/interpolated_sampled_corrected'
# original_positions_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/video_npy/walk_and_wave_joint.npy'
# output_log_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/rotation_angles_log.txt'
#
# # 创建输出目录
# os.makedirs(output_dir, exist_ok=True)
#
# # 加载原始的标准运动方向
# try:
#     original_positions_np = np.load(original_positions_path)
#     original_positions = torch.tensor(original_positions_np, dtype=torch.float32)
#     standard_motion_direction = compute_standard_facing_direction(original_positions)
#     print(f"标准运动方向向量: {standard_motion_direction.numpy()}")
# except Exception as e:
#     print(f"加载或计算标准运动方向时出错: {e}")
#     exit(1)
#
# # 打开日志文件记录旋转角度
# with open(output_log_path, 'w') as log_file:
#     log_file.write("File Name\tRotation Angle (Radians)\n")
#     log_file.write("-" * 40 + "\n")
#
#     # 列出所有 .npy 文件
#     file_list = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
#     print(f"在输入目录中找到 {len(file_list)} 个文件。")
#
#     for file_name in tqdm(file_list, desc="正在处理文件"):
#         input_file_path = os.path.join(input_dir, file_name)
#         output_file_path = os.path.join(output_dir, file_name)
#         try:
#             # 加载位置数据
#             positions_np = np.load(input_file_path)
#             positions = torch.tensor(positions_np, dtype=torch.float32)
#
#             # 提取 2D 坐标
#             if positions.ndim == 2 and positions.shape[1] == 2:
#                 positions_2D = positions
#             elif positions.ndim == 3 and positions.shape[2] >= 2:
#                 positions_2D = positions[:, 1, :2]
#             else:
#                 raise ValueError("位置数据的维度不符合预期，无法提取2D坐标。")
#
#             # 计算当前轨迹的运动方向
#             current_motion_direction = compute_motion_direction(positions_2D)
#
#             # 计算旋转角度
#             angle_rad = compute_signed_angle(current_motion_direction, standard_motion_direction)
#
#             # 输出和记录旋转角度
#             print(f"文件: {file_name}, 旋转弧度: {angle_rad:.4f}")
#             log_file.write(f"{file_name}\t{angle_rad:.4f}\n")
#
#             # 旋转轨迹
#             rotated_positions = rotate_2d(positions_2D, angle_rad)
#
#             # 平移，使第一个点位于 (0, 0)
#             translated_positions = rotated_positions - rotated_positions[0]
#
#             # 保存
#             translated_positions_np = translated_positions.numpy()
#             np.save(output_file_path, translated_positions_np)
#
#         except Exception as e:
#             print(f"处理文件 {file_name} 时出错: {e}")
#
# print("批量处理完成。旋转角度已记录在:", output_log_path)
