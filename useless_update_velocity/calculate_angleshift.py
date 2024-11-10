# import numpy as np
# import torch
#
# # 加载数据
# positions_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/interpolated_sampled_reconstructed_points_128_9.npy'
# data_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy'
# output_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/newheart11912.npy'
#
# rawdata = torch.tensor(np.load(data_path), dtype=torch.float32)
# positions = torch.tensor(np.load(positions_path), dtype=torch.float32)
# print("Rawdata shape:", rawdata.shape)
# print("Positions shape:", positions.shape)
#
# # 计算四元数的逆和旋转函数
# def qinv(q):
#     assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
#     mask = torch.ones_like(q)
#     mask[..., 1:] = -mask[..., 1:]
#     return q * mask
#
# def qrot(q, v):
#     assert q.shape[-1] == 4
#     assert v.shape[-1] == 3
#     assert q.shape[:-1] == v.shape[:-1]
#
#     original_shape = v.shape
#     q = q.contiguous().view(-1, 4)
#     v = v.contiguous().view(-1, 3)
#
#     qvec = q[:, 1:]
#     uv = torch.cross(qvec, v, dim=1)
#     uuv = torch.cross(qvec, uv, dim=1)
#     rotated_v = v + 2 * (q[:, :1] * uv + uuv)
#     return rotated_v.view(original_shape)
#
# # 计算原始朝向方向
# def compute_original_facing_direction(rawdata):
#     # 定义索引和偏移量
#     root_rot_velocity_dim = 1
#     root_linear_velocity_dim = 2
#     root_y_dim = 1
#     ric_data_dim = 21 * 3
#     rot_data_dim = 21 * 6
#     local_velocity_dim = 22 * 3
#     foot_contact_dim = 4
#
#     offset_ric_data = root_rot_velocity_dim + root_linear_velocity_dim + root_y_dim
#     r_hip_index = offset_ric_data + 11 * 3
#     l_hip_index = offset_ric_data + 16 * 3
#
#     r_hip_positions = rawdata[:4, [r_hip_index, r_hip_index + 2]]
#     l_hip_positions = rawdata[:4, [l_hip_index, l_hip_index + 2]]
#
#     across = r_hip_positions - l_hip_positions
#     across_norm = torch.norm(across, dim=1, keepdim=True)
#     across_direction = across / across_norm
#     facing_direction = torch.stack([-across_direction[:, 1], across_direction[:, 0]], dim=1)
#     facing_direction = facing_direction / torch.norm(facing_direction, dim=1, keepdim=True)
#
#     return facing_direction
#
# # 计算期望的朝向方向
# def compute_expected_facing_direction(positions):
#     delta_positions = positions[1:5] - positions[:4]
#     motion_direction = delta_positions / torch.norm(delta_positions, dim=1, keepdim=True)
#     return motion_direction
#
# # 计算外角的弧度和度数
# def compute_external_angles(facing_direction, motion_direction):
#     dot_product = (facing_direction * motion_direction).sum(dim=1)
#     facing_norm = torch.norm(facing_direction, dim=1)
#     motion_norm = torch.norm(motion_direction, dim=1)
#
#     cosine_similarity = dot_product / (facing_norm * motion_norm)
#     internal_angles_radians = torch.acos(cosine_similarity)
#     internal_angles_degrees = internal_angles_radians * (180 / torch.pi)
#
#     # 计算外角
#     external_angles_radians = torch.pi - internal_angles_radians
#     external_angles_degrees = 180 - internal_angles_degrees
#
#     # 输出平均外角
#     avg_external_angle_radians = external_angles_radians.mean()
#     avg_external_angle_degrees = external_angles_degrees.mean()
#
#     return avg_external_angle_radians, avg_external_angle_degrees
#
# # 获取前四帧的朝向
# facing_direction = compute_original_facing_direction(rawdata)
# motion_direction = compute_expected_facing_direction(positions)
#
# # 计算并输出平均外角
# avg_external_angle_radians, avg_external_angle_degrees = compute_external_angles(facing_direction, motion_direction)
# print("Average external angle in radians:", avg_external_angle_radians)
# print("Average external angle in degrees:", avg_external_angle_degrees)
