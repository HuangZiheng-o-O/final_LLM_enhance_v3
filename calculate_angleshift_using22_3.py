import numpy as np
import torch


# 四元数的逆和旋转函数
def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask

def qrot(q, v):
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = v.shape
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    rotated_v = v + 2 * (q[:, :1] * uv + uuv)
    return rotated_v.view(original_shape)

# 计算期望的朝向方向
def compute_expected_facing_direction(positions):
    # 仅使用前四帧
    delta_positions = positions[1:5] - positions[:4]
    motion_direction = delta_positions / torch.norm(delta_positions, dim=1, keepdim=True)
    return motion_direction

# 计算外角的弧度和度数
def compute_external_angles(facing_direction, motion_direction):
    dot_product = (facing_direction * motion_direction).sum(dim=1)
    facing_norm = torch.norm(facing_direction, dim=1)
    motion_norm = torch.norm(motion_direction, dim=1)

    cosine_similarity = dot_product / (facing_norm * motion_norm)
    internal_angles_radians = torch.acos(cosine_similarity)
    internal_angles_degrees = internal_angles_radians * (180 / torch.pi)

    # 计算外角
    external_angles_radians = torch.pi - internal_angles_radians
    external_angles_degrees = 180 - internal_angles_degrees

    # 输出平均外角
    avg_external_angle_radians = external_angles_radians.mean()
    avg_external_angle_degrees = external_angles_degrees.mean()

    return avg_external_angle_radians, avg_external_angle_degrees


# 加载数据
positions_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/interpolated_sampled_reconstructed_points_128_9.npy'
data_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy'
output_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/newheart11912.npy'
original_223positions = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/video_npy/walk_and_wave_joint.npy'

rawdata = torch.tensor(np.load(data_path), dtype=torch.float32)
positions = torch.tensor(np.load(positions_path), dtype=torch.float32)
original_positions = torch.tensor(np.load(original_223positions), dtype=torch.float32)

# 提取原始的 2D 坐标
extracted_2Dorigin = original_positions[:, 1:2, [0, 2]].squeeze(1)  # 提取形状 (128, 2)
# print("Extracted 2D origin:", extracted_2Dorigin)
print("Original_positions shape:", extracted_2Dorigin.shape)
print("Rawdata shape:", rawdata.shape)
print("Positions shape:", positions.shape)


# 获取前四帧的朝向
facing_direction = compute_expected_facing_direction(extracted_2Dorigin)
motion_direction = compute_expected_facing_direction(positions)

# 计算并输出平均外角
avg_external_angle_radians, avg_external_angle_degrees = compute_external_angles(facing_direction, motion_direction)
print("Average external angle in radians:", avg_external_angle_radians)
print("Average external angle in degrees:", avg_external_angle_degrees)

#################################################
#Average external angle in radians: tensor(2.4041)

