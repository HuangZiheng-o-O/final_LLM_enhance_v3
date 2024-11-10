import numpy as np
import torch


positions_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/interpolated_sampled_reconstructed_points_128_9.npy'
data_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy'
output_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/newheart11912.npy'


rawdata = torch.tensor(np.load(data_path), dtype=torch.float32)
positions = torch.tensor(np.load(positions_path), dtype=torch.float32)

print(positions.shape)



import os
import numpy as np
import torch

def compute_original_facing_direction(rawdata):
    # Lower legs
    l_idx1, l_idx2 = 17, 18
    # Right/Left foot
    fid_r, fid_l = [14, 15], [19, 20]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [11, 16, 5, 8]
    # l_hip, r_hip
    r_hip, l_hip = 11, 16
    joints_num = 21


    r_hip_index = 11  # 调整为您的骨架定义
    l_hip_index = 16

    r_hip_positions = rawdata[:4, r_hip_index, [0, 2]]
    l_hip_positions = rawdata[:4, l_hip_index, [0, 2]]

    across = r_hip_positions - l_hip_positions
    across_norm = torch.norm(across, dim=1, keepdim=True)
    across_direction = across / across_norm

    facing_direction = torch.stack([-across_direction[:, 1], across_direction[:, 0]], dim=1)
    facing_direction_norm = torch.norm(facing_direction, dim=1, keepdim=True)
    facing_direction = facing_direction / facing_direction_norm

    # 直接返回前四帧的朝向方向
    return facing_direction

def compute_expected_facing_direction(positions):
    delta_positions = positions[1:5] - positions[:4]
    motion_direction = delta_positions / torch.norm(delta_positions, dim=1, keepdim=True)

    # 直接返回前四帧的运动方向
    return motion_direction


def compute_angle_between_directions(average_facing_direction, average_motion_direction):
    dot_product = torch.dot(average_facing_direction, average_motion_direction)
    dot_product = dot_product.clamp(-1.0, 1.0)
    angle = torch.acos(dot_product)

    cross = average_facing_direction[0]*average_motion_direction[1] - average_facing_direction[1]*average_motion_direction[0]
    if cross < 0:
        angle = -angle

    return angle

def rotate_positions(positions, angle):
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    rotation_matrix = torch.tensor([[cos_angle, -sin_angle],
                                    [sin_angle, cos_angle]])

    rotated_positions = positions @ rotation_matrix
    return rotated_positions

def process_and_save_scaled_velocity_with_rotation(positions_path, data_path, output_directory, standard_velocity_abs_mean=0.015):
    os.makedirs(output_directory, exist_ok=True)
    rawdata = torch.tensor(np.load(data_path), dtype=torch.float32)
    positions = torch.tensor(np.load(positions_path), dtype=torch.float32)

    # 步骤 1：计算原始面朝方向
    average_facing_direction = compute_original_facing_direction(rawdata)

    # 提取 positions 的 x 和 z 坐标
    positions_2d = positions[:, [0, 2]]

    # 步骤 2：计算预期面朝方向（运动方向）
    average_motion_direction = compute_expected_facing_direction(positions_2d)

    # 步骤 3：计算夹角
    angle_offset = compute_angle_between_directions(average_facing_direction, average_motion_direction)
    print(f"Computed angle offset: {angle_offset.item() * 180 / np.pi} degrees")

    # 步骤 4：旋转 positions
    rotated_positions = rotate_positions(positions_2d, angle_offset)

    # 步骤 5：使用旋转后的 positions 重新计算
    rot_vel, standard_scaling_factor = compute_rot_vel_and_scaling(rotated_positions, standard_velocity_abs_mean)
    scaled_rot_vel = rot_vel * standard_scaling_factor
    root_linear_velocity = compute_root_linear_velocity(rotated_positions, scaled_rot_vel)

    # 更新 rawdata 中的 rot_vel 和 root_linear_velocity
    updated_rawdata = rawdata.clone()
    updated_rawdata[3:, 0] = scaled_rot_vel[3:]
    updated_rawdata[3:, 1:3] = root_linear_velocity[3:]

    # 保存更新后的数据
    original_filename = os.path.basename(positions_path)
    filename_wo_ext, ext = os.path.splitext(original_filename)
    new_filename = f"{filename_wo_ext}_adjusted{ext}"
    new_output_path = os.path.join(output_directory, new_filename)
    np.save(new_output_path, updated_rawdata.numpy())
    print(f"Saved updated data to {new_output_path}")




def compute_root_motion(positions,standard_scale_mode=False,standard_velocity_abs_mean=0.015,scaling_factor_mode = False,scaling_factor=1.0):
    """
    positions: Tensor of shape (seq_len, 2), positions in x and z
    Returns:
    rot_vel: Tensor of shape (seq_len,), rotation info
    root_linear_velocity: Tensor of shape (seq_len, 2), data[..., :-1, 1:3]
    """

    # positions = positions[:, :, [0, 2]]
    # # from # (128, 1, 2) to # (128,2)
    # positions = positions.squeeze(1)
    # # 将 positions 转换为 PyTorch Tensor
    # positions = torch.tensor(positions, dtype=torch.float32)
    #
    # print("positions.shape", positions.shape)

    print(positions.shape)
    seq_len = positions.shape[0]

    # Step 1: Compute global displacements
    delta_positions = positions[1:] - positions[:-1]  # Shape: (seq_len - 1, 2)
    delta_positions = torch.cat([torch.zeros(1, 2), delta_positions], dim=0)  # Pad to match seq_len

    # Step 2: Compute heading angles of movement vectors
    theta = torch.atan2(delta_positions[:, 1], delta_positions[:, 0])  # Shape: (seq_len,)

    # Step 3: Unwrap angles to ensure continuity
    theta_unwrapped = torch.from_numpy(np.unwrap(theta.numpy()))
    theta_unwrapped = theta_unwrapped.float()/10

    # Step 4: Compute rotation velocities
    rot_vel = torch.zeros(seq_len)
    rot_vel[0] = theta_unwrapped[0]
    rot_vel[1:] = theta_unwrapped[1:] - theta_unwrapped[:-1]

    standard_scaling_factor = (standard_velocity_abs_mean/rot_vel.abs().mean())

    print("important_standard_scaling_factor", standard_scaling_factor)

    if standard_scale_mode:
        rot_vel = rot_vel* standard_scaling_factor
    elif scaling_factor_mode:
        assert scaling_factor > 0, 'scaling_factor must be positive'
        assert scaling_factor <= 1, 'scaling_factor must be less than or equal to 1'
        rot_vel = rot_vel*scaling_factor
    else:
        rot_vel = rot_vel

    # Step 5: Compute cumulative rotation angles
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[1:] = rot_vel[:-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=0)

    # Step 6: Compute rotation quaternions
    r_rot_quat = torch.zeros(seq_len, 4)
    r_rot_quat[:, 0] = torch.cos(r_rot_ang/2)
    r_rot_quat[:, 2] = torch.sin(r_rot_ang/2)

    # Step 7: Prepare positions in 3D
    positions_3d = torch.zeros(seq_len, 3)
    positions_3d[:, [0, 2]] = positions

    # Step 8: Compute delta positions in 3D
    delta_positions_3d = positions_3d[1:] - positions_3d[:-1]  # Shape: (seq_len - 1, 3)

    # Step 9: Rotate delta positions to local frame
    r_rot_quat_inv = qinv(r_rot_quat[1:])  # Shape: (seq_len - 1, 4)
    local_delta_positions = qrot(r_rot_quat_inv, delta_positions_3d)  # Shape: (seq_len - 1, 3)

    # Step 10: Extract root_linear_velocity (only x and z components)
    root_linear_velocity = local_delta_positions[:, [0, 2]]  # Shape: (seq_len - 1, 2)

    # Step 11: Pad to match seq_len
    root_linear_velocity = torch.cat([torch.zeros(1, 2), root_linear_velocity], dim=0)

    return rot_vel, root_linear_velocity

# Quaternion inversion and rotation functions
def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
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

# Compute rot_vel and root_linear_velocity from positions
rot_vel, root_linear_velocity = compute_root_motion(positions)


# Update rawdata with new root_rot_velocity and root_linear_velocity
updated_rawdata = rawdata.clone()
updated_rawdata[..., 0] = rot_vel   # Replace root_rot_velocity
updated_rawdata[..., 1:3] = root_linear_velocity/2  # Replace root_linear_velocity


# Output the updated data to review the changes
print(updated_rawdata[:5])
np.save(output_path, updated_rawdata)