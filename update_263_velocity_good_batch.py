import os
import numpy as np
import torch
from jupyter_server.services.kernelspecs.handlers import pjoin

from trajectory_guidance.plot_scaled import root_linear_velocity_scaled
from useless_update_velocity.update_velocity3 import rot_vel


# Quaternion inversion function
def qinv(q):
    """
    Inverts a quaternion.

    Args:
        q (Tensor): Quaternion tensor of shape (..., 4).

    Returns:
        Tensor: Inverted quaternion.
    """
    assert q.shape[-1] == 4, 'q must be a tensor of shape (..., 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask

# Quaternion rotation function
def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.

    Args:
        q (Tensor): Quaternion tensor of shape (*, 4).
        v (Tensor): Vector tensor of shape (*, 3).

    Returns:
        Tensor: Rotated vectors of shape (*, 3).
    """
    assert q.shape[-1] == 4, 'q must have last dimension of size 4'
    assert v.shape[-1] == 3, 'v must have last dimension of size 3'
    assert q.shape[:-1] == v.shape[:-1], 'q and v must have the same shape except for last dimension'

    original_shape = v.shape
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    rotated_v = v + 2 * (q[:, :1] * uv + uuv)
    return rotated_v.view(original_shape)

# Function to compute rot_vel and standard_scaling_factor
def compute_rot_vel_and_scaling(positions, standard_velocity_abs_mean=0.015):
    """
    Computes the rotation velocity and standard scaling factor.

    Args:
        positions (Tensor): Tensor of shape (seq_len, 2), positions in x and z.
        standard_velocity_abs_mean (float): Desired mean absolute velocity for standard scaling.

    Returns:
        rot_vel (Tensor): Tensor of shape (seq_len,), rotation velocity.
        standard_scaling_factor (float): Scaling factor based on standard_velocity_abs_mean.
    """
    seq_len = positions.shape[0]

    # Step 1: Compute global displacements
    delta_positions = positions[1:] - positions[:-1]  # Shape: (seq_len - 1, 2)
    delta_positions = torch.cat([torch.zeros(1, 2), delta_positions], dim=0)  # Pad to match seq_len

    # Step 2: Compute heading angles of movement vectors
    theta = torch.atan2(delta_positions[:, 1], delta_positions[:, 0])  # Shape: (seq_len,)

    # Step 3: Unwrap angles to ensure continuity
    theta_unwrapped = torch.from_numpy(np.unwrap(theta.numpy()))
    theta_unwrapped = theta_unwrapped.float() / 10  # Adjust scaling as per original code

    # Step 4: Compute rotation velocities
    rot_vel = torch.zeros(seq_len)
    rot_vel[0] = theta_unwrapped[0]
    rot_vel[1:] = theta_unwrapped[1:] - theta_unwrapped[:-1]

    # Compute standard scaling factor
    mean_abs_rot_vel = rot_vel.abs().mean()
    standard_scaling_factor = standard_velocity_abs_mean / mean_abs_rot_vel if mean_abs_rot_vel != 0 else 1.0

    return rot_vel, standard_scaling_factor

# Function to compute root_linear_velocity using rot_vel and scaling_factor
def compute_root_linear_velocity(positions, rot_vel):
    """
    Computes the root linear velocity based on rot_vel and scaling_factor.

    Args:
        positions (Tensor): Tensor of shape (seq_len, 2), positions in x and z.
        rot_vel (Tensor): Tensor of shape (seq_len,), rotation velocity.
        scaling_factor (float): Scaling factor to apply to rot_vel.
        standard_scale_mode (bool): If True, apply standard scaling.
        scaling_factor_mode (bool): If True, apply scaling_factor.

    Returns:
        root_linear_velocity (Tensor): Tensor of shape (seq_len, 2), root linear velocity.
    """


    seq_len = positions.shape[0]

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

    return root_linear_velocity


# Paths
positions_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/interpolated_sampled_reconstructed_points_128_9.npy'
data_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy'
output_directory = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/263output_afterguidance/'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Convert numpy arrays to torch tensors with consistent dtype
rawdata = torch.tensor(np.load(data_path), dtype=torch.float32)
positions = torch.tensor(np.load(positions_path), dtype=torch.float32)


print(f"Positions shape: {positions.shape}")



# Step 1: Compute rot_vel and standard_scaling_factor
origin_rot_vel, standard_scaling_factor = compute_rot_vel_and_scaling(positions)
print(f"Standard Scaling Factor: {standard_scaling_factor}")

# Define scaling factors to batch process
scaling_factors = [standard_scaling_factor * 0.8,standard_scaling_factor, standard_scaling_factor * 3, standard_scaling_factor * 6, 1.0, 2.0]  # Example scaling factors; modify as needed
rot_vel = origin_rot_vel.clone()
# Extract original filename from positions_path
original_filename = os.path.basename(positions_path)
directory_path = os.path.dirname(positions_path)
filename_wo_ext, ext = os.path.splitext(original_filename)

for scaling_factor in scaling_factors:
    print(f"Processing with original_filename: {original_filename}")
    print(f"Processing with scaling_factor: {scaling_factor}")

    # Compute root_linear_velocity with the current scaling_factor

    scaled_rot_vel = rot_vel * scaling_factor

    root_linear_velocity = compute_root_linear_velocity(
        positions,
        scaled_rot_vel
    )


    # Update rawdata with new rot_vel and root_linear_velocity
    updated_rawdata = rawdata.clone()
    updated_rawdata[..., 0] = scaled_rot_vel   # Replace root_rot_velocity

    # 利用root_linear_velocity_standard = 0.03 标准化

    # 定义标准化的速度值
    root_linear_velocity_standard =0.036666445

    # 计算每个速度向量的模长（假设速度是二维的，可以忽略 z 分量）
    # root_linear_velocity它的形状为(N, 2)
    velocity_magnitude = np.linalg.norm(root_linear_velocity, axis=-1, keepdims=True)

    # # 标准化速度向量到 root_linear_velocity_standard
    # updated_rawdata[..., 1:3] = ( root_linear_velocity ) * root_linear_velocity_standard / velocity_magnitude
    # # updated_rawdata[..., 1:3] = root_linear_velocity / 2  # Replace root_linear_velocity
    #
    # # Construct new filename
    # new_filename = f"{filename_wo_ext}_rot_scale_{scaling_factor}{ext}"
    # new_output_path = os.path.join(output_directory, new_filename)
    #
    # # Save the updated data
    # np.save(new_output_path, updated_rawdata.numpy())
    #
    # print(f"Saved updated data to {new_output_path}")


    # 第一次赋值操作
    root_linear_velocity_scale = root_linear_velocity_standard / velocity_magnitude
    updated_rawdata[..., 1:3] = (root_linear_velocity) * root_linear_velocity_scale

    # 构建新的文件名
    new_filename_1 = f"{filename_wo_ext}_rot_scale_{scaling_factor}_root_linear_velocity_{root_linear_velocity_scale}{ext}"
    new_output_path_1 = os.path.join(output_directory, new_filename_1)

    # 保存第一个文件
    np.save(new_output_path_1, updated_rawdata.numpy())
    print(f"Saved updated data to {new_output_path_1}")

    # 第二次赋值操作
    root_linear_velocity_scale = 1.0
    updated_rawdata[..., 1:3] = (root_linear_velocity) * root_linear_velocity_scale  # Replace root_linear_velocity

    # 构建第二个文件名
    new_filename_2 = f"{filename_wo_ext}_rot_scale_{scaling_factor}_root_linear_velocity_{root_linear_velocity_scale}{ext}"
    new_output_path_2 = os.path.join(output_directory, new_filename_2)

    # 保存第二个文件
    np.save(new_output_path_2, updated_rawdata.numpy())
    print(f"Saved updated data to {new_output_path_2}")

# If you also want to handle standard_scale_mode=True, you can add similar processing here
