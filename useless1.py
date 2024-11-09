import numpy as np
import torch


positions_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/interpolated_sampled/Infinity_controlvelocity0.08_interpolated_128.npy'
data_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy'
output_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/newheart11912.npy'

# positions_data_type, raw_data_type
# 结果
# (dtype('float64'), dtype('float32'))


# Convert numpy arrays to torch tensors
# Convert numpy arrays to torch tensors with consistent dtype
rawdata = torch.tensor(np.load(data_path), dtype=torch.float32)
positions = torch.tensor(np.load(positions_path), dtype=torch.float32)

print(positions.shape)
# Define the compute_root_motion function
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
    print(rot_vel.abs().mean())


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