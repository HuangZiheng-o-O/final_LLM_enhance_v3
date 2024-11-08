import numpy as np
import torch
from torch.onnx.symbolic_opset9 import tensor

# %%
import torch

positions = torch.tensor([[ 0.06,  0.02],
                          [ 0.05,  0.05],
                          [ 0.041,  0.08],
                          [ 0.03,  0.11]])

print(positions)
# %%
"""
positions: Tensor of shape (seq_len, 2), positions in x and z
Returns:
rot_vel: Tensor of shape (seq_len,), rotation velocities
root_linear_velocity: Tensor of shape (seq_len, 2), velocities in local frame
"""

seq_len = positions.shape[0]

# Step 1: Compute global displacements
delta_positions = positions[1:] - positions[:-1]  # Shape: (seq_len - 1, 2)
print(delta_positions)
# %%


delta_positions = torch.cat([torch.zeros(1, 2), delta_positions], dim=0)  # Pad to match seq_len

print(delta_positions)
# tensor([[ 0.0000,  0.0000],
#         [-0.0100,  0.0300],
#         [-0.0090,  0.0300],
#         [-0.0110,  0.0300]])

# %%
# Step 2: 计算delta_positions每一个后面的向量和前一个夹脚
# [-0.0100,  0.0300]和[ 0.0000,  0.0000]的夹角
# [-0.0090,  0.0300]和[-0.0100,  0.0300]的夹角
# [-0.0110,  0.0300]和[-0.0090,  0.0300]的夹角
# ...


theta = torch.atan2(delta_positions[:, 1], delta_positions[:, 0])  # Shape: (seq_len,)

# Step 2: Unwrap angles to ensure continuity
theta_unwrapped = torch.from_numpy(np.unwrap(theta.numpy()))

# Step 4: Compute rotation velocities
rot_vel = torch.zeros(seq_len)
rot_vel[1:] = theta_unwrapped[1:] - theta_unwrapped[:-1]
rot_vel[0] = 0  # First frame has no rotation velocity

# Step 5: Compute previous cumulative rotation angles
theta_prev = torch.zeros(seq_len)
theta_prev[1:] = theta_unwrapped[:-1]

# Step 6: Compute rotation matrices for -theta_prev
cos_theta = torch.cos(-theta_prev)
sin_theta = torch.sin(-theta_prev)

rotation_matrices = torch.stack([
    torch.stack([cos_theta, -sin_theta], dim=-1),
    torch.stack([sin_theta, cos_theta], dim=-1)
], dim=-2)  # Shape: (seq_len, 2, 2)

# Step 7: Rotate global displacements to local velocities
root_linear_velocity = torch.einsum('nij,nj->ni', rotation_matrices, delta_positions)
print("rot_vel", rot_vel)
print("root_linear_velocity", root_linear_velocity)


rot_vel, root_linear_velocity


# Example usage
positions_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/checkshape_of22/extracted_data.npy'
data_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy'
output_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/updated_data_with_rotations9.npy'

import numpy as np
import torch

# # 假设 positions 是 (seq_len, 2) 的张量，包含 128 个 x, z 坐标
# positions = torch.tensor([...])  # 您的 x, z 坐标数据
# Load positions and data
positions = np.load(positions_path)
positions = positions[:,:,[0,2]]
# from # (128, 1, 2) to # (128,2)
positions = positions.squeeze(1)
# 将 positions 转换为 PyTorch Tensor
positions = torch.tensor(positions, dtype=torch.float32)


# 计算旋转速度和局部线速度
rot_vel, root_linear_velocity = compute_root_motion(positions)

# # 初始化 data 数组，形状为 (seq_len, 263)
# data = torch.zeros(128, 263)
#
# # 插入 root_rot_velocity
# data[..., 0] = rot_vel
#
# # 插入 root_linear_velocity
# data[..., 1:3] = root_linear_velocity
#
# # 插入 root_y（假设为常数或已知值）
# data[..., 3] = root_y_value  # root_y_value 是一个标量，或形状为 (seq_len,) 的张量
#
# # 后续处理，如插入关节数据、可视化等
