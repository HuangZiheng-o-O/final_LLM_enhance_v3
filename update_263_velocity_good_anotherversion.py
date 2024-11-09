import numpy as np
import torch

# Quaternion inversion and rotation functions
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

# Function to compute the angle between two vectors
def compute_angle_between_vectors(v1, v2, epsilon=1e-8):
    dot_product = torch.sum(v1 * v2, dim=1)
    norm_v1 = torch.norm(v1, dim=1)
    norm_v2 = torch.norm(v2, dim=1)
    cos_theta = dot_product / (norm_v1 * norm_v2 + epsilon)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    angles = torch.acos(cos_theta)
    return angles

# Define the compute_root_motion function
def compute_root_motion(positions):
    print("positions.shape:", positions.shape)
    seq_len = positions.shape[0]

    # Step 1: Compute global displacements
    delta_positions = positions[1:] - positions[:-1]
    delta_positions = torch.cat([torch.zeros(1, 2, dtype=positions.dtype, device=positions.device), delta_positions], dim=0)
    print("delta_positions:\n", delta_positions)

    # Step 2: Calculate rot_vel using compute_angle_between_vectors function
    A = delta_positions[:-1]
    B = delta_positions[1:]
    rot_vel = compute_angle_between_vectors(A, B)
    rot_vel = torch.cat([torch.zeros(1, dtype=rot_vel.dtype, device=rot_vel.device), rot_vel], dim=0)

    print("rot_vel (radians):\n", rot_vel)

    # Step 5: Compute cumulative rotation angles
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[1:] = rot_vel[:-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=0)
    print("cumulative rotation angles (radians):\n", r_rot_ang)

    # Step 6: Compute rotation quaternions
    r_rot_quat = torch.zeros(seq_len, 4, dtype=positions.dtype, device=positions.device)
    r_rot_quat[:, 0] = torch.cos(r_rot_ang)
    r_rot_quat[:, 2] = torch.sin(r_rot_ang)

    print("rotation quaternions:\n", r_rot_quat)

    # Step 7: Prepare positions in 3D
    positions_3d = torch.zeros(seq_len, 3, dtype=positions.dtype, device=positions.device)
    positions_3d[:, [0, 2]] = positions

    # Step 8: Compute delta positions in 3D
    delta_positions_3d = positions_3d[1:] - positions_3d[:-1]

    # Step 9: Rotate delta positions to local frame
    r_rot_quat_inv = qinv(r_rot_quat[1:])
    local_delta_positions = qrot(r_rot_quat_inv, delta_positions_3d)

    # Step 10: Extract root_linear_velocity (only x and z components)
    root_linear_velocity = local_delta_positions[:, [0, 2]]
    root_linear_velocity = torch.cat([torch.zeros(1, 2, dtype=root_linear_velocity.dtype, device=root_linear_velocity.device), root_linear_velocity], dim=0)

    print("root_linear_velocity:\n", root_linear_velocity)

    return rot_vel, root_linear_velocity

# 主程序部分
def main():
    positions_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/interpolated_sampled_reconstructed_points_128_9.npy'
    data_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy'
    output_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/newheart1194.npy'

    # Load dat
    rawdata_np = np.load(data_path)
    positions_np = np.load(positions_path)

    # Convert numpy arrays to torch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rawdata = torch.tensor(rawdata_np, dtype=torch.float32, device=device)
    positions = torch.tensor(positions_np, dtype=torch.float32, device=device)

    print("positions.shape:", positions.shape)

    # Compute rot_vel and root_linear_velocity from positions
    rot_vel, root_linear_velocity = compute_root_motion(positions)

    # Update rawdata
    updated_rawdata = rawdata.clone()
    updated_rawdata[..., 0] = rot_vel.cpu()
    updated_rawdata[..., 1:3] = root_linear_velocity.cpu()

    # Output and save
    print("updated_rawdata[:5]:\n", updated_rawdata[:5])
    np.save(output_path, updated_rawdata.numpy())
    print(f"Updated data saved to {output_path}")

if __name__ == "__main__":
    main()