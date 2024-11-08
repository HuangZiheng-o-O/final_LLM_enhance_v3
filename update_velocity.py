import numpy as np


def compute_rotation_from_positions(positions):
    print(positions.shape)
    """
    Computes rotation data from sequential positions.

    Parameters:
        positions (np.ndarray): Array of shape (seq_len, 2), where each row is (x, z).

    Returns:
        np.ndarray: Rotation velocities (rot_vel) as a 2D array of shape (seq_len, 1).
        np.ndarray: Absolute rotation angles (r_rot_ang) as a 1D array of shape (seq_len,).
    """
    # Step 1: Compute movement vectors (differences between consecutive positions)
    delta_positions = positions[1:] - positions[:-1]
    print(delta_positions)

    # Step 2: Calculate movement direction angles
    theta = np.arctan2(delta_positions[:, 1], delta_positions[:, 0])
    print(theta)

    # Initialize rotation angles array with zero at the start
    r_rot_ang = np.zeros(positions.shape[0])
    r_rot_ang[1:] = theta
    # r_rot_ang = np.unwrap(r_rot_ang)  # Ensure continuity of angles

    # Step 3: Calculate rotation velocities (differences in angles)
    rot_vel = np.zeros_like(r_rot_ang)
    rot_vel[1:] = r_rot_ang[1:] - r_rot_ang[:-1]
    print("rot_vel", rot_vel)

    # Normalize rotation velocities to (-π, π)
    rot_vel = (rot_vel + np.pi) % (2 * np.pi) - np.pi

    print(rot_vel)

    return rot_vel.reshape(-1, 1), r_rot_ang  # Return rotation velocity as (seq_len, 1)


def calculate_root_linear_velocity(positions, r_rot_ang):
    """
    Calculates root linear velocity based on positions and cumulative rotation angles.

    Parameters:
        positions (np.ndarray): Array of shape (seq_len, 2), where each row is (x, z).
        r_rot_ang (np.ndarray): Array of shape (seq_len,), containing cumulative rotation angles in radians.

    Returns:
        np.ndarray: Root linear velocities as a 2D array of shape (seq_len, 2).
    """
    # Step 1: Compute global displacements between consecutive positions
    delta_positions = positions[1:] - positions[:-1]
    delta_positions = np.vstack([np.zeros((1, 2)), delta_positions])  # Pad with zeros for the first step

    # Step 2: Compute rotation matrices for -theta_{i-1} at each time step
    theta_prev = np.hstack([0, r_rot_ang[:-1]])
    cos_theta = np.cos(theta_prev)
    sin_theta = np.sin(theta_prev)

    # Rotation matrices for each step
    rotation_matrices = np.stack([
        np.stack([cos_theta, sin_theta], axis=-1),
        np.stack([-sin_theta, cos_theta], axis=-1)
    ], axis=-2)

    # Step 3: Rotate global displacements to local velocities
    root_linear_velocity = np.einsum('nij,nj->ni', rotation_matrices, delta_positions)

    return root_linear_velocity


def update_data_with_rotations(data, positions):
    """
    Updates data with calculated root rotation velocity and root linear velocity.

    Parameters:
        data (np.ndarray): Target data array of shape (seq_len, 263).
        positions (np.ndarray): Sequential (x, z) positions array of shape (seq_len, 2).

    Returns:
        np.ndarray: Updated data array with root rotation and linear velocities.
    """
    # Compute rotation velocity and cumulative rotation angles
    root_rot_velocity, r_rot_ang = compute_rotation_from_positions(positions)

    # Compute root linear velocity
    root_linear_velocity = calculate_root_linear_velocity(positions, r_rot_ang)

    # Insert root_rot_velocity and root_linear_velocity into data
    data[..., 0] = root_rot_velocity.squeeze(-1)
    data[..., 1:3] = root_linear_velocity

    return data


# Example usage
positions_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/interpolated_sampled_reconstructed_points_128_9.npy'
data_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy'
output_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/updated_data_with_rotations9.npy'

# Load positions and data
positions = np.load(positions_path)
data = np.load(data_path)

# Update data with calculated rotation and linear velocities
updated_data = update_data_with_rotations(data, positions)

# Save the modified data structure
np.save(output_path, updated_data)
print(f"Updated data saved to {output_path}")
