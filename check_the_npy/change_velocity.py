import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load the .npy file
file_path = '../S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy'  # Replace with your file path
data_structure = np.load(file_path, allow_pickle=True)
print(np.shape(data_structure))

# Extract root_linear_velocity from data_structure
root_linear_velocity = data_structure[:, 1:3]  # Shape should be (seq_len, 2)

# Load the npy file containing velocity changes
npysave_path = '../trajectory_guidance/npysave/S_shape.npy'
velocity_change_npy = np.load(npysave_path, allow_pickle=True)
print(np.shape(velocity_change_npy))  # Expected shape: (1, 199, 2)

# Extract the velocity data (remove the first dimension)
velocity_data = velocity_change_npy[0, :, :]  # Shape: (199, 2)

# **Step 1:** Interpolate the 199 points to get a new array of shape (128, 2)
original_indices = np.linspace(0, 1, num=199)
new_indices = np.linspace(0, 1, num=128)

# Interpolate x and y separately
interp_func_x = interp1d(original_indices, velocity_data[:, 0], kind='linear')
interp_func_y = interp1d(original_indices, velocity_data[:, 1], kind='linear')

# Create the new velocity array with shape (128, 2)
velocity_change_npy_new = np.zeros((128, 2))
velocity_change_npy_new[:, 0] = interp_func_x(new_indices)
velocity_change_npy_new[:, 1] = interp_func_y(new_indices)

# **Step 2:** Two choices for mapping and scaling

# Get min and max of root_linear_velocity for scaling
min_root_x, max_root_x = np.min(root_linear_velocity[:, 0]), np.max(root_linear_velocity[:, 0])
min_root_y, max_root_y = np.min(root_linear_velocity[:, 1]), np.max(root_linear_velocity[:, 1])

# --- Choice 1 ---
# Map x and y of velocity_change_npy_new to x and y of root_linear_velocity
choice1 = velocity_change_npy_new.copy()

# Scale x
min_choice1_x, max_choice1_x = np.min(choice1[:, 0]), np.max(choice1[:, 0])
choice1[:, 0] = (choice1[:, 0] - min_choice1_x) / (max_choice1_x - min_choice1_x)
choice1[:, 0] = choice1[:, 0] * (max_root_x - min_root_x) + min_root_x

# Scale y
min_choice1_y, max_choice1_y = np.min(choice1[:, 1]), np.max(choice1[:, 1])
choice1[:, 1] = (choice1[:, 1] - min_choice1_y) / (max_choice1_y - min_choice1_y)
choice1[:, 1] = choice1[:, 1] * (max_root_y - min_root_y) + min_root_y

# --- Choice 2 ---
# Swap x and y, then map to root_linear_velocity's x and y
choice2 = velocity_change_npy_new.copy()
choice2 = choice2[:, [1, 0]]  # Swap x and y

# Scale x (which was originally y)
min_choice2_x, max_choice2_x = np.min(choice2[:, 0]), np.max(choice2[:, 0])
choice2[:, 0] = (choice2[:, 0] - min_choice2_x) / (max_choice2_x - min_choice2_x)
choice2[:, 0] = choice2[:, 0] * (max_root_x - min_root_x) + min_root_x

# Scale y (which was originally x)
min_choice2_y, max_choice2_y = np.min(choice2[:, 1]), np.max(choice2[:, 1])
choice2[:, 1] = (choice2[:, 1] - min_choice2_y) / (max_choice2_y - min_choice2_y)
choice2[:, 1] = choice2[:, 1] * (max_root_y - min_root_y) + min_root_y


# **Visualization of both choices**
def reconstruct_trajectory(velocity_data, title):
    # Reconstruct the trajectory from velocity data
    reconstructed_points = [(0, 0)]
    for vx, vy in velocity_data:
        last_point = reconstructed_points[-1]
        new_point = (last_point[0] + vx, last_point[1] + vy)
        reconstructed_points.append(new_point)
    reconstructed_points = np.array(reconstructed_points)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(reconstructed_points[:, 0], reconstructed_points[:, 1], 'bx-', label='Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Visualize Choice 1
reconstruct_trajectory(choice1, 'Reconstructed Trajectory from Choice 1')

reconstruct_trajectory(choice2, 'Reconstructed Trajectory from Choice 2')

# **Replace the corresponding data in data_structure and save to new .npy files**

# For Choice 1
data_structure_choice1 = data_structure.copy()
data_structure_choice1[:, 1:3] = choice1
np.save('data_structure_choice1.npy', data_structure_choice1)

# For Choice 2
data_structure_choice2 = data_structure.copy()
data_structure_choice2[:, 1:3] = choice2
np.save('data_structure_choice2.npy', data_structure_choice2)
