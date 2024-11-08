import numpy as np

# Load the data
file_path = '/mnt/data/updated_data_with_rotations.npy'
data = np.load(file_path)

# Define the index range for root_linear_velocity
# Here, assuming root_linear_velocity follows root_rot_velocity,
# we select columns 1 and 2 for swapping.
root_linear_velocity_indices = (1, 2)

# Swap the x and z components of root_linear_velocity
data[:, root_linear_velocity_indices] = data[:, root_linear_velocity_indices[::-1]]

# Save the modified data
updated_file_path = '/mnt/data/updated_data_with_swapped_root_linear_velocity.npy'
np.save(updated_file_path, data)
