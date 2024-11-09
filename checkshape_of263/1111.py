
import numpy as np
import torch
file_path_new = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/checkshape_of263/extracted_root_velocity.npy'
data = np.load(file_path_new)
# Extract the second and third columns for root_linear_velocity_x and root_linear_velocity_y
root_linear_velocity_x_y = data[:, 1:3]

# Calculate the magnitude of each velocity vector
velocity_magnitude = np.linalg.norm(root_linear_velocity_x_y, axis=1)

velocity_magnitude_mean = np.mean(velocity_magnitude)

print(velocity_magnitude_mean)