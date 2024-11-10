import numpy as np
import torch
import pandas as pd
file_path_new = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/checkV/Triangle_controlvelocity0.08_interpolated_128_rot_scale_0.327_root_linear_velocity_0.183.npy'
data_new = np.load(file_path_new)

# Get the shape of the data
data_new_shape = data_new.shape
data_new_shape
#%%
# Extract the first three columns for root_rot_velocity and root_linear_velocity
# Column indices: root_rot_velocity (0), root_linear_velocity (1, 2)

root_rot_velocity = data_new[:, 0].reshape(128, 1)  # Shape (128, 1)

root_rot_velocity

#%%
root_linear_velocity = data_new[:, 1:3]             # Shape (128, 2)
root_linear_velocity
#%%
# Combine extracted data for output in a single array if needed
extracted_data_combined = np.hstack((root_rot_velocity, root_linear_velocity))  # Shape (128, 3)
extracted_data_combined
#%%
# Save extracted data as .npy
extracted_npy_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/checkV/extracted_root_velocity_triangle.npy'
np.save(extracted_npy_path, extracted_data_combined)

# Save as .xlsx for convenience
extracted_df = pd.DataFrame(extracted_data_combined, columns=["root_rot_velocity", "root_linear_velocity_x", "root_linear_velocity_y"])
extracted_xlsx_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/checkV/extracted_root_velocity_triangle.xlsx'
extracted_df.to_excel(extracted_xlsx_path, index=False)

extracted_npy_path, extracted_xlsx_path
