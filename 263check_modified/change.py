import numpy as np
import torch
import  pandas as pd

file_path_new = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/263check_modified/V_controlvelocity0.08_interpolated_128_rot_scale_0.497_root_linear_velocity_0.207.npy'
data_new = np.load(file_path_new)

# Get the shape of the data
data_new_shape = data_new.shape
data_new_shape
#
root_linear_velocity = data_new[:, 1:3]             # Shape (128, 2)

data_new[:, 1] = 0
data_new[:, 2] = 0.3

np.save('/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/263check_modified/change.npy', data_new)
