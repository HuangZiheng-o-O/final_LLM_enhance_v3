# Load the newly uploaded .npy file to check its shape
file_path_new = '/mnt/data/raw_sample0_repeat0_len128.npy'
data_new = np.load(file_path_new)

# Get the shape of the data
data_new_shape = data_new.shape
data_new_shape

# Extract the first three columns for root_rot_velocity and root_linear_velocity
# Column indices: root_rot_velocity (0), root_linear_velocity (1, 2)

root_rot_velocity = data_new[:, 0].reshape(128, 1)  # Shape (128, 1)
root_linear_velocity = data_new[:, 1:3]             # Shape (128, 2)

# Combine extracted data for output in a single array if needed
extracted_data_combined = np.hstack((root_rot_velocity, root_linear_velocity))  # Shape (128, 3)

# Save extracted data as .npy
extracted_npy_path = '/mnt/data/extracted_root_velocity.npy'
np.save(extracted_npy_path, extracted_data_combined)

# Save as .xlsx for convenience
extracted_df = pd.DataFrame(extracted_data_combined, columns=["root_rot_velocity", "root_linear_velocity_x", "root_linear_velocity_y"])
extracted_xlsx_path = '/mnt/data/extracted_root_velocity.xlsx'
extracted_df.to_excel(extracted_xlsx_path, index=False)

extracted_npy_path, extracted_xlsx_path
