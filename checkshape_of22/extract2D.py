import pandas as pd

# Extract the desired slice from the data
extracted_data = data[:, 1:2, :]  # Extracting shape (128, 1, 3)

# Save as .npy file
npy_output_path = '/mnt/data/extracted_data.npy'
np.save(npy_output_path, extracted_data)

# Convert to DataFrame for saving as .xlsx
# Reshape to 2D for Excel compatibility: 128 rows, 3 columns
df_extracted = pd.DataFrame(extracted_data.reshape(128, 3), columns=["x", "y", "z"])

# Save as .xlsx file
xlsx_output_path = '/mnt/data/extracted_data.xlsx'
df_extracted.to_excel(xlsx_output_path, index=False)

npy_output_path, xlsx_output_path
