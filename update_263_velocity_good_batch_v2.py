import os
import numpy as np
import torch

# Quaternion inversion function
def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (..., 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask

# Quaternion rotation function
def qrot(q, v):
    assert q.shape[-1] == 4, 'q must have last dimension of size 4'
    assert v.shape[-1] == 3, 'v must have last dimension of size 3'
    assert q.shape[:-1] == v.shape[:-1], 'q and v must have the same shape except for last dimension'

    original_shape = v.shape
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    rotated_v = v + 2 * (q[:, :1] * uv + uuv)
    return rotated_v.view(original_shape)

# Function to compute rot_vel and standard_scaling_factor
def compute_rot_vel_and_scaling(positions, standard_velocity_abs_mean=0.015):
    seq_len = positions.shape[0]
    delta_positions = positions[1:] - positions[:-1]
    delta_positions = torch.cat([torch.zeros(1, 2), delta_positions], dim=0)
    theta = torch.atan2(delta_positions[:, 1], delta_positions[:, 0])
    theta_unwrapped = torch.from_numpy(np.unwrap(theta.numpy()))
    theta_unwrapped = theta_unwrapped.float()

    rot_vel = torch.zeros(seq_len)
    rot_vel[0] = theta_unwrapped[0]
    rot_vel[1:] = theta_unwrapped[1:] - theta_unwrapped[:-1]

    mean_abs_rot_vel = rot_vel.abs().mean()
    print("standard_velocity_abs_mean", standard_velocity_abs_mean,"mean_abs_rot_vel", mean_abs_rot_vel)
    standard_scaling_factor = standard_velocity_abs_mean / mean_abs_rot_vel if mean_abs_rot_vel != 0 else 1.0

    return rot_vel, standard_scaling_factor

# Function to compute root_linear_velocity using rot_vel and scaling_factor
def compute_root_linear_velocity(positions, rot_vel):
    seq_len = positions.shape[0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[1:] = rot_vel[:-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=0)

    r_rot_quat = torch.zeros(seq_len, 4)
    r_rot_quat[:, 0] = torch.cos(r_rot_ang/2)
    r_rot_quat[:, 2] = torch.sin(r_rot_ang/2)

    positions_3d = torch.zeros(seq_len, 3)
    positions_3d[:, [0, 2]] = positions

    delta_positions_3d = positions_3d[1:] - positions_3d[:-1]
    r_rot_quat_inv = qinv(r_rot_quat[1:])
    local_delta_positions = qrot(r_rot_quat_inv, delta_positions_3d)

    root_linear_velocity = local_delta_positions[:, [0, 2]]
    root_linear_velocity = torch.cat([torch.zeros(1, 2), root_linear_velocity], dim=0)

    return root_linear_velocity

# Main processing function
def process_and_save_scaled_velocity(positions_path, data_path, output_directory, standard_velocity_abs_mean=0.015):
    os.makedirs(output_directory, exist_ok=True)
    rawdata = torch.tensor(np.load(data_path), dtype=torch.float32)
    positions = torch.tensor(np.load(positions_path), dtype=torch.float32)

    origin_rot_vel, standard_scaling_factor = compute_rot_vel_and_scaling(positions, standard_velocity_abs_mean)

    #    standard_scaling_factor = standard_velocity_abs_mean 0.015 / mean_abs_rot_vel if mean_abs_rot_vel != 0 else 1.0
    print("standard_scaling_factor", standard_scaling_factor)
    if standard_scaling_factor <= 0.12:
        scaling_factors = [standard_scaling_factor, standard_scaling_factor * 3,
                           standard_scaling_factor * 6, 1.0]
    elif standard_scaling_factor <0.2:
        scaling_factors = [standard_scaling_factor, standard_scaling_factor * 3, 1.0]
    elif standard_scaling_factor <0.5:
        scaling_factors = [ standard_scaling_factor, 1.0]
    elif standard_scaling_factor >1 and standard_scaling_factor <1.5:
        scaling_factors = [1.0]
    else:
        scaling_factors = [standard_scaling_factor, 1.0]

    original_filename = os.path.basename(positions_path)
    filename_wo_ext, ext = os.path.splitext(original_filename)

    for scaling_factor in scaling_factors:
        scaled_rot_vel = origin_rot_vel * scaling_factor
        root_linear_velocity = compute_root_linear_velocity(positions, scaled_rot_vel)

        updated_rawdata = rawdata.clone()
        updated_rawdata[..., 0] = scaled_rot_vel

        root_linear_velocity_standard = 0.036666445
        velocity_magnitude = np.linalg.norm(root_linear_velocity, axis=-1, keepdims=True)
        velocity_magnitude_absmean = np.mean(np.abs(velocity_magnitude))
        root_linear_velocity_scalestandard = root_linear_velocity_standard / velocity_magnitude_absmean

        # updated_rawdata[..., 1:3] = root_linear_velocity * root_linear_velocity_scale
        # new_filename_1 = f"{filename_wo_ext}_rot_scale_{scaling_factor}_root_linear_velocity_{root_linear_velocity_scale}{ext}"
        # new_output_path_1 = os.path.join(output_directory, new_filename_1)
        # np.save(new_output_path_1, updated_rawdata.numpy())
        # print(f"Saved updated data to {new_output_path_1}")
        #
        # root_linear_velocity_scale = 1.0
        # updated_rawdata[..., 1:3] = root_linear_velocity * root_linear_velocity_scale
        # new_filename_2 = f"{filename_wo_ext}_rot_scale_{scaling_factor}_root_linear_velocity_{root_linear_velocity_scale}{ext}"
        # new_output_path_2 = os.path.join(output_directory, new_filename_2)
        # np.save(new_output_path_2, updated_rawdata.numpy())
        # print(f"Saved updated data to {new_output_path_2}")
        #

        root_linear_velocity_scales = [root_linear_velocity_scalestandard,1.0]  # 列出不同的 root_linear_velocity_scale 值

        for scale in root_linear_velocity_scales:
            if scale == standard_scaling_factor:
                new_filename = f"{filename_wo_ext}_rot_scale_{scaling_factor:.3f}_root_linear_velocity_{scale:.3f}_using_standard_scale_root_linear{ext}"
            elif scale  == 1.0:
                new_filename = f"{filename_wo_ext}_rot_scale_{scaling_factor:.3f}_root_linear_velocity_{scale:.3f}_not_scale_root_linear{ext}"
            else:
                new_filename = f"{filename_wo_ext}_rot_scale_{scaling_factor:.3f}_root_linear_velocity_{scale:.3f}{ext}"
            updated_rawdata[..., 1:3] = root_linear_velocity * scale

            new_output_path = os.path.join(output_directory, new_filename)
            np.save(new_output_path, updated_rawdata.numpy())
            print(f"Saved updated data to {new_output_path}")


# # Run the function with your specified paths
# positions_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/interpolated_sampled/Infinity_controlvelocity0.08_interpolated_128.npy'
# data_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy'
# output_directory = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/263output_afterguidance/'
#
# process_and_save_scaled_velocity(positions_path, data_path, output_directory)


import concurrent.futures
from glob import glob

def main():
    # Define directories
    positions_directory = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/interpolated_sampled/'
    aaaa = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy'
    data_directory = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/raw/'
    output_directory = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/263output_afterguidance/'

    # Find all positions files
    positions_files = glob(os.path.join(positions_directory, '*.npy'))

    # Assuming that data files have a corresponding name to positions files.
    # Modify the mapping logic below if your data files have a different naming convention.
    tasks = []
    for pos_path in positions_files:
        filename = os.path.basename(pos_path)


        # # Example mapping: Replace 'interpolated' with 'raw_sample' or adjust as needed
        # # Modify this according to your actual naming convention
        # # For instance, if positions file is 'Infinity_controlvelocity0.08_interpolated_128.npy'
        # # and data file is 'raw_sample0_repeat0_len128.npy', you need a way to map them
        # # Here, we'll assume they share the same base name before certain patterns
        # # You may need to adjust this part based on your actual filenames
        #
        # # Example: Extracting a unique identifier from the positions filename
        # # Adjust this parsing as per your filename structure
        # base_name = filename.replace('_interpolated', '').replace('.npy', '')
        # # Now, find the corresponding data file
        # # This is a placeholder logic; adjust according to your actual file naming
        #
        # data_files = glob(os.path.join(data_directory, f"*{base_name}*.npy"))

        data_files = glob(os.path.join(data_directory, "*.npy"))

        if not data_files:
            print(f"No corresponding data file found for positions file: {pos_path}")
            continue
        data_path = data_files[0]  # Take the first match; adjust if multiple matches are possible
        tasks.append((pos_path, data_path, output_directory))

    # Define the number of workers; adjust as per your CPU cores
    max_workers = min(32, os.cpu_count() + 4)  # Example: Adjust based on your system

    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task in tasks:
            positions_path, data_path, out_dir = task
            futures.append(executor.submit(process_and_save_scaled_velocity, positions_path, data_path, out_dir))

        # Optionally, you can monitor the progress
        for future in concurrent.futures.as_completed(futures):
            future.result()
            # try:
            #     future.result()
            # except Exception as exc:
            #     print(f"Generated an exception: {exc}")

if __name__ == "__main__":
    main()