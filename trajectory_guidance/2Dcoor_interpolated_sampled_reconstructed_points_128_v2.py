import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import concurrent.futures
import glob
import os


def process_and_save_velocity_data(input_filepath, output_filepath, target_mean=0.08):
    """
    Processes velocity data to control mean values, reconstructs coordinates starting from (0,0),
    and saves the result as an .npy file.

    Parameters:
        input_filepath (str): Path to the input .npy file with velocity data.
        output_filepath (str): Path to save the processed coordinates as .npy file.
        target_mean (float): Target mean value for velocity adjustment. Default is 0.08.
    """
    try:
        # Load velocity data from the input .npy file
        root_linear_velocity = np.load(input_filepath)
        velocity_vectors = root_linear_velocity.reshape(-1, 2)

        # Control the mean velocity for each direction
        mean_velocity_x = np.mean(np.abs(velocity_vectors[:, 0]))
        mean_velocity_y = np.mean(np.abs(velocity_vectors[:, 1]))
        if mean_velocity_x != 0:
            velocity_vectors[:, 0] *= target_mean / mean_velocity_x
        if mean_velocity_y != 0:
            velocity_vectors[:, 1] *= target_mean / mean_velocity_y

        # Reconstruct points starting from (0, 0)
        reconstructed_points = np.cumsum(velocity_vectors, axis=0)
        reconstructed_points = np.vstack(([0, 0], reconstructed_points))

        # Save the reconstructed points to the specified output file
        np.save(output_filepath, reconstructed_points)
        print(f"Reconstructed points saved as {output_filepath}")

    except FileNotFoundError:
        print(f"Error: The file {input_filepath} does not exist. Please ensure the correct filename and path.")
    except Exception as e:
        print(f"An error occurred while processing {input_filepath}: {e}")


def visualize_reconstructed_points(filepath, output_folder):
    """
    Visualizes the reconstructed points from a .npy file and saves the plot as a .png image.

    Parameters:
        filepath (str): Path to the .npy file with reconstructed points.
        output_folder (str): Directory to save the visualization images.
    """
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        reconstructed_points = np.load(filepath)
        plt.figure(figsize=(8, 6))
        plt.plot(reconstructed_points[:, 0], reconstructed_points[:, 1], 'bx-', label='Reconstructed Points')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('Reconstructed Points from Velocity Vectors')
        plt.legend()
        plt.grid(True)

        filename = os.path.basename(filepath).replace('.npy', '.png')
        save_path = os.path.join(output_folder, filename)
        plt.savefig(save_path)
        plt.close()  # 关闭图形以释放内存
        print(f"Visualization saved as {save_path}")

    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")
    except Exception as e:
        print(f"An error occurred while visualizing {filepath}: {e}")


def process_and_visualize(file_path, output_npy_folder, output_image_folder, target_mean=0.08):
    """
    Processes a single .npy file and visualizes the reconstructed points.

    Parameters:
        file_path (str): Path to the input .npy file with velocity data.
        output_npy_folder (str): Directory to save the processed .npy files.
        output_image_folder (str): Directory to save the visualization images.
        target_mean (float): Target mean value for velocity adjustment.

    Returns:
        None
    """
    try:
        if not os.path.exists(output_npy_folder):
            os.makedirs(output_npy_folder)

        # Generate output .npy filename
        base_filename = os.path.basename(file_path).replace('.npy', f'_controlvelocity{target_mean}.npy')
        output_npy_path = os.path.join(output_npy_folder, base_filename)

        # Process and save velocity data
        process_and_save_velocity_data(file_path, output_npy_path, target_mean)

        # Visualize the reconstructed points
        visualize_reconstructed_points(output_npy_path, output_image_folder)

    except Exception as e:
        print(f"An error occurred in processing {file_path}: {e}")


def process_and_visualize_all(input_folder, output_npy_folder, output_image_folder, target_mean=0.08, max_workers=None):
    """
    Processes and visualizes all .npy files in the specified input folder in parallel.

    Parameters:
        input_folder (str): Directory containing input .npy files.
        output_npy_folder (str): Directory to save the processed .npy files.
        output_image_folder (str): Directory to save the visualization images.
        target_mean (float): Target mean value for velocity adjustment.
        max_workers (int, optional): Maximum number of worker processes. Defaults to the number of processors on the machine.

    Returns:
        None
    """
    try:
        if not os.path.exists(output_npy_folder):
            os.makedirs(output_npy_folder)
        if not os.path.exists(output_image_folder):
            os.makedirs(output_image_folder)

        # Find all .npy files in the input folder
        files = glob.glob(os.path.join(input_folder, "*.npy"))
        if not files:
            print(f"No .npy files found in {input_folder}.")
            return

        print(f"Found {len(files)} .npy files in {input_folder}. Starting processing...")

        # Use ProcessPoolExecutor for CPU-bound tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_and_visualize,
                    file,
                    output_npy_folder,
                    output_image_folder,
                    target_mean
                )
                for file in files
            ]
            for future in concurrent.futures.as_completed(futures):
                if future.exception():
                    print(f"Error during processing: {future.exception()}")
                else:
                    print("Processing and visualization completed for a file.")

        print("All files have been processed and visualized.")

    except Exception as e:
        print(f"An error occurred in process_and_visualize_all: {e}")


def uniform_sample_points(input_filepath, output_filepath, num_points=128):
    """
    Uniformly samples points by selecting points at evenly spaced intervals to cover the entire trajectory.

    Parameters:
        input_filepath (str): Path to the input .npy file with original points.
        output_filepath (str): Path to save the uniformly sampled points as .npy file.
        num_points (int): Number of points to sample. Default is 128.
    """
    try:
        # Load original points
        points = np.load(input_filepath)

        if len(points) < 2:
            print(f"Not enough points in {input_filepath} to perform uniform sampling.")
            return

        # Calculate evenly spaced indices to select points uniformly from the entire set
        indices = np.round(np.linspace(0, len(points) - 1, num_points)).astype(int)
        sampled_points = points[indices]

        # Save sampled points
        np.save(output_filepath, sampled_points)
        print(f"Uniformly sampled points saved as {output_filepath}")
    except Exception as e:
        print(f"An error occurred during uniform sampling for {input_filepath}: {e}")


def interpolate_sample_points(input_filepath, output_filepath, num_points=128):
    """
    Interpolates points to create a specified number of uniformly spaced points along the path.

    Parameters:
        input_filepath (str): Path to the input .npy file with original points.
        output_filepath (str): Path to save the interpolated sampled points as .npy file.
        num_points (int): Number of points to sample. Default is 128.
    """
    try:
        # Load original points
        points = np.load(input_filepath)

        if len(points) < 2:
            print(f"Not enough points in {input_filepath} to perform interpolation.")
            return

        # Calculate cumulative distance for interpolation
        deltas = np.diff(points, axis=0)
        distances = np.sqrt((deltas ** 2).sum(axis=1))
        cumulative_distances = np.cumsum(distances)
        cumulative_distances = np.insert(cumulative_distances, 0, 0)  # Include the starting point

        # Create interpolation functions for each dimension
        interp_func_x = interp1d(cumulative_distances, points[:, 0], kind='linear')
        interp_func_y = interp1d(cumulative_distances, points[:, 1], kind='linear')

        # Generate target distances for interpolated points
        sampled_distances = np.linspace(0, cumulative_distances[-1], num_points)

        # Generate interpolated points
        sampled_points = np.column_stack((interp_func_x(sampled_distances), interp_func_y(sampled_distances)))

        # Save interpolated sampled points
        np.save(output_filepath, sampled_points)
        print(f"Interpolated sampled points saved as {output_filepath}")
    except Exception as e:
        print(f"An error occurred during interpolation for {input_filepath}: {e}")


def visualize_points(original_filepath, sampled_filepath, method_name, output_image_folder):
    """
    Visualizes the original and sampled points for comparison.

    Parameters:
        original_filepath (str): Path to the .npy file with original points.
        sampled_filepath (str): Path to the .npy file with sampled points.
        method_name (str): Name of the sampling method ('Uniform' or 'Interpolated').
        output_image_folder (str): Directory to save the visualization images.
    """
    try:
        if not os.path.exists(output_image_folder):
            os.makedirs(output_image_folder)

        # Load original and sampled points
        original_points = np.load(original_filepath)
        sampled_points = np.load(sampled_filepath)

        plt.figure(figsize=(8, 6))
        plt.plot(original_points[:, 0], original_points[:, 1], 'bx-', label='Original Points', alpha=0.5)
        plt.plot(sampled_points[:, 0], sampled_points[:, 1], 'ro-',
                 label=f'{method_name} Sampled Points ({len(sampled_points)})', alpha=0.8)
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title(f'Comparison of Original and {method_name} Sampled Points')
        plt.legend()
        plt.grid(True)

        # Generate save path
        original_filename = os.path.basename(original_filepath).replace('.npy', '')
        sampled_filename = os.path.basename(sampled_filepath).replace('.npy', '')
        save_filename = f"{original_filename}_vs_{sampled_filename}.png"
        save_path = os.path.join(output_image_folder, save_filename)
        plt.savefig(save_path)
        plt.close()  # 关闭图形以释放内存
        print(f"Sampling comparison visualization saved as {save_path}")

    except Exception as e:
        print(f"An error occurred while visualizing sampling for {original_filepath}: {e}")


def process_sampling_and_visualize(file_path, output_npy_folder, output_image_folder, sampling_method='interpolated',
                                   num_points=128):
    """
    Processes sampling (uniform or interpolated) for a single .npy file and visualizes the result.

    Parameters:
        file_path (str): Path to the input .npy file with original points.
        output_npy_folder (str): Directory to save the sampled .npy files.
        output_image_folder (str): Directory to save the visualization images.
        sampling_method (str): 'uniform' or 'interpolated'.
        num_points (int): Number of points to sample. Default is 128.

    Returns:
        None
    """
    try:
        if not os.path.exists(output_npy_folder):
            os.makedirs(output_npy_folder)
        if not os.path.exists(output_image_folder):
            os.makedirs(output_image_folder)

        base_filename = os.path.basename(file_path).replace('.npy', '')

        if sampling_method.lower() == 'uniform':
            sampled_filename = f"{base_filename}_uniform_{num_points}.npy"
            sampled_path = os.path.join(output_npy_folder, sampled_filename)
            uniform_sample_points(file_path, sampled_path, num_points)
            visualize_points(file_path, sampled_path, 'Uniform', output_image_folder)

        elif sampling_method.lower() == 'interpolated':
            sampled_filename = f"{base_filename}_interpolated_{num_points}.npy"
            sampled_path = os.path.join(output_npy_folder, sampled_filename)
            interpolate_sample_points(file_path, sampled_path, num_points)
            visualize_points(file_path, sampled_path, 'Interpolated', output_image_folder)

        else:
            print(f"Unknown sampling method: {sampling_method}. Skipping file {file_path}.")

    except Exception as e:
        print(f"An error occurred in sampling and visualization for {file_path}: {e}")


def process_sampling_and_visualize_all(input_folder, output_sampled_folder, output_sampling_image_folder,
                                       sampling_method='interpolated', num_points=128, max_workers=None):
    """
    Processes sampling (uniform or interpolated) and visualizes all .npy files in the specified input folder in parallel.

    Parameters:
        input_folder (str): Directory containing input .npy files.
        output_sampled_folder (str): Directory to save the sampled .npy files.
        output_sampling_image_folder (str): Directory to save the sampling visualization images.
        sampling_method (str): 'uniform' or 'interpolated'. Default is 'interpolated'.
        num_points (int): Number of points to sample. Default is 128.
        max_workers (int, optional): Maximum number of worker processes. Defaults to the number of processors on the machine.

    Returns:
        None
    """
    try:
        if not os.path.exists(output_sampled_folder):
            os.makedirs(output_sampled_folder)
        if not os.path.exists(output_sampling_image_folder):
            os.makedirs(output_sampling_image_folder)

        # Find all .npy files in the input folder
        files = glob.glob(os.path.join(input_folder, "*.npy"))
        if not files:
            print(f"No .npy files found in {input_folder}.")
            return

        print(f"Found {len(files)} .npy files in {input_folder}. Starting sampling and visualization...")

        # Use ProcessPoolExecutor for CPU-bound tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_sampling_and_visualize,
                    file,
                    output_sampled_folder,
                    output_sampling_image_folder,
                    sampling_method,
                    num_points
                )
                for file in files
            ]
            for future in concurrent.futures.as_completed(futures):
                if future.exception():
                    print(f"Error during sampling and visualization: {future.exception()}")
                else:
                    print("Sampling and visualization completed for a file.")

        print("All files have been sampled and visualized.")

    except Exception as e:
        print(f"An error occurred in process_sampling_and_visualize_all: {e}")


if __name__ == "__main__":
    # 第一部分：处理和可视化速度数据
    output_npy_folder_velocity = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/npysave_controlvelocity0dot8'
    output_image_folder_velocity = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/npysave_controlvelocity0dot8_images'

    input_folder_velocity = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/npysave'

    # 开始批量处理和可视化速度数据
    process_and_visualize_all(
        input_folder=input_folder_velocity,
        output_npy_folder=output_npy_folder_velocity,
        output_image_folder=output_image_folder_velocity,
        target_mean=0.08
    )

    # 第二部分：统一采样和插值采样
    output_sampled_folder_uniform = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/uniform_sampled'
    output_sampling_image_folder_uniform = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/uniform_sampled_images'

    output_sampled_folder_interpolated = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/interpolated_sampled'
    output_sampling_image_folder_interpolated = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/interpolated_sampled_images'

    # 统一采样
    process_sampling_and_visualize_all(
        input_folder=output_npy_folder_velocity,
        output_sampled_folder=output_sampled_folder_uniform,
        output_sampling_image_folder=output_sampling_image_folder_uniform,
        sampling_method='uniform',
        num_points=128
    )

    # 插值采样
    process_sampling_and_visualize_all(
        input_folder=output_npy_folder_velocity,
        output_sampled_folder=output_sampled_folder_interpolated,
        output_sampling_image_folder=output_sampling_image_folder_interpolated,
        sampling_method='interpolated',
        num_points=128
    )
