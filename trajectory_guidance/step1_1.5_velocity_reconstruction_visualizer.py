import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import glob
import os

"""

该代码提供了批量处理和可视化速度数据的功能。
它首先加载和调整速度数据的均值，然后根据调整后的速度重建坐标，并保存为 .npy 文件。
接着，生成相应的可视化图像并保存为 .png 文件。
代码支持对多个文件的并行处理，提高处理效率，适用于大规模数据集。
target_mean=0.08

"""
def process_and_save_velocity_data(input_filepath, output_filepath, target_mean=0.08):
    """
    Processes velocity data to control mean values, reconstructs coordinates starting from (0,0),
    and saves the result as an .npy file.

    Parameters:
    input_filepath (str): Path to the input .npy file with velocity data.
    output_filepath (str): Path to save the processed coordinates as .npy file.
    target_mean (float): Target mean value for velocity adjustment. Default is 0.08.

    Returns:
    None
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

    Returns:
    None
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


if __name__ == "__main__":

    # 开始批量处理和可视化
    process_and_visualize_all(
        input_folder = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/npysave',
        output_npy_folder = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/npysave_controlvelocity0dot8',
        output_image_folder = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/npysave_controlvelocity0dot8_images',
        target_mean=0.08
    )
