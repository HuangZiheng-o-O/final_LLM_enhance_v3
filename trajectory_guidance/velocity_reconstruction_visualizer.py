import numpy as np
import matplotlib.pyplot as plt

import concurrent.futures
import glob
import os
"""
该代码的主要功能是处理并保存速度数据，以控制平均速度值，并从给定的速度向量重建坐标点。具体步骤如下：

process_and_save_velocity_data 函数：该函数从一个包含速度数据的 .npy 文件中加载数据，控制速度向量的均值，重构坐标点，并将结果保存到另一个 .npy 文件中。其步骤如下：

加载输入文件中的速度数据，并将其转换为二维速度向量。
控制速度的均值，使其与给定的目标均值一致。通过计算x和y方向的平均速度，并按比例调整每个方向的速度。
通过累加速度值从原点 (0, 0) 重构每个坐标点，生成路径。
将重构后的点保存到指定的输出文件中。
visualize_reconstructed_points 函数：该函数从一个 .npy 文件中加载重构后的坐标点并进行可视化。其步骤如下：

加载文件中的重构点数据。
使用 matplotlib 绘制重构后的坐标点路径图，将 x 轴作为 X 坐标，y 轴作为 Z 坐标。
显示坐标路径图，方便用户观察重构后的路径。
使用示例：调用 process_and_save_velocity_data 函数处理原始速度数据并保存结果，然后使用 visualize_reconstructed_points 函数来可视化处理后的坐标路径。

"""

def process_and_save_velocity_data(input_filepath, output_filepath, target_mean=0.08):
    """
    Processes velocity data to control mean values, reconstructs coordinates starting from (0,0),
    and saves the result as an .npy file.

    Parameters:
        input_filepath (str): Path to the input .npy file with velocity data.
        output_filepath (str): Path to save the processed coordinates as .npy file.
        target_mean (float): Target mean value for velocity adjustment. Default is 0.02.

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
        reconstructed_points = [(0, 0)]
        for vx, vy in velocity_vectors:
            last_point = reconstructed_points[-1]
            new_point = (last_point[0] + vx, last_point[1] + vy)
            reconstructed_points.append(new_point)

        # Convert list to numpy array for saving
        reconstructed_points = np.array(reconstructed_points)

        # Save the reconstructed points to the specified output file
        np.save(output_filepath, reconstructed_points)
        print(f"Reconstructed points saved as {output_filepath}")

    except FileNotFoundError:
        print(f"Error: The file {input_filepath} does not exist. Please ensure the correct filename and path.")
    except Exception as e:
        print(f"An error occurred while reading or processing the npy file: {e}")


def visualize_reconstructed_points(filepath, output_folder='/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/npysave_controlvelocity_image'):
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
        plt.show()
        filename = os.path.basename(filepath).replace('.npy', '')
        plt.savefig(os.path.join(output_folder, f"{filename}.png"))
    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# # Example usage:
# process_and_save_velocity_data('/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/Heart.npy',
#                                '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/reconstructed_points9.npy')
# visualize_reconstructed_points('/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/reconstructed_points9.npy')


def process_and_visualize(file_path, output_path):
    process_and_save_velocity_data(file_path, output_path)
    visualize_reconstructed_points(output_path)

def process_and_visualize_all(input_folder, output_folder):
    files = glob.glob(f"{input_folder}/*.npy")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_and_visualize,
                file,
                f"{output_folder}/npysave_controlvelocity0dot8"
            )
            for file in files
        ]
        for future in concurrent.futures.as_completed(futures):
            if future.exception():
                print(f"Error: {future.exception()}")
            else:
                print("Processing and visualization completed for a file.")

if __name__ == "__main__":
    # import concurrent.futures
    # import glob
    # import os

    output_folder = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/npysave_controlvelocity0dot8'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    else:
        print(f"Output directory exists: {output_folder}")


    process_and_visualize_all('/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/npysave',
         '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/npysave_controlvelocity0dot8')
