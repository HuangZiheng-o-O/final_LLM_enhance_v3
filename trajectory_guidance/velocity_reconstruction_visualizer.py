import numpy as np
import matplotlib.pyplot as plt
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


def visualize_reconstructed_points(filepath):
    """
    Loads and visualizes the reconstructed points from a .npy file.

    Parameters:
        filepath (str): Path to the .npy file with reconstructed points.

    Returns:
        None
    """
    try:
        # Load the reconstructed points
        reconstructed_points = np.load(filepath)

        # Visualization
        plt.figure(figsize=(8, 6))
        plt.plot(reconstructed_points[:, 0], reconstructed_points[:, 1], 'bx-', label='Reconstructed Points')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('Reconstructed Points from Velocity Vectors')
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist. Please ensure the correct filename and path.")
    except Exception as e:
        print(f"An error occurred while reading or processing the npy file: {e}")

# Example usage:
process_and_save_velocity_data('/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/Heart.npy',
                               '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/reconstructed_points9.npy')
visualize_reconstructed_points('/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/reconstructed_points9.npy')
