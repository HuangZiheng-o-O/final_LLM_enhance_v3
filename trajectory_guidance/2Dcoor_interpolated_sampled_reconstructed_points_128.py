import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def uniform_sample_points(input_filepath, output_filepath, num_points=128):
    """
    Uniformly samples points by selecting points at evenly spaced intervals to cover the entire trajectory.

    Parameters:
        input_filepath (str): Path to the input .npy file with original points.
        output_filepath (str): Path to save the uniformly sampled points as .npy file.
        num_points (int): Number of points to sample. Default is 128.
    """
    # Load original points
    points = np.load(input_filepath)

    # Calculate evenly spaced indices to select points uniformly from the entire set
    indices = np.round(np.linspace(0, len(points) - 1, num_points)).astype(int)
    sampled_points = points[indices]

    # Save sampled points
    np.save(output_filepath, sampled_points)
    print(f"Uniformly sampled points saved as {output_filepath}")


# # Redo the uniform sampling and visualize it
# uniform_output_path = '/mnt/data/uniform_sampled_reconstructed_points_128_corrected.npy'
# uniform_sample_points(input_path, uniform_output_path)
# visualize_points(input_path, uniform_output_path, 'Uniform (Corrected)')
#
# uniform_output_path


def interpolate_sample_points(input_filepath, output_filepath, num_points=128):
    """
    Interpolates points to create a specified number of uniformly spaced points along the path.

    Parameters:
        input_filepath (str): Path to the input .npy file with original points.
        output_filepath (str): Path to save the interpolated sampled points as .npy file.
        num_points (int): Number of points to sample. Default is 128.
    """
    # Load original points
    points = np.load(input_filepath)

    # Calculate cumulative distance for interpolation
    distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0)  # Include the starting point

    # Create interpolation functions for each dimension
    interp_func_x = interp1d(distances, points[:, 0], kind='linear')
    interp_func_y = interp1d(distances, points[:, 1], kind='linear')

    # Generate target distances for interpolated points
    sampled_distances = np.linspace(0, distances[-1], num_points)

    # Generate interpolated points
    sampled_points = np.column_stack((interp_func_x(sampled_distances), interp_func_y(sampled_distances)))

    # Save interpolated sampled points
    np.save(output_filepath, sampled_points)
    print(f"Interpolated sampled points saved as {output_filepath}")


def visualize_points(original_filepath, sampled_filepath, method_name):
    """
    Visualizes the original and sampled points for comparison.

    Parameters:
        original_filepath (str): Path to the .npy file with original points.
        sampled_filepath (str): Path to the .npy file with sampled points.
        method_name (str): Name of the sampling method ('Uniform' or 'Interpolated').
    """
    # Load original and sampled points
    original_points = np.load(original_filepath)
    sampled_points = np.load(sampled_filepath)

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.plot(original_points[:, 0], original_points[:, 1], 'bx-', label='Original Points', alpha=0.5)
    plt.plot(sampled_points[:, 0], sampled_points[:, 1], 'ro-',
             label=f'{method_name} Sampled Points ({len(sampled_points)})', alpha=0.8)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title(f'Comparison of Original and {method_name} Sampled Points')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage of both functions and visualization
input_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/reconstructed_points9.npy'

# # Uniform sampling
# uniform_output_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/uniform_sampled_reconstructed_points_128.npy'
# uniform_sample_points(input_path, uniform_output_path)
# visualize_points(input_path, uniform_output_path, 'Uniform')

# Interpolated sampling
interpolated_output_path = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/hearteg/interpolated_sampled_reconstructed_points_128_9.npy'
interpolate_sample_points(input_path, interpolated_output_path)
visualize_points(input_path, interpolated_output_path, 'Interpolated')

print(interpolated_output_path)
