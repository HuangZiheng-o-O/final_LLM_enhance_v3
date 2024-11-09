import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import glob
import os

import numpy as np
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

def process_and_visualize(file_path, output_folder):
    filename = os.path.basename(file_path).replace('.npy', '_controlvelocity0dot8.npy')
    output_path = os.path.join(output_folder, filename)
    process_and_save_velocity_data(file_path, output_path)
    visualize_reconstructed_points(output_path)

def process_and_visualize_all(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files = glob.glob(f"{input_folder}/*.npy")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_and_visualize,
                file,
                output_folder
            )
            for file in files
        ]
        for future in concurrent.futures.as_completed(futures):
            if future.exception():
                print(f"Error: {future.exception()}")
            else:
                print("Processing and visualization completed for a file.")

if __name__ == "__main__":
    output_folder = '/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/npysave_controlvelocity0dot8'
    process_and_visualize_all('/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/npysave',
                              output_folder)
