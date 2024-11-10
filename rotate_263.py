import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from math import atan2, degrees


def compute_mean_vector(vectors):
    """
    计算给定向量的平均向量。
    参数:
    - vectors: np.ndarray, 形状为 (n, 2)
    返回:
    - mean_vector: np.ndarray, 形状为 (2,)
    """
    return np.mean(vectors, axis=0)


def compute_rotation_angle(mean_standard, mean_target):
    """
    计算将目标向量旋转到标准向量所需的角度。
    参数:
    - mean_standard: np.ndarray, 形状为 (2,)
    - mean_target: np.ndarray, 形状为 (2,)
    返回:
    - rotation_angle: float, 旋转角度（弧度）
    """
    angle_standard = atan2(mean_standard[1], mean_standard[0])
    angle_target = atan2(mean_target[1], mean_target[0])
    rotation_angle = angle_standard - angle_target

    if rotation_angle > np.pi:
        rotation_angle -= 2 * np.pi
    elif rotation_angle < -np.pi:
        rotation_angle += 2 * np.pi

    return rotation_angle


def build_rotation_matrix(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])


def rotate_root_linear_velocity(target_file_path, standard_mean, output_dir):
    try:
        target_data = np.load(target_file_path)
        target_velocity = target_data[:, 1:3]
        target_vectors_for_rotation = target_velocity[1:4]
        mean_target = compute_mean_vector(target_vectors_for_rotation)

        rotation_angle = compute_rotation_angle(standard_mean, mean_target)
        rotation_matrix = build_rotation_matrix(rotation_angle)

        target_vectors_all = target_velocity[1:]
        rotated_vectors = np.dot(target_vectors_all, rotation_matrix.T)

        rotated_data = target_data.copy()
        rotated_data[1:, 1:3] = rotated_vectors

        filename = os.path.basename(target_file_path)
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, rotated_data)
        return (target_file_path, True, "Success")
    except Exception as e:
        return (target_file_path, False, str(e))


def rotate_all_files_parallel(standard_file, target_dir, output_dir, num_workers=None):
    os.makedirs(output_dir, exist_ok=True)
    standard_data = np.load(standard_file)
    standard_velocity = standard_data[:, 1:3]
    standard_vectors_for_rotation = standard_velocity[1:4]
    standard_mean = compute_mean_vector(standard_vectors_for_rotation)

    print("金标准的平均 root_linear_velocity 向量 (rows 1:4):", standard_mean)
    print("金标准方向角度（弧度）:", atan2(standard_mean[1], standard_mean[0]))
    print("金标准方向角度（度）:", degrees(atan2(standard_mean[1], standard_mean[0])))

    all_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.npy')]
    print(f"找到 {len(all_files)} 个文件需要处理。")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        process_func = partial(rotate_root_linear_velocity, standard_mean=standard_mean, output_dir=output_dir)
        futures = {executor.submit(process_func, file_path): file_path for file_path in all_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            file_path = futures[future]
            try:
                file_path, success, message = future.result()
                if not success:
                    print(f"文件 {file_path} 处理失败: {message}")
            except Exception as exc:
                print(f"文件 {file_path} 生成异常: {exc}")

    print("所有文件处理完成。")
    print(f"旋转后的文件保存在目录: {output_dir}")


if __name__ == "__main__":
    standard_file = "/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy"
    target_dir = "/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/263output_afterguidance"
    output_dir = "/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/trajectory_guidance/263output_afterguidance_rotated3"

    rotate_all_files_parallel(standard_file, target_dir, output_dir)
