import os
from trajectory_guidance.step1 import step1_generate_velocity
from trajectory_guidance.step2_2Dcoor_interpolated_sampled_reconstructed_points_128_v2 import \
    step2_process_velocity_data_main_func
from trajectory_guidance.step3_update_263_velocity_v4 import step3_get263data_parallel_process_files
from trajectory_guidance.step4_replace_first_three_frames import step4_replace_first_three_frames_in_directory

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
def main():
    # 设定路径，只需在此处统一调整
    trajectories_file_path = './prompt/trajectories.txt'
    instruct_file_path = './prompt/ChainPrompt.txt'
    result_file_path = './output/newResult.txt'
    npysave_path = "./output/npysave"
    follow_up_question_file = "./prompt/follow_up_question.txt"
    log_file = "./log/model_interaction_log.txt"

    # npysave_path = './output/npysave' /Users/huangziheng/PycharmProjects/trajectory_guidance_pipeline/trajectory_guidance/original263data/0.npy

    output_npy_folder_velocity = './output/npysave_controlvelocity0dot08'
    output_image_folder_velocity = './output/npysave_controlvelocity0dot08_images'
    output_sampled_folder_uniform = './output/uniform_sampled'
    output_sampling_image_folder_uniform = './output/uniform_sampled_images'
    positions_interpolated = './output/interpolated_sampled'
    output_sampling_image_folder_interpolated = './output/interpolated_sampled_images'
    target_mean = 0.08
    num_points = 140


    # positions_interpolated = './output/interpolated_sampled/'
    raw_data_directory = '/Users/huangziheng/PycharmProjects/trajectory_guidance_pipeline/trajectory_guidance/original263data/'
    final_correct_263before3rep = './output/263final_correct/'

    # output_dir_replace = "./output/263final_correct"
    raw_path = "/Users/huangziheng/PycharmProjects/trajectory_guidance_pipeline/trajectory_guidance/original263data/0.npy"
    new_output_dir_replace = "./output/263final_correct_replace3frames"


    for path in [
        npysave_path, output_npy_folder_velocity, output_image_folder_velocity,
        output_sampled_folder_uniform, output_sampling_image_folder_uniform,
        positions_interpolated, output_sampling_image_folder_interpolated,
        final_correct_263before3rep, new_output_dir_replace
    ]:
        ensure_directory_exists(path)


    # # Step 1: 生成速度数据
    # os.environ['WORLD_SIZE'] = '1'
    # step1_generate_velocity(
    #     trajectories_file_path=trajectories_file_path,
    #     instruct_file_path=instruct_file_path,
    #     result_file_path=result_file_path,
    #     npysave_path=npysave_path,
    #     follow_up_question_file=follow_up_question_file,
    #     log_file=log_file
    # )

    # Step 2: 处理速度数据
    step2_process_velocity_data_main_func(
        input_folder=npysave_path,
        output_npy_folder_velocity=output_npy_folder_velocity,
        output_image_folder_velocity=output_image_folder_velocity,
        output_sampled_folder_uniform=output_sampled_folder_uniform,
        output_sampling_image_folder_uniform=output_sampling_image_folder_uniform,
        output_sampled_folder_interpolated=positions_interpolated,
        output_sampling_image_folder_interpolated=output_sampling_image_folder_interpolated,
        target_mean=target_mean,
        num_points=num_points
    )

    # Step 3: 更新为263格式数据
    step3_get263data_parallel_process_files(
        positions_directory=positions_interpolated,
        data_directory=raw_data_directory,
        output_directory=final_correct_263before3rep
    )

    # Step 4: 替换前三帧
    step4_replace_first_three_frames_in_directory(
        output_dir=final_correct_263before3rep,
        raw_path=raw_path,
        new_output_dir=new_output_dir_replace
    )


if __name__ == "__main__":
    main()
