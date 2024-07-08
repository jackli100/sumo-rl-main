import argparse
from importlib.metadata import files
import os
import sys
from test_class import generate_result_folder, TrafficMatrix, Train, ShowResults, EmailSender, get_files_to_send  # 替换为实际模块名称
import csv
import re
import subprocess

import os
import csv
import re

def save_results_to_csv(result_folder, args, log_content):
    csv_file_path = os.path.join(result_folder, 'result.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Parameter', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'Parameter': 'proportion_of_saturations', 'Value': args.proportion_of_saturations})
        writer.writerow({'Parameter': 'net_path', 'Value': args.net_path})
        writer.writerow({'Parameter': 'total_timesteps', 'Value': args.total_timesteps})
        writer.writerow({'Parameter': 'model_path', 'Value': args.model_path})
        writer.writerow({'Parameter': 'num_of_episodes', 'Value': args.num_of_episodes})
        writer.writerow({'Parameter': 'seed', 'Value': args.seed})
        writer.writerow({'Parameter': 'fix_seed', 'Value': args.fix_seed})
        writer.writerow({'Parameter': 'fix_ts', 'Value': args.fix_ts})
        writer.writerow({'Parameter': 'learning_rate', 'Value': args.learning_rate})
        writer.writerow({'Parameter': 'learning_starts', 'Value': args.learning_starts})
        writer.writerow({'Parameter': 'train_freq', 'Value': args.train_freq})
        writer.writerow({'Parameter': 'target_update_interval', 'Value': args.target_update_interval})
        writer.writerow({'Parameter': 'exploration_initial_eps', 'Value': args.exploration_initial_eps})
        writer.writerow({'Parameter': 'exploration_final_eps', 'Value': args.exploration_final_eps})
        writer.writerow({'Parameter': 'verbose', 'Value': args.verbose})
        writer.writerow({'Parameter': 'tripinfo', 'Value': args.tripinfo})
        writer.writerow({'Parameter': 'emissioninfo', 'Value': args.emissioninfo})
        writer.writerow({'Parameter': 'buffer_size', 'Value': args.buffer_size})
        writer.writerow({'Parameter': 'batch_size', 'Value': args.batch_size})
        writer.writerow({'Parameter': 'gamma', 'Value': args.gamma})
        writer.writerow({'Parameter': 'note', 'Value': args.note})

        # 使用正则表达式提取路径、平均值和中位数
        combined_csv_match = re.search(r'Combined CSV saved to (.+)', log_content)
        combined_plot_stopped_match = re.search(r'Combined plot for system_total_stopped saved to (.+)', log_content)
        combined_plot_waiting_time_match = re.search(r'Combined plot for system_total_waiting_time saved to (.+)', log_content)
        combined_plot_mean_waiting_time_match = re.search(r'Combined plot for system_mean_waiting_time saved to (.+)', log_content)

        overall_avg_stopped_match = re.search(r'Overall average system_total_stopped: ([\d.]+)', log_content)
        overall_median_stopped_match = re.search(r'Overall median system_total_stopped: ([\d.]+)', log_content)

        overall_avg_waiting_time_match = re.search(r'Overall average system_total_waiting_time: ([\d.]+)', log_content)
        overall_median_waiting_time_match = re.search(r'Overall median system_total_waiting_time: ([\d.]+)', log_content)

        overall_avg_mean_waiting_time_match = re.search(r'Overall average system_mean_waiting_time: ([\d.]+)', log_content)
        overall_median_mean_waiting_time_match = re.search(r'Overall median system_mean_waiting_time: ([\d.]+)', log_content)

        if combined_csv_match:
            writer.writerow({'Parameter': 'Combined CSV Path', 'Value': combined_csv_match.group(1)})
        if combined_plot_stopped_match:
            writer.writerow({'Parameter': 'Combined Plot for system_total_stopped Path', 'Value': combined_plot_stopped_match.group(1)})
        if combined_plot_waiting_time_match:
            writer.writerow({'Parameter': 'Combined Plot for system_total_waiting_time Path', 'Value': combined_plot_waiting_time_match.group(1)})
        if combined_plot_mean_waiting_time_match:
            writer.writerow({'Parameter': 'Combined Plot for system_mean_waiting_time Path', 'Value': combined_plot_mean_waiting_time_match.group(1)})

        if overall_avg_stopped_match:
            writer.writerow({'Parameter': 'Overall Average system_total_stopped', 'Value': overall_avg_stopped_match.group(1)})
        if overall_median_stopped_match:
            writer.writerow({'Parameter': 'Overall Median system_total_stopped', 'Value': overall_median_stopped_match.group(1)})

        if overall_avg_waiting_time_match:
            writer.writerow({'Parameter': 'Overall Average system_total_waiting_time', 'Value': overall_avg_waiting_time_match.group(1)})
        if overall_median_waiting_time_match:
            writer.writerow({'Parameter': 'Overall Median system_total_waiting_time', 'Value': overall_median_waiting_time_match.group(1)})

        if overall_avg_mean_waiting_time_match:
            writer.writerow({'Parameter': 'Overall Average system_mean_waiting_time', 'Value': overall_avg_mean_waiting_time_match.group(1)})
        if overall_median_mean_waiting_time_match:
            writer.writerow({'Parameter': 'Overall Median system_mean_waiting_time', 'Value': overall_median_mean_waiting_time_match.group(1)})
        if args.mode == 'train':
            # 使用正则表达式匹配每一行的平均值
            episode_avg_waiting_time_matches = re.findall(r'Average system_total_waiting_time for episode (\d+): ([\d.]+)', log_content)

            # 处理匹配结果
            episode_avg_waiting_times = [(int(match[0]), float(match[1])) for match in episode_avg_waiting_time_matches]

            # 输出结果
            for episode, avg_waiting_time in episode_avg_waiting_times:
                writer.writerow({'Parameter': f'Average system_total_waiting_time for episode {episode}', 'Value': avg_waiting_time})

    return csv_file_path


def upload_to_onedrive(local_file, remote_folder):
    # 获取母文件夹名称
    parent_folder_name = os.path.basename(os.path.dirname(local_file))
    
    # 获取目标文件名称
    file_name = os.path.basename(local_file)
    
    # 生成新的文件名称
    new_file_name = f"{parent_folder_name}-{file_name}"
    
    # 生成新的本地文件路径
    new_local_file_path = os.path.join(os.path.dirname(local_file), new_file_name)
    
    # 重命名文件
    os.rename(local_file, new_local_file_path)
    
    # 使用rclone上传文件
    rclone_command = f"rclone copyto {new_local_file_path} {remote_folder}/{new_file_name}"
    try:
        subprocess.run(rclone_command, check=True, shell=True)
        print(f"Successfully uploaded {new_local_file_path} to {remote_folder}/{new_file_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to upload {new_local_file_path} to {remote_folder}/{new_file_name}. Error: {e}")

import os
import argparse

def main(result_folder, proportion_of_saturations, net_path, total_timesteps, model_path, num_of_episodes, seed, 
         fix_seed, fixed_ts, learning_rate, learning_starts, train_freq, target_update_interval, exploration_initial_eps, exploration_final_eps, verbose,
         tripinfo, emissioninfo, buffer_size, batch_size, gamma, mode):
    log_file_path = os.path.join(result_folder, 'output_log.txt')
    matrix = TrafficMatrix(proportion_of_saturations, result_folder)
    matrix.create_xml()
    route_path = matrix.output_file

    train = Train(
        output_folder=result_folder, 
        net_file=net_path, 
        route_file=route_path, 
        total_timesteps=total_timesteps, 
        trained_model=model_path, 
        num_of_episodes=num_of_episodes, 
        seed=seed, 
        fix_seed=fix_seed, 
        fix_ts=fixed_ts, 
        learning_rate=learning_rate, 
        learning_starts=learning_starts, 
        train_freq=train_freq, 
        target_update_interval=target_update_interval, 
        exploration_initial_eps=exploration_initial_eps, 
        exploration_final_eps=exploration_final_eps, 
        verbose=verbose, 
        tripinfo=tripinfo, 
        emissioninfo=emissioninfo,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma
    )
    
    train.print_hyperparameters()
    
    if mode == 'train':
        train.train()
    elif mode == 'evaluate':
        train.evaluate()
    else:
        raise ValueError("Invalid mode. Use 'train' or 'evaluate'.")

    show_results = ShowResults(result_folder, log_file_path)
    show_results.main()

def str_to_float_list(arg_str):
    return list(map(float, arg_str.split(',')))

def train_mode(args, csv_file_path, result_folder, log_content):
    result_remote_path = "mydrive:results-train"
    upload_to_onedrive(csv_file_path, result_remote_path)
    model_local_path = os.path.join(result_folder, 'model.zip')
    model_remote_path = "mydrive:models"
    upload_to_onedrive(model_local_path, model_remote_path)
    
    # send email
    email_sender = EmailSender(args.sender_email, args.auth_code)
    files_to_send = get_files_to_send(result_folder)

    subject = f'Training Results {args.note}'
    body = f'''Here is the key information for the training:

    --proportion_of_saturations: {args.proportion_of_saturations}
    --net_path: {args.net_path}
    --total_timesteps: {args.total_timesteps}
    --model_path: {args.model_path}
    --num_of_episodes: {args.num_of_episodes}
    --seed: {args.seed}
    --fix_seed: {args.fix_seed}
    --fix_ts: {args.fix_ts}
    --learning_rate: {args.learning_rate}
    --learning_starts: {args.learning_starts}
    --train_freq: {args.train_freq}
    --target_update_interval: {args.target_update_interval}
    --exploration_initial_eps: {args.exploration_initial_eps}
    --exploration_final_eps: {args.exploration_final_eps}
    --verbose: {args.verbose}
    --tripinfo: {args.tripinfo}
    --emissioninfo: {args.emissioninfo}
    --buffer_size: {args.buffer_size}
    --batch_size: {args.batch_size}
    --gamma: {args.gamma}
    --note: {args.note}

    Please find the attached files for more details.

    Log details:
    {log_content}
    '''
    email_sender.send_email(args.recipient_email, subject, body, attachments=files_to_send)

def evaluate_mode(args, csv_file_path, result_folder, log_content):
    result_remote_path = "mydrive:results-evaluate"
    upload_to_onedrive(csv_file_path, result_remote_path)
    
    # send email
    email_sender = EmailSender(args.sender_email, args.auth_code)
    files_to_send = get_files_to_send(result_folder)

    subject = f'Evaluation Results {args.note}'
    body = f'''Here is the key information for the evaluation:

    --proportion_of_saturations: {args.proportion_of_saturations}
    --net_path: {args.net_path}
    --total_timesteps: {args.total_timesteps}
    --model_path: {args.model_path}
    --num_of_episodes: {args.num_of_episodes}
    --seed: {args.seed}
    --fix_seed: {args.fix_seed}
    --fix_ts: {args.fix_ts}
    --learning_rate: {args.learning_rate}
    --learning_starts: {args.learning_starts}
    --train_freq: {args.train_freq}
    --target_update_interval: {args.target_update_interval}
    --exploration_initial_eps: {args.exploration_initial_eps}
    --exploration_final_eps: {args.exploration_final_eps}
    --verbose: {args.verbose}
    --tripinfo: {args.tripinfo}
    --emissioninfo: {args.emissioninfo}
    --buffer_size: {args.buffer_size}
    --batch_size: {args.batch_size}
    --gamma: {args.gamma}
    --note: {args.note}

    Please find the attached files for more details.

    Log details:
    {log_content}
    '''
    email_sender.send_email(args.recipient_email, subject, body, attachments=files_to_send)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate traffic model.')

    parser.add_argument('--proportion_of_saturations', type=str_to_float_list, default=[0.75, 0.75, 0.75, 0.75], help='Proportion of saturations for traffic matrix')
    parser.add_argument('--net_path', type=str, required=True, help='Path to the network file')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='Total number of timesteps for training')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model file')
    parser.add_argument('--num_of_episodes', type=int, default=10, help='Number of episodes for training')
    parser.add_argument('--seed', type=int, default=24, help='Random seed')
    parser.add_argument('--fix_seed', action='store_true', help='Fix seed yes or no')
    parser.add_argument('--fix_ts', action='store_true', help='Fix signal yes or no')
    parser.add_argument('--tripinfo', action='store_true', help='Print tripinfo yes or no')
    parser.add_argument('--emissioninfo', action='store_true', help='Print emissioninfo yes or no')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--learning_starts', type=int, default=0, help='Learning starts')
    parser.add_argument('--train_freq', type=int, default=1, help='Training frequency')
    parser.add_argument('--target_update_interval', type=int, default=2000, help='Target update interval')
    parser.add_argument('--exploration_initial_eps', type=float, default=0.05, help='Initial exploration epsilon')
    parser.add_argument('--exploration_final_eps', type=float, default=0.01, help='Final exploration epsilon')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--buffer_size', type=int, default=200000, help='Buffer size for replay buffer')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards')
    parser.add_argument('--sender_email', type=str, default='1811743445@qq.com', help='Sender email address')
    parser.add_argument('--auth_code', type=str, default='zbfuppehtwtkejfg', help='Authorization code for the sender email')
    parser.add_argument('--recipient_email', type=str, default='zl22n23@soton.ac.uk', help='Recipient email address')
    parser.add_argument('--note', type=str, default='', help='Note to include in the email')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], required=True, help="Mode to run: 'train' or 'evaluate'")
    

    args = parser.parse_args()

    result_folder = generate_result_folder()

    main(result_folder, args.proportion_of_saturations, args.net_path, args.total_timesteps, args.model_path, args.num_of_episodes, args.seed, args.fix_seed, args.fix_ts,
         args.learning_rate, args.learning_starts, args.train_freq, args.target_update_interval, args.exploration_initial_eps, args.exploration_final_eps, args.verbose, 
         args.tripinfo, args.emissioninfo, args.buffer_size, args.batch_size, args.gamma, args.mode)

    # 此时就有了output_log.txt文件，理论上可以读取这个文件的内容，然后发送邮件
    # 读取 output_log.txt 文件内容
    log_file_path = os.path.join(result_folder, 'output_log.txt')
    with open(log_file_path, 'r') as log_file:
        log_content = log_file.read()
    
    csv_file_path = save_results_to_csv(result_folder, args, log_content)

    # upload to onedrive
    # 上传结果文件和模型文件到 OneDrive
    parent_folder_name = os.path.basename(result_folder)
    if args.mode == 'train':
        train_mode(args, csv_file_path, result_folder, log_content)
    elif args.mode == 'evaluate':
        evaluate_mode(args, csv_file_path, result_folder, log_content)
