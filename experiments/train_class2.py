import argparse
import os
import sys
from train_class import generate_result_folder, TrafficMatrix, Train, ShowResults, EmailSender  # 替换为实际模块名称

def main(result_folder, proportion_of_saturations, net_path, total_timesteps, model_path, num_of_episodes, seed, fix_seed, learning_rate, learning_starts, train_freq, target_update_interval, exploration_initial_eps, exploration_final_eps, verbose):
    log_file_path = os.path.join(result_folder, 'output_log.txt')
    matrix = TrafficMatrix(proportion_of_saturations, result_folder)
    matrix.create_xml()
    route_path = matrix.output_file

    train = Train(result_folder, net_path, route_path, total_timesteps, model_path, num_of_episodes=num_of_episodes, seed=seed, fix_seed=fix_seed, 
                  learning_rate=learning_rate, learning_starts=learning_starts, train_freq=train_freq, target_update_interval=target_update_interval, 
                  exploration_initial_eps=exploration_initial_eps, exploration_final_eps=exploration_final_eps, verbose=verbose)
    train.print_hyperparameters()
    train.train()

    metric = "system_total_waiting_time"
    show_results = ShowResults(result_folder, log_file_path, metric, 1, int(num_of_episodes + 1))
    show_results.main()

def log_command_line_arguments(log_file_path):
    with open(log_file_path, 'a') as log_file:
        log_file.write(' '.join(sys.argv) + '\n')

def get_files_to_send(result_folder):
    # 获取 result_folder 中所有的 .png 和 .txt 文件
    files = []
    for root, _, filenames in os.walk(result_folder):
        for filename in filenames:
            if filename.endswith('.png') or filename.endswith('.txt'):
                files.append(os.path.join(root, filename))
    return files
def str_to_float_list(arg_str):
    return list(map(float, arg_str.split(',')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate traffic model.')

    parser.add_argument('--proportion_of_saturations', type=str_to_float_list, default=[0.75, 0.75, 0.75, 0.75], help='Proportion of saturations for traffic matrix')
    parser.add_argument('--net_path', type=str, required=True, help='Path to the network file')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total number of timesteps for training')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model file')
    parser.add_argument('--num_of_episodes', type=int, default=10, help='Number of episodes for training')
    parser.add_argument('--seed', type=int, default=24, help='Random seed')
    # 使用 action='store_true' 和 action='store_false' 来处理布尔值参数
    parser.add_argument('--fix_seed', action='store_true', help='Fix seed yes or no')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--learning_starts', type=int, default=0, help='Learning starts')
    parser.add_argument('--train_freq', type=int, default=1, help='Training frequency')
    parser.add_argument('--target_update_interval', type=int, default=2000, help='Target update interval')
    parser.add_argument('--exploration_initial_eps', type=float, default=0.05, help='Initial exploration epsilon')
    parser.add_argument('--exploration_final_eps', type=float, default=0.01, help='Final exploration epsilon')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--sender_email', type=str, default='1811743445@qq.com', help='Sender email address')
    parser.add_argument('--auth_code', type=str, default='zbfuppehtwtkejfg', help='Authorization code for the sender email')
    parser.add_argument('--recipient_email', type=str, default='zl22n23@soton.ac.uk', help='Recipient email address')

    args = parser.parse_args()

    # 生成 result_folder
    result_folder = generate_result_folder()

    # 记录命令行输入到文件
    command_log_file_path = os.path.join(result_folder, 'command_log.txt')
    log_command_line_arguments(command_log_file_path)

    # 执行主要的训练和评估逻辑
    main(result_folder, args.proportion_of_saturations, args.net_path, args.total_timesteps, args.model_path, args.num_of_episodes, args.seed, args.fix_seed, 
         args.learning_rate, args.learning_starts, args.train_freq, args.target_update_interval, args.exploration_initial_eps, args.exploration_final_eps, args.verbose)

    # 获取要发送的文件
    files_to_send = get_files_to_send(result_folder)

    # 发送邮件
    email_sender = EmailSender(args.sender_email, args.auth_code)
    subject = 'Training Results'
    body = 'Please find the attached result files from the latest training run.'
    email_sender.send_email(args.recipient_email, subject, body, attachments=files_to_send)
