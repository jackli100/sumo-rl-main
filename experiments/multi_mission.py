import subprocess

def run_program(command):
    # 运行命令
    subprocess.run(command, check=True)

if __name__ == "__main__":
    # 定义要运行的命令和参数
    commands = [
        ["python", r"D:\trg1vr\sumo-rl-main\sumo-rl-main\experiments\train_class2.py",
         '--num_of_episodes', '10',
         '--net_path', r"D:\trg1vr\sumo-rl-main\sumo-rl-main\sumo_rl\nets\2way-single-intersection\single-intersection-3.net.xml",
         '--total_timesteps', '500000',
         '--proportion_of_saturations', '0.5,0.5,0.5,0.5'],          
    ]

    # 运行命令
    for command in commands:
        run_program(command)
