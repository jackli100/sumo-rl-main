import subprocess

def run_program(command):
    # 运行命令
    subprocess.run(command, check=True)

if __name__ == "__main__":
    # 定义要运行的命令和参数
    commands = [
        ["python", r"D:\trg1vr\sumo-rl-main\sumo-rl-main\experiments\test_class2.py",
         '--num_of_episodes', '5',
         '--model_path', r"models\2-stage-0.85-2.zip",
         '--net_path', r"D:\trg1vr\sumo-rl-main\sumo-rl-main\sumo_rl\nets\2way-single-intersection\single-intersection-3.net.xml",
         '--total_timesteps', '5000',
         '--tripinfo',
         '--seed', '50',
            '--emissioninfo',
         '--proportion_of_saturations', '0.85,0.85,0.85,0.85',  
         '--note', 'test models\2-stage-0.85-2.zip']        
    ]

    # 运行命令
    for command in commands:
        run_program(command)
