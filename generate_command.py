import os

def generate_commands(model_dir, net_paths):
    commands = []

    for model_file in os.listdir(model_dir):
        if model_file.endswith('.zip') and '-F.zip' in model_file:
            model_path = os.path.join(model_dir, model_file)
            parts = model_file.split('-')
            saturation = parts[0]
            net_key = parts[1]
            if net_key in net_paths:
                net_path = net_paths[net_key]
                saturation_str = ','.join([f"{float(saturation):.2f}"] * 4)
                command = [
                    "python", "experiments/test_class2.py",
                    '--num_of_episodes', '2',
                    '--net_path', net_path,
                    '--total_timesteps', '200000',
                    '--seed', '50',
                    '--emissioninfo',
                    '--tripinfo',
                    '--proportion_of_saturations', saturation_str,
                    '--note', f'{net_path.split("/")[-1]}, {saturation_str} saturation',
                    '--mode', 'evaluate',
                    '--model', model_path
                ]
                commands.append(command)
    
    return commands

if __name__ == "__main__":
    model_dir = r'D:\trg1vr\sumo-rl-main\sumo-rl-main\renamed_models'  # 请修改为你的模型文件夹路径
    net_paths = {
        '2': "sumo_rl/nets/2way-single-intersection/single-intersection-2.net.xml",
        '3': "sumo_rl/nets/2way-single-intersection/single-intersection-3.net.xml",
        '4': "sumo_rl/nets/2way-single-intersection/single-intersection-4.net.xml"
    }

    commands = generate_commands(model_dir, net_paths)

    # 打印生成的命令行
    for command in commands:
        print(' '.join(command))
