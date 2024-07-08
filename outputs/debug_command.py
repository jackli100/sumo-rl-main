def generate_commands():
    # 定义参数
    net_paths = [
        "sumo_rl/nets/2way-single-intersection/single-intersection-3.net.xml",
        "sumo_rl/nets/2way-single-intersection/single-intersection-2.net.xml",
        "sumo_rl/nets/2way-single-intersection/single-intersection-4.net.xml"
    ]
    
    fix_ts_options = [True, False]
    saturations = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    
    commands = []
    
    for net_path in net_paths:
        for saturation in saturations:
            saturation_str = ','.join([f"{saturation:.2f}"] * 4)
            if "single-intersection-3" in net_path or "single-intersection-4" in net_path:
                for fix_ts in fix_ts_options:
                    command = [
                        "python", "experiments/train_class2.py",
                        '--num_of_episodes', '9',
                        '--net_path', net_path,
                        '--total_timesteps', '900000',
                        '--proportion_of_saturations', saturation_str,
                        '--note', f'{net_path.split("/")[-1]}, {saturation_str} saturation',
                        '--mode', 'train',
                    ]
                    if fix_ts:
                        command.insert(9, '--fix_ts')
                    commands.append(command)
            else:
                command = [
                    "python", "experiments/train_class2.py",
                    '--num_of_episodes', '9',
                    '--net_path', net_path,
                    '--total_timesteps', '900000',
                    '--proportion_of_saturations', saturation_str,
                    '--note', f'{net_path.split("/")[-1]}, {saturation_str} saturation',
                    '--mode', 'train',
                ]
                commands.append(command)
    
    return commands

def print_commands(commands):
    for command in commands:
        print(" ".join(command))

if __name__ == "__main__":
    commands = generate_commands()
    print_commands(commands)
