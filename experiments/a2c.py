import os
import sys

import gymnasium as gym
from stable_baselines3 import A2C


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment


if __name__ == "__main__":
    # 修改这个变量
    output_folder = "outputs/a2c-0623-3"

    # 维持默认即可
    csv_name = "a2c"
    tripinfo_name = "tripinfos.xml"
    out_csv_name = os.path.join(output_folder, csv_name)
    tripinfo_output_name = os.path.join(output_folder, tripinfo_name)
    tripinfo_cmd = f"--tripinfo {tripinfo_output_name}"

    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection-2.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name=out_csv_name,
        single_agent=True,
        use_gui=False,
        num_seconds=100000,
    )
    # 创建结果文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # 初始化A2C模型
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0007,  # 学习率
        n_steps=5,  # 这里设置为5步更新，可以根据需要调整
        gamma=0.99,  # 折扣因子，通常在0.99左右
        gae_lambda=1.0,  # GAE参数，通常设置为1
        ent_coef=0.05,  # 熵系数，用于鼓励策略的多样性
        vf_coef=0.5,  # 值函数的系数
        max_grad_norm=0.2,  # 最大梯度范数，用于梯度裁剪
        verbose=1,
    )

    model.learn(total_timesteps=500000)
    # Save the model
    model_save_path=os.path.join(output_folder, "model.zip")
    model.save(model_save_path)