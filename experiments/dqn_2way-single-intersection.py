import os
import sys

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment


if __name__ == "__main__":
    # 修改这个变量
    output_folder = "outputs/l0.0001-more-than-half-trained-10e5"

    # 维持默认即可
    csv_name = "dqn"
    tripinfo_name = "tripinfos.xml"
    out_csv_name = os.path.join(output_folder, csv_name)
    tripinfo_output_name = os.path.join(output_folder, tripinfo_name)
    tripinfo_cmd = f"--tripinfo {tripinfo_output_name}"

    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection-2.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name=out_csv_name,
        single_agent=True,
        use_gui=True,
        num_seconds=100000,
        reward_fn="average-speed",  
    )
    # 创建结果文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载模型
    model = DQN.load("dqn_single_intersection.zip")
    model.set_env(env)
    model.learn(total_timesteps=100000)

