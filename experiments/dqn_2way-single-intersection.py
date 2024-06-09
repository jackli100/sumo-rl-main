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
    
    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection-2.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name="outputs/2way-single-intersection-l0.0001-small/dqn",
        single_agent=True,
        use_gui=False,
        num_seconds=10000,
        vehicle_output=True,
        vehicle_output_name="outputs/2way-single-intersection-l0.0001-small/vehicles.xml",
    )
    # 加载模型
    model = DQN.load("dqn_single_intersection")
    # 将环境设置到模型中
    model.set_env(env)
    model.learn(total_timesteps=10000)
