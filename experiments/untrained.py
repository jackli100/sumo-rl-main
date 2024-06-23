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
    output_folder = "outputs/dqn-0622-7"

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
        use_gui=False,
        num_seconds=200,
    )
    # 创建结果文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.0001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=2000,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=1000)
    # Save the model
    model_save_path=os.path.join(output_folder, "model.zip")
    model.save(model_save_path)
    
   