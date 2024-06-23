import os
import sys
from datetime import datetime
import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment

def run_simulation(net_file, route_file, output_folder):
    csv_name = "dqn"
    tripinfo_name = "tripinfos.xml"
    out_csv_name = os.path.join(output_folder, csv_name)
    tripinfo_output_name = os.path.join(output_folder, tripinfo_name)
    tripinfo_cmd = f"--tripinfo-output {tripinfo_output_name}"

    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv_name,
        single_agent=True,
        use_gui=False,
        num_seconds=100000,
    )

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

    # Start the simulation before learning
    env.reset()
    model.learn(total_timesteps=500000)
    model_save_path = os.path.join(output_folder, "model.zip")
    model.save(model_save_path)
    env.close()  # Ensure the environment is properly closed
    traci.close()  # Explicitly close traci to release all resources

if __name__ == "__main__":
    net_files = [
        "sumo_rl/nets/2way-single-intersection/single-intersection-3.net.xml",
        "sumo_rl/nets/2way-single-intersection/single-intersection-4.net.xml"
    ]
    
    route_files = [
        "sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        "sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml"
    ]
    
    for i in range(3):
        now = datetime.now()
        output_folder = now.strftime(f"outputs/dqn-%Y%m%d-%H%M-{i+1}")
        run_simulation(net_files[i], route_files[i], output_folder)
