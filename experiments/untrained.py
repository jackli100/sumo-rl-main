import os
import sys
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import tensorflow as tf
from datetime import datetime

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
from sumo_rl import SumoEnvironment

class TensorboardCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = 0
        self.current_length = 0

    def _on_step(self) -> bool:
        self.current_rewards += self.locals['rewards'][0]
        self.current_length += 1

        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_rewards)
            self.episode_lengths.append(self.current_length)

            with self.writer.as_default():
                tf.summary.scalar('reward', self.current_rewards, step=self.num_timesteps)
                tf.summary.scalar('episode_length', self.current_length, step=self.num_timesteps)
                self.writer.flush()  # Ensure data is written to disk

            # Reset for next episode
            print(f"Episode done: total reward: {self.current_rewards}, length: {self.current_length}")
            self.current_rewards = 0
            self.current_length = 0

        return True

if __name__ == "__main__":
    output_folder = "outputs/testtest"
    log_dir = os.path.join(output_folder, "logs")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

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
        num_seconds=100000,
    )

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
        tensorboard_log=log_dir,
    )

    callback = TensorboardCallback(log_dir)
    model.learn(total_timesteps=500000, callback=callback)

    model_save_path = os.path.join(output_folder, "model.zip")
    model.save(model_save_path)
