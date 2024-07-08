from math import e
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import string
import csv
import asyncio
import xml.etree.ElementTree as ET
from xml.dom import minidom
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
from sumo_rl import SumoEnvironment
import statistics

class TrafficMatrix:
    """
    Represents a traffic matrix for a simulation.
    """

    def __init__(self, output_folder, proportion_of_saturations=[0.75, 0.75, 0.75, 0.75]):
        '''
        Initializes a TrafficMatrix object.

        Args:
            output_folder (str): The output folder path. use function to generate
            proportion_of_saturations (list): The proportion of saturations for each direction of traffic flow, the sequence is [N, S, W, E].        
        '''
        self.prop = proportion_of_saturations
        self.capacity_straight = 2080
        self.capacity_left = 1411
        self.capacity_right = 1411
        self.green_time_proportion = (30 - 4) / 120
        self.convert_to_seconds = 1 / 3600
        self.volumes = self._generate_volumes()
        self.output_folder = output_folder
        self.output_file = os.path.join(output_folder, "output.rou.xml")

    def _generate_volumes(self):
        '''
        Generates the traffic volumes for each direction based on the proportion of saturations.

        Returns:
            list: The traffic volumes for each direction.
        '''
        n_propoertion, s_propoertion, w_propoertion, e_propoertion = self.prop
        volumes = [self.capacity_straight * n_propoertion, self.capacity_right * n_propoertion, self.capacity_left * n_propoertion,
                   self.capacity_straight * s_propoertion, self.capacity_left * s_propoertion, self.capacity_right * s_propoertion,
                   self.capacity_straight * w_propoertion, self.capacity_right * w_propoertion, self.capacity_left * w_propoertion,
                   self.capacity_straight * e_propoertion, self.capacity_left * e_propoertion, self.capacity_right * e_propoertion]
        volumes = [round(volume * self.green_time_proportion * self.convert_to_seconds, 3) for volume in volumes]
        return volumes
    
    def create_xml(self):
        '''
        This method creates an XML file with specified routes and flows for SUMO simulation.

        The method first defines the volumes for different traffic flows.
        Then, it creates the root element of the XML file.
        Next, it adds a vType element to specify the vehicle type.
        After that, it adds route elements for each route in the simulation.
        Finally, it adds flow elements with specified periods for each route.

        Args:
            None

        Returns:
            None
        '''
        ns, nw, ne, sn, sw, se, we, ws, wn, ew, es, en = self.volumes
        # 创建根元素
        root = ET.Element("routes")

        # 添加vType元素
        ET.SubElement(root, "vType", accel="2.6", decel="4.5", id="CarA", length="5.0", minGap="2.5", maxSpeed="55.55", sigma="0.5")

        # 添加route元素
        routes = ["n_t t_s", "n_t t_w", "n_t t_e", "s_t t_n", "s_t t_w", "s_t t_e", "w_t t_e", "w_t t_s", "w_t t_n", "e_t t_w", "e_t t_s", "e_t t_n"]
        for i, route in enumerate(routes, start=1):
            ET.SubElement(root, "route", id=f"route{i:02d}", edges=route)

        # 添加flow元素并替换period值
        periods = [ns, nw, ne, sn, sw, se, we, ws, wn, ew, es, en]
        for i, period in enumerate(periods, start=1):
            ET.SubElement(root, "flow", id=f"flow{i:02d}", begin="0", end="100000", period=f"exp({period})", route=f"route{i:02d}", type="CarA", color="1,1,0")

        # 创建树结构并进行格式化
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_string = reparsed.toprettyxml(indent="  ")

        # 将格式化后的XML写入文件
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(pretty_string)    



import os
import optuna
from stable_baselines3 import DQN
from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from joblib import Parallel, delayed
from sumo_rl import SumoEnvironment  # Import your SumoEnvironment
import json

class Train:
    def __init__(self, 
                 output_folder, 
                 net_file, 
                 route_file, 
                 total_timesteps=10000, 
                 trained_model=None, 
                 num_of_episodes=10, 
                 n_eval_episodes=2,
                 seed=10, 
                 fix_seed=False, 
                 fix_ts=False,
                 learning_rate=0.0001, 
                 learning_starts=0, 
                 train_freq=1, 
                 target_update_interval=2000, 
                 exploration_initial_eps=0.05, 
                 exploration_final_eps=0.01, 
                 training_fraction=1, 
                 verbose=1, 
                 tripinfo=False, 
                 emissioninfo=False, 
                 buffer_size=200000, 
                 batch_size=256, 
                 gamma=0.99):
        self.output_folder = output_folder
        self.tripinfo = tripinfo
        self.emissioninfo = emissioninfo
        self.net_file = net_file
        self.route_file = route_file
        self.csv_name = "dqn"
        self.output_folder = output_folder
        self.tripinfo_name = "tripinfos.xml"
        self.out_csv_name = os.path.join(output_folder, self.csv_name)
        self.tripinfo_output_name = os.path.join(output_folder, self.tripinfo_name)
        self.tripinfo_cmd = f"--tripinfo {self.tripinfo_output_name}"
        self.total_timesteps = total_timesteps
        self.model_save_path = os.path.join(output_folder, "model.zip")
        self.num_of_episodes = num_of_episodes
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.fix_seed = fix_seed
        self.fix_ts = fix_ts
        # initialize the environment
        self.env = SumoEnvironment(
            net_file=self.net_file,
            route_file=self.route_file,
            out_csv_name=self.out_csv_name,
            single_agent=True,
            use_gui=False,
            num_seconds=int(self.total_timesteps / self.num_of_episodes),
            sumo_seed=self.seed,
            fixed_seed=self.fix_seed,
            fixed_ts=self.fix_ts,
            tripinfo=self.tripinfo,
            emissioninfo=self.emissioninfo,
            output_folder=self.output_folder
        )
        self.trained_model = trained_model
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.training_fraction = training_fraction
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.verbose = verbose


    def train(self):
        """
        Trains the model using the specified parameters.

        If a pre-trained model is provided, it loads the model and sets the environment.(It means continue training the model.)
        Otherwise, it creates a new DQN model with the specified parameters.(It means train a new model.)
        """
        if self.trained_model:
            model = DQN.load(self.trained_model)
            model.set_env(self.env)
        else:
            model = DQN(
                env=self.env,
                policy="MlpPolicy",
                learning_rate=self.learning_rate,
                learning_starts=self.learning_starts,
                train_freq=self.train_freq,
                target_update_interval=self.target_update_interval,
                exploration_initial_eps=self.exploration_initial_eps,
                exploration_final_eps=self.exploration_final_eps,
                exploration_fraction=self.training_fraction,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                verbose=self.verbose
            )

            model.learn(total_timesteps=self.total_timesteps)
            model.save(self.model_save_path)

    def evaluate(self):
        """
        Evaluates the trained model using the specified number of episodes.
        """
        if self.trained_model is None:
            raise ValueError("No trained model specified for evaluation.")
        
        model = DQN.load(self.trained_model)
        model.set_env(self.env)
        
        evaluate_policy(model, self.env, n_eval_episodes=self.n_eval_episodes)
    
    def optimize_hyperparameters(self, n_trials=50, n_jobs=4):
        def objective(trial):
            env = SumoEnvironment(
                net_file=self.net_file,
                route_file=self.route_file,
                out_csv_name=self.out_csv_name,
                single_agent=True,
                use_gui=False,
                num_seconds=int(self.total_timesteps / self.num_of_episodes),
                sumo_seed=self.seed,
                fixed_seed=self.fix_seed,
                fixed_ts=self.fix_ts,
                tripinfo=self.tripinfo,
                emissioninfo=self.emissioninfo,
                output_folder=self.output_folder
            )
            env = DummyVecEnv([lambda: env])
            
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
            learning_starts = trial.suggest_int('learning_starts', 0, 5000)
            train_freq = trial.suggest_int('train_freq', 1, 5)
            target_update_interval = trial.suggest_int('target_update_interval', 1000, 10000)
            exploration_initial_eps = trial.suggest_uniform('exploration_initial_eps', 0.04, 1.0)
            exploration_final_eps = trial.suggest_uniform('exploration_final_eps', 0.01, 0.1)
            exploration_fraction = trial.suggest_uniform('exploration_fraction', 0.2, 0.8)
            buffer_size = trial.suggest_int('buffer_size', 50000, 500000)
            batch_size = trial.suggest_int('batch_size', 32, 256)
            
            model = DQN(
                env=env,
                policy="MlpPolicy",
                learning_rate=learning_rate,
                learning_starts=learning_starts,
                train_freq=train_freq,
                target_update_interval=target_update_interval,
                exploration_initial_eps=exploration_initial_eps,
                exploration_final_eps=exploration_final_eps,
                exploration_fraction=exploration_fraction,
                buffer_size=buffer_size,
                batch_size=batch_size,
                verbose=0
            )
            
            model.learn(total_timesteps=self.total_timesteps)
            
            # 在训练过程中记录反馈结果
            rewards = []
            obs = env.reset()
            for _ in range(self.total_timesteps):
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                rewards.append(reward)
                if done:
                    obs = env.reset()

            mean_reward = sum(rewards) / len(rewards)
            
            # Record the trial results
            trial_results = {
                'learning_rate': learning_rate,
                'learning_starts': learning_starts,
                'train_freq': train_freq,
                'target_update_interval': target_update_interval,
                'exploration_initial_eps': exploration_initial_eps,
                'exploration_final_eps': exploration_final_eps,
                'exploration_fraction': exploration_fraction,
                'buffer_size': buffer_size,
                'batch_size': batch_size,
                'mean_reward': mean_reward
            }
            
            self._log_trial_results(trial_results)
            
            return mean_reward

        # Create Optuna study and set parallel optimization
        study = optuna.create_study(direction='maximize')
        
        def parallel_objective(trial_id):
            study.optimize(objective, n_trials=1, n_jobs=1)
        
        Parallel(n_jobs=n_jobs)(delayed(parallel_objective)(i) for i in range(n_trials))
        
        print(f"Best hyperparameters: {study.best_params}")

        # Update class attributes with the best hyperparameters found
        best_params = study.best_params
        self.learning_rate = best_params['learning_rate']
        self.learning_starts = best_params['learning_starts']
        self.train_freq = best_params['train_freq']
        self.target_update_interval = best_params['target_update_interval']
        self.exploration_initial_eps = best_params['exploration_initial_eps']
        self.exploration_final_eps = best_params['exploration_final_eps']
        self.training_fraction = best_params['exploration_fraction']
        self.buffer_size = best_params['buffer_size']
        self.batch_size = best_params['batch_size']

    def _log_trial_results(self, trial_results):
        log_file = os.path.join(self.output_folder, 'hyperparameter_optimization_log.json')
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(trial_results) + '\n')

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statistics

class ShowResults:
    def __init__(self, result_folder, log_file_path, metrics=None, file_prefix="dqn_conn0_ep"):
        self.result_folder = result_folder
        self.log_file_path = log_file_path
        if metrics is None:
            self.metrics = ['system_total_stopped', 'system_total_waiting_time', 'system_mean_waiting_time']
        else:
            self.metrics = metrics
        self.file_prefix = file_prefix

    def read_and_concatenate_csv(self, file_paths):
        dfs = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                dfs.append(pd.read_csv(file_path))
            else:
                raise FileNotFoundError(f"File not found: {file_path}")

        # Add a cumulative step column to each dataframe
        for i in range(1, len(dfs)):
            dfs[i]['step'] += dfs[i-1]['step'].max()

        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df

    def save_plot(self, steps, values, filename, ylabel, title):
        plt.figure()
        plt.plot(steps, values, label=ylabel)
        plt.xlabel('step')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def calculate_statistics(self, values):
        average = statistics.mean(values)
        median = statistics.median(values)
        return average, median

    def main(self):
        # Log file setup
        with open(self.log_file_path, 'w') as log_file:
            log_file.write("Execution started at {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Find the relevant CSV files in the result folder
            file_paths = sorted([os.path.join(self.result_folder, f) for f in os.listdir(self.result_folder) if f.startswith(self.file_prefix) and f.endswith('.csv')])

            # Ensure all file paths exist
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")

            # Read and concatenate the CSV files, even if combined.csv exists
            combined_df = self.read_and_concatenate_csv(file_paths)
            combined_csv_path = os.path.join(self.result_folder, "combined.csv")
            combined_df.to_csv(combined_csv_path, index=False)
            log_file.write("Combined CSV saved to {}\n".format(combined_csv_path))

            for metric in self.metrics:
                # Get steps and metric values from combined.csv
                steps = combined_df['step']
                values = combined_df[metric]
                
                combined_plot_filename = os.path.join(self.result_folder, f'{metric}_combined.png')
                self.save_plot(steps, values, combined_plot_filename, metric, f'{metric} (Combined)')
                log_file.write("Combined plot for {} saved to {}\n".format(metric, combined_plot_filename))

                # Calculate and log average and median of the selected metric
                avg_value, median_value = self.calculate_statistics(combined_df[metric])
                log_file.write(f'Overall average {metric}: {avg_value:.3f}\n')
                log_file.write(f'Overall median {metric}: {median_value:.3f}\n')
            
            # Calculate and log average value of the selected metric for each episode
            episode_averages = []
            for i, file_path in enumerate(file_paths, start=1):
                df = pd.read_csv(file_path)
                avg_value = df[self.metrics[1]].mean()
                episode_averages.append((i, avg_value))
                log_file.write(f'Average {self.metrics[1]} for episode {i}: {avg_value:.3f}\n')
            

# Example of usage
# processor = ShowResults(result_folder="path/to/results", log_file_path="path/to/log.txt")
# processor.main()


# 假设 TrafficMatrix, Train, 和 ShowResults 类已经定义好了

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime
import random

class EmailSender:
    def __init__(self, username, auth_code):
        self.server = 'smtp.qq.com'
        self.port = 465  # SSL端口
        self.username = username
        self.auth_code = auth_code

    def send_email(self, recipient, subject, body, attachments=None):
        # 创建邮件对象
        msg = MIMEMultipart()
        msg['From'] = self.username
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # 添加附件
        if attachments:
            for file_path in attachments:
                part = MIMEBase('application', 'octet-stream')
                with open(file_path, 'rb') as file:
                    part.set_payload(file.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
                msg.attach(part)

        # 连接到 SMTP 服务器
        with smtplib.SMTP_SSL(self.server, self.port) as server:
            server.login(self.username, self.auth_code)
            server.send_message(msg)
            print("Email sent successfully!")

def generate_result_folder():
    # 获取当前时间
    now = datetime.now()
    # 格式化时间为日期、小时、分钟和秒
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H-%M-%S")
    # 生成随机数
    random_number = random.randint(1000, 9999)
    # 生成文件夹名称
    folder_name = f"{time}-{random_number}"
    # 构建完整路径，确保目录存在
    base_dir = "/root/autodl-tmp/outputs"
    result_folder = os.path.join(base_dir, date, folder_name)
    
    # 确保基目录和日期目录存在
    os.makedirs(result_folder, exist_ok=True)
    
    return result_folder


def get_files_to_send(result_folder):
    # 获取 result_folder 中所有的 .png 和 .txt 文件
    files = []
    for root, _, filenames in os.walk(result_folder):
        for filename in filenames:
            if filename.endswith('.png') or filename.endswith('.txt') or filename.endswith('.zip') or filename == 'result.csv':
                files.append(os.path.join(root, filename))
    return files



