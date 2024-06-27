import numpy as np
import pandas as pd
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
from stable_baselines3.dqn.dqn import DQN
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
from sumo_rl import SumoEnvironment

class TrafficMatrix:
    '''
    矩阵类，用于生成流量矩阵并保存到文件
    '''
    def __init__(self, proportion_of_saturations, output_folder):
        '''
        param: proportion_of_saturations: 饱和度比例，列表，依次是北侧，南侧，西侧，东侧的车流比例
        '''
        self.prop = proportion_of_saturations
        self.capacity_straight = 2080
        self.capacity_left = 1411
        self.capacity_right = 1411
        self.green_time_proportion = (30 - 4) / 120
        self.convert_to_seconds = 1/3600
        self.volumes = self._generate_volumes()
        self.output_folder = output_folder
        self.output_file = os.path.join(output_folder, "output.rou.xml")

    def _generate_volumes(self):
        '''
        在创建对象的时候，就生成这个对象的流量矩阵
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
        ns: straight flow from north to south
        nw: right flow from north to west
        ne: left flow from north to east
        sn: straight flow from south to north
        sw: left flow from south to west
        se: right flow from south to east
        we: straight flow from west to east
        ws: right flow from west to south
        wn: left flow from west to north
        ew: straight flow from east to west
        es: left flow from east to south
        en: right flow from east to north
        '''
        ns, nw, ne, sn, sw, se, we, ws, wn, ew, es, en = self.volumes
        # 创建根元素
        root = ET.Element("routes")

        # 添加vType元素
        vType = ET.SubElement(root, "vType", accel="2.6", decel="4.5", id="CarA", length="5.0", minGap="2.5", maxSpeed="55.55", sigma="0.5")

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

    def print_volumes(self):
        print(f"Volumes: {self.volumes}")

class Train:
    def __init__(self, output_folder, net_file, route_file, total_timesteps, trained_model=None, num_of_episodes=10, seed=10, fix_seed=False, \
                 learning_rate=0.0001, learning_starts=0, train_freq=1, target_update_interval=2000, exploration_initial_eps=0.05, exploration_final_eps=0.01, verbose=1):
        self.output_folder = output_folder
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
        self.seed = seed
        self.fix_seed = fix_seed
        self.env = SumoEnvironment(
            net_file=self.net_file,
            route_file=self.route_file,
            out_csv_name=self.out_csv_name,
            single_agent=True,
            use_gui=False,
            num_seconds=int(self.total_timesteps/self.num_of_episodes), 
            sumo_seed=self.seed,
            fixed_seed=self.fix_seed
        )
        self.trained_model = trained_model
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.verbose = verbose

   
        


   
    def print_hyperparameters(self):
        with open(os.path.join(self.output_folder, "hyperparameters.txt"), "w") as f:  
            f.write(f"total_timesteps: {self.total_timesteps}\n")
            f.write(f"model_save_path: {self.model_save_path}\n")
            f.write(f"num_of_episodes: {self.num_of_episodes}\n")
            f.write(f"seed: {self.seed}\n")
            f.write(f"net_file: {self.net_file}\n")
            f.write(f"route_file: {self.route_file}\n")
            f.write(f"out_csv_name: {self.out_csv_name}\n")
            f.write("fixed_seed: {}\n".format(self.fix_seed))
            f.write(f"tripinfo_output_name: {self.tripinfo_output_name}\n")
            f.write(f"tripinfo_cmd: {self.tripinfo_cmd}\n")
            f.write(f"trained_model: {self.trained_model}\n")
            f.write(f"learning_rate: {self.learning_rate}\n")
            f.write(f"learning_starts: {self.learning_starts}\n")
            f.write(f"train_freq: {self.train_freq}\n")
            f.write(f"target_update_interval: {self.target_update_interval}\n")
            f.write(f"exploration_initial_eps: {self.exploration_initial_eps}\n")
            f.write(f"exploration_final_eps: {self.exploration_final_eps}\n")
            f.write(f"verbose: {self.verbose}\n")

    def train(self):
           
        if self.trained_model:
            model = DQN.load(self.trained_model)
            # Set the environment
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
                verbose=self.verbose
            )

        model.learn(total_timesteps=self.total_timesteps)
        # Save the model
        model.save(self.model_save_path)
        self.env.reset()
    
class ShowResults:
    def __init__(self, result_folder, log_file_path, metric, first_episode=1, last_episode=7):
        self.result_folder = result_folder
        self.log_file_path = log_file_path
        self.metric = metric
        self.first_episode = first_episode
        self.last_episode = last_episode

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

    def main(self):
        # Log file setup
        with open(self.log_file_path, 'w') as log_file:
            log_file.write("Execution started at {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Find the relevant CSV files in the result folder
            file_prefix = "dqn_conn0_ep"
            file_paths = [os.path.join(self.result_folder, f"{file_prefix}{i}.csv") for i in range(self.first_episode, self.last_episode+1)]
            
            # Ensure all file paths exist
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                
            # if "combined.csv" in result_folder:
            if not os.path.exists(os.path.join(self.result_folder, "combined.csv")):    
                # Read and concatenate the CSV files
                combined_df = self.read_and_concatenate_csv(file_paths)
                combined_csv_path = os.path.join(self.result_folder, "combined.csv")
                combined_df.to_csv(combined_csv_path, index=False)
                log_file.write("Combined CSV saved to {}\n".format(combined_csv_path))

            # get steps from combined.csv
            steps = pd.read_csv(os.path.join(self.result_folder, "combined.csv"))['step'] 
            values = pd.read_csv(os.path.join(self.result_folder, "combined.csv"))[self.metric]
            combined_plot_filename = os.path.join(self.result_folder, f'{self.metric}_combined.png')
            self.save_plot(steps, values, combined_plot_filename, self.metric, f'{self.metric} (Combined)')
            log_file.write("Combined plot saved to {}\n".format(combined_plot_filename))

            # Plot and save the selected metric for the last episode
            last_episode_path = file_paths[-1]
            last_episode_df = pd.read_csv(last_episode_path)
            steps = last_episode_df['step']
            values = last_episode_df[self.metric]
            last_episode_plot_filename = os.path.join(self.result_folder, f'{self.metric}_last_episode.png')
            self.save_plot(steps, values, last_episode_plot_filename, self.metric, f'{self.metric} (Last Episode)')
            log_file.write("Last episode plot saved to {}\n".format(last_episode_plot_filename))

            # Calculate and log average value of the selected metric for the last episode
            average_value = values.mean()
            log_file.write(f'Average {self.metric} for the last episode: {average_value:.3f}\n')

            log_file.write("Execution finished at {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    


import asyncio
import os
import string
from datetime import datetime

# 假设 TrafficMatrix, Train, 和 ShowResults 类已经定义好了

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime

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
    # 生成文件夹名称
    folder_name = f"{time}"
    # 使用 os.path.join 构建路径
    result_folder = os.path.join("outputs", date, folder_name)
    
    # 如果文件夹不存在则创建
    os.makedirs(result_folder, exist_ok=True)
    
    return result_folder

def get_files_to_send(result_folder):
    # 获取 result_folder 中所有的 .png 和 .txt 文件
    files = []
    for root, _, filenames in os.walk(result_folder):
        for filename in filenames:
            if filename.endswith('.png') or filename.endswith('.txt'):
                files.append(os.path.join(root, filename))
    return files

if __name__ == "__main__":
    result_folder = generate_result_folder()
    log_file_path = os.path.join(result_folder, 'output_log.txt')
    proportion_of_saturations = [0.75, 0.75, 0.75, 0.75]
    matrix = TrafficMatrix(proportion_of_saturations, result_folder)
    matrix.create_xml()
    route_path = matrix.output_file
    net_path = r"D:\trg1vr\sumo-rl-main\sumo-rl-main\sumo_rl\nets\2way-single-intersection\single-intersection-2.net.xml"
    total_timesteps = 1000
    model_path = r"D:\trg1vr\sumo-rl-main\sumo-rl-main\outputs\2024-06-25\14-59-01-Vw1OU1\model.zip"
    num_of_episodes = 10
    fix_seed = False
    train = Train(result_folder, net_path, route_path, total_timesteps, model_path, num_of_episodes=num_of_episodes, seed=24, fix_seed=fix_seed,\
                  learning_rate=0.0001, learning_starts=0, train_freq=1, target_update_interval=2000, exploration_initial_eps=0.5, exploration_final_eps=0.01, verbose=1)
    train.print_hyperparameters()
    train.train()
    metric = "system_total_waiting_time"
    show_results = ShowResults(result_folder, log_file_path, metric,1,int(num_of_episodes+1))
    show_results.main()

    # 邮件发送配置
    sender_email = '1811743445@qq.com'
    auth_code = 'zbfuppehtwtkejfg'
    recipient_email = 'zl22n23@soton.ac.uk'
    subject = 'Training Results'
    body = 'Please find the attached result files from the latest training run.'

    # 获取要发送的文件
    files_to_send = get_files_to_send(result_folder)

    # 发送邮件
    email_sender = EmailSender(sender_email, auth_code)
    email_sender.send_email(recipient_email, subject, body, attachments=files_to_send)

