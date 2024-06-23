import numpy as np
import pandas as pd
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
        self.capacity_right = 1513
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
    def __init__(self, hyper_para_csv, output_folder, net_file, route_file, total_timesteps):
        self.output_folder = output_folder
        self.hyper_para_csv = hyper_para_csv
        self.params = self._load_params_from_csv()
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


    def _load_params_from_csv(self):
        with open(self.hyper_para_csv, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            params = [row[1] for row in reader if row]
        print(f"Loaded params: {params}")
        return params
    
    def train(self):
        env = SumoEnvironment(
            net_file=self.net_file,
            route_file=self.route_file,
            out_csv_name=self.out_csv_name,
            single_agent=True,
            use_gui=False,
            num_seconds=int(self.total_timesteps/5), 
        )
    
        model = DQN(
            env=env,
            policy="MlpPolicy",
            learning_rate = float(self.params[0]),
            learning_starts = int(self.params[1]),
            train_freq = int(self.params[2]),
            target_update_interval = int(self.params[3]),
            exploration_initial_eps = float(self.params[4]),
            exploration_final_eps = float(self.params[5]),
            verbose = int(self.params[6])
        )   

        model.learn(total_timesteps=self.total_timesteps)
        # Save the model
        model.save(self.model_save_path)
    
class ShowResults:
    def __init__(self, result_folder, log_file_path, metric):
        self.result_folder = result_folder
        self.log_file_path = log_file_path
        self.metric = metric

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
            file_paths = [os.path.join(self.result_folder, f"{file_prefix}{i}.csv") for i in range(1, 7)]
            
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
    

def generate_result_folder():
    # 获取当前时间
    now = datetime.now()
    # 格式化时间为小时和分钟
    hour = now.strftime("%H")
    minute = now.strftime("%M")
    # 生成文件夹名称
    result_folder = f"outputs/dqn-{hour}-{minute}"
    
    # 如果文件夹不存在则创建
    os.makedirs(result_folder, exist_ok=True)
    
    return result_folder

# 创建程序实例并运行
if __name__ == "__main__":

    result_folder = generate_result_folder()
    log_file_path = os.path.join(result_folder, 'output_log.txt')
    proportion_of_saturations = [0.75, 0.75, 0.75, 0.75]
    
    # step 1: generate the traffic matrix
    matrix = TrafficMatrix(proportion_of_saturations, result_folder)
    matrix.create_xml()
    
    # step 2: train the model
    route_path = matrix.output_file
    net_path = r"D:\trg1vr\sumo-rl-main\sumo-rl-main\sumo_rl\nets\2way-single-intersection\single-intersection-2.net.xml"
    total_timesteps = 10000
    hyper_para_csv = "experiments/hyper_para.csv"
    train = Train(hyper_para_csv, result_folder, net_path, route_path, total_timesteps)
    train.train()

    # step 3: show the results
    metric = "system_total_waiting_time"
    show_results = ShowResults(result_folder, log_file_path, metric)
    show_results.main()

