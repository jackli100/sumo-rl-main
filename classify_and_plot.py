import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_saturation_value(proportion_of_saturations):
    """
    Extracts the saturation value from the given proportion of saturations.

    Args:
        proportion_of_saturations (str): A string representing the proportion of saturations.

    Returns:
        float or None: The extracted saturation value, or None if no match is found.
    """
    match = re.search(r'\[([\d\.]+)', proportion_of_saturations)
    if match:
        return float(match.group(1))
    return None

def categorize_files(folder_path, target_saturations):
    """
    Categorizes files based on their saturation values and specific conditions.

    Args:
        folder_path (str): The path to the folder containing the files.
        target_saturations (list): A list of target saturation values.

    Returns:
        dict: A dictionary containing categorized files based on saturation values and conditions.
            The keys are the target saturation values, and the values are lists of file paths.
            The values are lists of length 5, where:
            - Index 0: Two-stage, no fix_ts
            - Index 1: Four-stage, no fix_ts
            - Index 2: Two-stage, fix_ts
            - Index 3: Four-stage, fix_ts
            - Index 4: 17-stage
            The file paths are categorized based on specific conditions.

    """
    categorized_files = {saturation: [None, None, None, None, None] for saturation in target_saturations}

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Convert data to a dictionary for easy parameter lookup
            data_dict = {item['Parameter']: item['Value'] for item in data}
            
            if 'proportion_of_saturations' in data_dict:
                proportion_of_saturations = data_dict['proportion_of_saturations']
                saturation_value = extract_saturation_value(proportion_of_saturations)

                if saturation_value in target_saturations:
                    fix_ts = data_dict.get('fix_ts', None)
                    net_path = data_dict.get('net_path', '')

                    if '-3' in net_path and fix_ts == 'False':
                        # two-stage, no fix_ts
                        categorized_files[saturation_value][0] = file_path
                    elif '-4' in net_path and fix_ts == 'False':
                        # four-stage, no fix_ts
                        categorized_files[saturation_value][1] = file_path
                    elif '-3' in net_path and fix_ts == 'True':
                        # two-stage, fix_ts
                        categorized_files[saturation_value][2] = file_path
                    elif '-4' in net_path and fix_ts == 'True':
                        # four-stage, fix_ts
                        categorized_files[saturation_value][3] = file_path
                    elif '-2' in net_path:
                        # 17-stage
                        categorized_files[saturation_value][4] = file_path

    return categorized_files

def get_overall_average_system_average_waiting_time(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    for item in json_data:
        if item["Parameter"] == "Overall Average system_mean_waiting_time":
            return float(item["Value"])
    return None

def get_overall_average_system_total_stopped(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    for item in json_data:
        if item["Parameter"] == "Overall Average system_total_stopped":
            return float(item["Value"])
    return None

def get_overall_average_system_total_waiting_time(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    for item in json_data:
        if item["Parameter"] == "Overall Average system_total_waiting_time":
            return float(item["Value"])
    return None

def convert_paths_to_tuples(categorized_files):
    """
    Converts file paths in the categorized_files dictionary to tuples of metrics.

    Args:
        categorized_files (dict): A dictionary containing categorized file paths.

    Returns:
        dict: A dictionary with file paths converted to tuples of metrics, where:
            - Index 0: Average system_total_waiting_time
            - Index 1: Average system_total_stopped
            - Index 2: Average system_average_waiting_time
            
    """
    for saturation in categorized_files:
        for i in range(len(categorized_files[saturation])):
            file_path = categorized_files[saturation][i]
            if file_path is not None:
                total_waiting_time = get_overall_average_system_total_waiting_time(file_path)
                average_stopped = get_overall_average_system_total_stopped(file_path)
                average_waiting_time = get_overall_average_system_average_waiting_time(file_path)
                categorized_files[saturation][i] = (total_waiting_time, average_stopped, average_waiting_time)
            else:
                categorized_files[saturation][i] = (None, None, None)
    return categorized_files

def plot_metrics(categorized_files, metric_index):
    """
    Plots the specified metric across different saturation values.

    Args:
    categorized_files (dict): A dictionary containing categorized files based on saturation values and conditions.
    metric_index (int): Index of the metric to plot (0 for total stopped, 1 for median stopped, 2 for average waiting time).

    The values are lists of length 5, where:
        - Index 0: Two-stage, no fix_ts
        - Index 1: Four-stage, no fix_ts
        - Index 2: Two-stage, fix_ts
        - Index 3: Four-stage, fix_ts
        - Index 4: 17-stage
    The file paths are categorized based on specific conditions.
    """
    if metric_index not in [0, 1, 2]:
        raise ValueError("Invalid metric_index. It should be 0, 1, or 2.")

    metrics_labels = ["Average system_total_stopped", "Median system_total_stopped", "Average system_total_waiting_time"]
    metric_name = metrics_labels[metric_index]

    saturation_values = sorted(categorized_files.keys())
    labels = ["Two-stage, no fix_ts", "Four-stage, no fix_ts", "Two-stage, fix_ts", "Four-stage, fix_ts", "17-stage"]

    for idx in range(5):
        y_values = []
        for saturation in saturation_values:
            file_data = categorized_files[saturation][idx]
            if file_data is not None:
                y_values.append(file_data[metric_index])
            else:
                y_values.append(None)
        
        # 如果 y_values 全是 None，跳过这个类别
        if all(value is None for value in y_values):
            continue
        
        plt.plot(saturation_values, y_values, label=labels[idx])

    plt.xlabel('Saturation')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Saturation')
    plt.legend()
    plt.grid(True)
    plt.show()

def clean_data(categorized_files):
    for saturation in categorized_files:
        for i in range(len(categorized_files[saturation])):
            file_data = categorized_files[saturation][i]
            if file_data is not None:
                if any(value is not None and value > 10000 for value in file_data):
                    categorized_files[saturation][i] = (None, None, None)
    return categorized_files

if __name__ == '__main__':
    # Example usage:
    folder_path = r'D:\trg1vr\sumo-rl-main\sumo-rl-main\cleaned_result_evaluate_in_json'  # Replace with your folder path
    target_saturations = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    categorized_files = categorize_files(folder_path, target_saturations)
    converted_files = convert_paths_to_tuples(categorized_files)
    converted_files = clean_data(converted_files)

    # 打印转换后的结果
    for saturation, files in converted_files.items():
        print(f"Saturation {saturation}: {files}")
        print(f"Length of tuple for saturation {saturation}: {len(files)}")    

    plot_metrics(converted_files, 2)  # Plot total stopped