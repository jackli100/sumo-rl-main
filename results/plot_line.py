import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def extract_saturation_value(proportion_of_saturations):
    match = re.search(r'\[([\d\.]+)', proportion_of_saturations)
    if match:
        return float(match.group(1))
    return None

def plot_waiting_time_from_csv_list(file_paths, labels=['DQN', 'fixed time'], saturation_value=None, stage=None):
    plt.figure(figsize=(10, 6))

    for file_path, label in zip(file_paths, labels):
        if file_path is not None:  # Check if the file_path is not None
            data = pd.read_csv(file_path)
            waiting_time_data = data[data['Parameter'].str.contains('Average system_total_waiting_time for episode')]
            waiting_time_data['Episode'] = waiting_time_data['Parameter'].str.extract('(\d+)').astype(int)
            waiting_time_data['Average_system_total_waiting_time'] = waiting_time_data['Value'].astype(float)
            waiting_time_data = waiting_time_data.sort_values('Episode')

            plt.plot(waiting_time_data['Episode'], waiting_time_data['Average_system_total_waiting_time'], marker='o', linestyle='-', label=label)

    plt.title(f'Average system_total_waiting_time for episodes 1-9\nSaturation: {saturation_value}, Stage: {stage}')
    plt.xlabel('Episode')
    plt.ylabel('Average system_total_waiting_time')
    plt.grid(True)
    plt.xticks(waiting_time_data['Episode'])
    plt.ylim(0, max(waiting_time_data['Average_system_total_waiting_time']) + 100)
    plt.legend()
    
    # Save the plot to the current directory
    file_name = f"Saturation_{saturation_value}_{stage}.png"
    plt.savefig(file_name)
    plt.close()

def categorize_files(folder_path):
    categorized_files = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            
            # Check if the required columns exist
            if 'Parameter' in data.columns and 'Value' in data.columns:
                proportion_of_saturations = data.loc[data['Parameter'] == 'proportion_of_saturations', 'Value'].values[0]
                saturation_value = extract_saturation_value(proportion_of_saturations)
                
                if saturation_value in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
                    fix_ts = data.loc[data['Parameter'] == 'fix_ts', 'Value'].values[0] if 'fix_ts' in data['Parameter'].values else None
                    net_path = data.loc[data['Parameter'] == 'net_path', 'Value'].values[0] if 'net_path' in data['Parameter'].values else ''

                    if saturation_value not in categorized_files:
                        categorized_files[saturation_value] = {'two-stage': [], 'four-stage': []}

                    if fix_ts == 'True':
                        if '-3' in net_path:
                            categorized_files[saturation_value]['two-stage'].append((file_path, None))
                        elif '-4' in net_path:
                            categorized_files[saturation_value]['four-stage'].append((file_path, None))
                    else:
                        if '-3' in net_path:
                            categorized_files[saturation_value]['two-stage'].append((None, file_path))
                        elif '-4' in net_path:
                            categorized_files[saturation_value]['four-stage'].append((None, file_path))

    # Combine tuples with matching (fix_ts == 'True', fix_ts != 'True') for each category and saturation value
    combined_files = {}

    for saturation_value, stages in categorized_files.items():
        combined_files[saturation_value] = {'two-stage': [], 'four-stage': []}
        for stage in ['two-stage', 'four-stage']:
            fix_ts_true_files = [f[0] for f in stages[stage] if f[0] is not None]
            fix_ts_false_files = [f[1] for f in stages[stage] if f[1] is not None]

            for true_file, false_file in zip(fix_ts_true_files, fix_ts_false_files):
                combined_files[saturation_value][stage].append((true_file, false_file))
    
    return combined_files

# Example usage:
folder_path = r'D:\trg1vr\sumo-rl-main\sumo-rl-main\results'  # Replace with your folder path
combined_files = categorize_files(folder_path)

# Generate plots for each category
for saturation, categories in combined_files.items():
    for stage, file_pairs in categories.items():
        for file_pair in file_pairs:
            plot_waiting_time_from_csv_list(file_pair, labels=['fixed time', 'DQN'], saturation_value=saturation, stage=stage)
            print(f"Generated plot for saturation {saturation}, stage {stage}, files: {file_pair}")
