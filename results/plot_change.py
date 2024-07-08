import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_waiting_time_from_csv(file_path):
    data = pd.read_csv(file_path)
    waiting_time_data = data[data['Parameter'].str.contains('Average system_total_waiting_time for episode')]
    waiting_time_data['Episode'] = waiting_time_data['Parameter'].str.extract('(\d+)').astype(int)
    waiting_time_data['Average_system_total_waiting_time'] = waiting_time_data['Value'].astype(float)
    waiting_time_data = waiting_time_data.sort_values('Episode')

    plt.figure(figsize=(10, 6))
    plt.plot(waiting_time_data['Episode'], waiting_time_data['Average_system_total_waiting_time'], marker='o', linestyle='-')
    plt.title(f'Average system_total_waiting_time for episodes 1-9')
    plt.xlabel('Episode')
    plt.ylabel('Average system_total_waiting_time')
    plt.grid(True)
    plt.xticks(waiting_time_data['Episode'])
    plt.ylim(0, max(waiting_time_data['Average_system_total_waiting_time']) + 100)

    save_path = file_path.replace('.csv', '-average-waiting-time-change.png')
    plt.savefig(save_path)
    plt.close()

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            save_path = file_path.replace('.csv', '-average-waiting-time-change.png')
            if not os.path.exists(save_path):
                file_path)
                print(f'Generated plot for {filename}')
            else:
                print(f'Plot already exists for {filename}, skipping.')

# Example usage:
folder_path = '/path/to/your/folde'
process_folder(folder_path)
