import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def read_and_concatenate_csv(file_paths):
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

def save_plot(steps, values, filename, ylabel, title):
    plt.figure()
    plt.plot(steps, values, label=ylabel)
    plt.xlabel('step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def main(metric):
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Log file setup
    log_filename = os.path.join(script_dir, 'output_log.txt')
    with open(log_filename, 'w') as log_file:
        log_file.write("Execution started at {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        # Find the relevant CSV files in the script directory
        file_prefix = "dqn_conn0_ep"
        file_paths = [os.path.join(script_dir, f"{file_prefix}{i}.csv") for i in range(1, 13)]
        
        # Ensure all file paths exist
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

        # Read and concatenate the CSV files
        combined_df = read_and_concatenate_csv(file_paths)
        combined_csv_path = os.path.join(script_dir, "combined.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        log_file.write("Combined CSV saved to {}\n".format(combined_csv_path))

        # Plot and save the selected metric for combined data
        steps = combined_df['step']
        values = combined_df[metric]
        combined_plot_filename = os.path.join(script_dir, f'{metric}_combined.png')
        save_plot(steps, values, combined_plot_filename, metric, f'{metric} (Combined)')
        log_file.write("Combined plot saved to {}\n".format(combined_plot_filename))

        # Plot and save the selected metric for the last episode
        last_episode_path = file_paths[-1]
        last_episode_df = pd.read_csv(last_episode_path)
        steps = last_episode_df['step']
        values = last_episode_df[metric]
        last_episode_plot_filename = os.path.join(script_dir, f'{metric}_last_episode.png')
        save_plot(steps, values, last_episode_plot_filename, metric, f'{metric} (Last Episode)')
        log_file.write("Last episode plot saved to {}\n".format(last_episode_plot_filename))

        # Calculate and log average value of the selected metric for the last episode
        average_value = values.mean()
        log_file.write(f'Average {metric} for the last episode: {average_value:.3f}\n')

        log_file.write("Execution finished at {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

if __name__ == "__main__":
    # User can select the metric they want to plot and analyze
    metric = 'system_total_waiting_time'  # or 'system_total_waiting_time' or 'system_mean_waiting_time'
    main(metric)
