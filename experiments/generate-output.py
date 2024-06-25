from train_class import ShowResults
import os
import sys

Result_folder = r"D:\trg1vr\sumo-rl-main\sumo-rl-main\outputs\2024-06-24\15-45-09-cJ43Qy"
log_file_path = os.path.join(Result_folder, 'output_log.txt')
metric = "system_total_waiting_time"
show_results = ShowResults(Result_folder, log_file_path, metric,1,6)
show_results.main()