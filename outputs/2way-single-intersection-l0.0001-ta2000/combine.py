import pandas as pd

# Load the uploaded CSV files
file_paths = [
    r"outputs\2way-single-intersection-l0.0001-ta2000\dqn_conn0_ep1.csv",
    r"outputs\2way-single-intersection-l0.0001-ta2000\dqn_conn0_ep2.csv",
    r"outputs\2way-single-intersection-l0.0001-ta2000\dqn_conn0_ep3.csv",
    r"outputs\2way-single-intersection-l0.0001-ta2000\dqn_conn0_ep4.csv",
    r"outputs\2way-single-intersection-l0.0001-ta2000\dqn_conn0_ep5.csv",
    r"outputs\2way-single-intersection-l0.0001-ta2000\dqn_conn0_ep6.csv",
]

# Read the CSV files
dfs = [pd.read_csv(file_path) for file_path in file_paths]

# Add a cumulative step column to each dataframe
for i in range(1, len(dfs)):
    dfs[i]['step'] += dfs[i-1]['step'].max()

# Concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)
# Save the combined dataframe to a new CSV file
combined_df.to_csv("combined.csv", index=False)