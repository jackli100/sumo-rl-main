import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files
df1 = pd.read_csv('durations1.csv')
df4 = pd.read_csv('durations2.csv')
df3 = pd.read_csv('durations3.csv')
df2 = pd.read_csv('durations4.csv')

# Define the bins
max_duration = 500
bins = np.arange(0, max_duration + 50, 50)  # Using 50 as the bin width

# Set up the histogram plot
plt.figure(figsize=(12, 6))

# Plot histograms for both datasets
n, bins, patches = plt.hist([df1['duration'], df2['duration'], df3['duration'], df4['duration']], \
                    bins=bins, label=['Trained', '200000~300000', '100000~200000','0~100000'])
# Calculate the middle points for bin labels
bin_centers = 0.5 * np.diff(bins) + bins[:-1]

# Add titles and labels
plt.title('Comparison of Duration Frequency Distributions by 50-unit Intervals')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.xticks(bin_centers, [f"{int(b)}-{int(b+50)}" for b in bins[:-1]], rotation=90)
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
