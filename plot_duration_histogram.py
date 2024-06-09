import pandas as pd
import matplotlib.pyplot as plt

# Load the durations CSV
df = pd.read_csv('durations.csv')

# Plot the frequency distribution histogram for 'duration'
plt.hist(df['duration'], bins=50, color='blue', alpha=0.7)
plt.title('Frequency Distribution of Duration')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
