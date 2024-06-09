import xml.etree.ElementTree as ET
import pandas as pd

# Load and parse the XML file
file_path = r'D:\trg1vr\sumo-rl-main\sumo-rl-main\outputs\2way-single-intersection-l0.0001-small\vehicles.xml'  # 修改此路径为你的本地文件路径
tree = ET.parse(file_path)
root = tree.getroot()

# Extract 'duration' values from the XML
durations = []
for tripinfo in root.findall('tripinfo'):
    duration = float(tripinfo.get('duration'))
    durations.append(duration)

# Create a DataFrame from the extracted durations
df = pd.DataFrame(durations, columns=['duration'])

# Save to CSV
csv_path = 'durations.csv'
df.to_csv(csv_path, index=False)

print(f'Duration data saved to {csv_path}')
