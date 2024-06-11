import xml.etree.ElementTree as ET
import pandas as pd

# Load and parse the XML file
file_path = r'outputs\l0.0001-more-than-half-trained-10e5\tripinfos_20240610232747.xml'  # 修改此路径为你的本地文件路径
tree = ET.parse(file_path)
root = tree.getroot()

# Extract 'duration' values from the XML for depart times < 29000 seconds
durations = []
for tripinfo in root.findall('tripinfo'):
    depart = float(tripinfo.get('depart'))
    duration = float(tripinfo.get('duration'))
    durations.append(duration)

# Create a DataFrame from the extracted durations
df = pd.DataFrame(durations, columns=['duration'])

# Save to CSV
csv_path = 'durations4.csv' # 1 --- TRAINED 2 --- UNTRAINED 3 --- HALF-TRAINED 4 --- MORE THAN HALF TRAINED
df.to_csv(csv_path, index=False)
print(f'Duration data saved to {csv_path}')
