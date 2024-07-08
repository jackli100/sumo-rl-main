import os
import pandas as pd
import re

def clean_column_names(columns):
    return [col.strip() for col in columns]

def extract_naming_info(data):
    # Clean column names
    data.columns = clean_column_names(data.columns)
    
    # Check if the required columns are present
    if 'Parameter' not in data.columns or 'Value' not in data.columns:
        print(f"Data does not contain the required columns: {data.columns}")
        return None

    # Determine XX from the proportion_of_saturations (two digits after the decimal)
    saturation_levels = data.loc[data['Parameter'] == 'proportion_of_saturations', 'Value'].values[0]
    XX = saturation_levels.strip('[]').split(',')[0].strip()
    XX = XX.split('.')[-1].ljust(2, '0')  # Extracting the digits after the decimal and ensuring two digits
    
    # Extract YY from the net_path
    net_path = data.loc[data['Parameter'] == 'net_path', 'Value'].values[0]
    if '-3' in net_path:
        YY = '02'
    elif '-4' in net_path:
        YY = '04'
    elif '-2' in net_path:
        YY = '17'
    else:
        YY = '00'  # Default value if no match is found
    
    # Determine Y/N from the presence of fix_ts with True value
    if any((data['Parameter'] == 'fix_ts') & (data['Value'] == 'True')):
        YN = 'Y'
    else:
        YN = 'N'
    
    # Combine to form the naming string
    naming_string = f"{XX}-{YY}-{YN}"
    return naming_string

def generate_name_mapping(directory):
    name_mapping = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            try:
                # Extract numeric parts from the original filename
                original_numbers = '-'.join(re.findall(r'\d+', filename))
                filepath = os.path.join(directory, filename)
                data = pd.read_csv(filepath)
                new_name = extract_naming_info(data)
                if new_name is not None:
                    # Only store numeric parts of the original and new names
                    new_numbers = '-'.join(re.findall(r'\d+', new_name))
                    name_mapping[original_numbers] = new_numbers
                else:
                    print(f"Skipping file {filename} due to missing required columns")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    return name_mapping

# Use the function to generate name mapping in the specified directory
directory_path = r'D:\trg1vr\sumo-rl-main\sumo-rl-main\results'  # Replace with your directory path

# Generate and print the name mapping
name_mapping = generate_name_mapping(directory_path)
print(name_mapping)
