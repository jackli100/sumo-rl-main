import os
import pandas as pd
import json
import re
import shutil

def convert_csv_to_json(src_folder, dest_folder):
    """
    Convert CSV files to JSON format and save them in the destination folder.

    Args:
        src_folder (str): The path to the folder containing the CSV files.
        dest_folder (str): The path to the folder where the JSON files will be saved.

    Returns:
        None
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    for file_name in os.listdir(src_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(src_folder, file_name)
            df = pd.read_csv(file_path)
            json_data = df.to_json(orient='records', force_ascii=False)
            json_file_name = file_name.replace('.csv', '.json')
            json_file_path = os.path.join(dest_folder, json_file_name)
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json_file.write(json_data)
            print(f"Converted {file_name} to {json_file_name}")

def clean_whitespace_in_json(file_path, dest_folder):
    """
    Cleans the whitespace in a JSON file and saves the cleaned JSON to a destination folder.

    Args:
        file_path (str): The path to the JSON file to be cleaned.
        dest_folder (str): The destination folder where the cleaned JSON will be saved.

    Returns:
        None

    Raises:
        json.JSONDecodeError: If there is an error decoding the JSON from the file.

    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
        return

    cleaned_data = []
    for entry in data:
        cleaned_entry = {}
        for key, value in entry.items():
            cleaned_key = key.strip()
            cleaned_value = value.strip() if value is not None else value
            cleaned_entry[cleaned_key] = cleaned_value
        cleaned_data.append(cleaned_entry)

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    file_name = os.path.basename(file_path)
    new_file_path = os.path.join(dest_folder, file_name)

    with open(new_file_path, 'w', encoding='utf-8') as file:
        json.dump(cleaned_data, file, ensure_ascii=False, indent=4)

    print(f"Processed and saved cleaned JSON to {new_file_path}")

def process_json_folder(src_folder, dest_folder):
    for root, dirs, files in os.walk(src_folder):
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                clean_whitespace_in_json(file_path, dest_folder)

def remove_episode_json_files(folder):
    for file_name in os.listdir(folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                if any(entry.get('Parameter') == 'num_of_episodes' and entry.get('Value') != '9' for entry in data):
                    os.remove(file_path)
                    print(f"Removed {file_name} due to non-zero episodes")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file_path}: {e}")

def extract_info_and_convert_to_dict(file_path):
    pattern = r'(\d{2}-\d{2}-\d{2}-\d{4})'
    match = re.search(pattern, file_path)
    
    if not match:
        return None
    
    extracted_string = match.group(1)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError:
        return None
    
    relevant_info = {}
    for entry in data:
        try:
            parameter = entry['Parameter'].strip()
            value = entry['Value']
            if value is not None:
                value = value.strip()
            if parameter in ['net_path', 'fix_ts', 'proportion_of_saturations']:
                relevant_info[parameter] = value
        except (KeyError, AttributeError):
            continue
    
    if not all(key in relevant_info for key in ['net_path', 'fix_ts', 'proportion_of_saturations']):
        return None
    
    try:
        proportion_of_saturations = json.loads(relevant_info['proportion_of_saturations'])[0]
        net_path = relevant_info['net_path'].split('-')[-1][0]
        fix_ts = relevant_info['fix_ts'][0]
    except Exception:
        return None
    
    result_string = f"{proportion_of_saturations}-{net_path}-{fix_ts}"
    result_dict = {extracted_string: result_string}
    
    return result_dict

def process_folder(src_folder):
    result_list = []
    
    for root, dirs, files in os.walk(src_folder):
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                result_dict = extract_info_and_convert_to_dict(file_path)
                if result_dict:
                    result_list.append(result_dict)
    
    return result_list

def copy_and_rename_zip_files_with_dict(src_folder, dest_folder, rename_dict):
    pattern = r'(\d{2}-\d{2}-\d{2}-\d{4})'
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for root, dirs, files in os.walk(src_folder):
        for file_name in files:
            if file_name.endswith('.zip'):
                match = re.search(pattern, file_name)
                if match:
                    extracted_string = match.group(1)
                    if extracted_string in rename_dict:
                        new_file_name = rename_dict[extracted_string] + os.path.splitext(file_name)[1]
                        old_file_path = os.path.join(root, file_name)
                        new_file_path = os.path.join(dest_folder, new_file_name)
                        print(f"Copying '{old_file_path}' to '{new_file_path}'")
                        shutil.copy2(old_file_path, new_file_path)
                        print(f"Copied and renamed '{file_name}' to '{new_file_name}'")
                    else:
                        print(f"Pattern '{extracted_string}' not found in rename_dict")
                else:
                    print(f"No match found in file name: {file_name}")

if __name__ == "__main__":
    csv_src_folder = r'D:\trg1vr\sumo-rl-main\sumo-rl-main\results'
    json_dest_folder = 'D:\\trg1vr\\sumo-rl-main\\sumo-rl-main\\result_in_json'
    convert_csv_to_json(csv_src_folder, json_dest_folder)

    cleaned_json_folder = 'D:\\trg1vr\\sumo-rl-main\\sumo-rl-main\\cleaned_result_in_json'
    process_json_folder(json_dest_folder, cleaned_json_folder)

    # 删除那些"num_of_episodes"不等于0的JSON文件
    remove_episode_json_files(cleaned_json_folder)

    result_list = process_folder(cleaned_json_folder)
    rename_dict = {list(d.keys())[0]: list(d.values())[0] for d in result_list}

    print("Rename Dictionary:")
    for key, value in rename_dict.items():
        print(f"{key}: {value}")

    zip_src_folder = r'D:\trg1vr\sumo-rl-main\sumo-rl-main\models'
    zip_dest_folder = r'D:\trg1vr\sumo-rl-main\sumo-rl-main\renamed_models'
    copy_and_rename_zip_files_with_dict(zip_src_folder, zip_dest_folder, rename_dict)
