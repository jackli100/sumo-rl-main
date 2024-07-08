import os
import zipfile
import json

def check_zip_files_for_total_timesteps(src_folder, required_timesteps):
    missing_timesteps_files = []
    satisfying_files = []

    for root, dirs, files in os.walk(src_folder):
        for file_name in files:
            if file_name.endswith('.zip'):
                file_path = os.path.join(root, file_name)
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_file:
                        file_contains_required_timesteps = False
                        for file in zip_file.namelist():
                            if file == 'data':  # 文件名为data（无后缀）
                                with zip_file.open(file) as json_file:
                                    data = json.load(json_file)
                                    if '_total_timesteps' in data and data['_total_timesteps'] == required_timesteps:
                                        file_contains_required_timesteps = True
                                        break
                        if file_contains_required_timesteps:
                            satisfying_files.append(file_name)
                        else:
                            missing_timesteps_files.append(file_name)
                except zipfile.BadZipFile:
                    print(f"Error: {file_path} is a bad zip file")
                except json.JSONDecodeError:
                    print(f"Error: {file_path} contains invalid JSON")

    return missing_timesteps_files, satisfying_files

if __name__ == "__main__":
    zip_src_folder = r'D:\trg1vr\sumo-rl-main\sumo-rl-main\renamed_models'
    required_timesteps = 900000
    
    missing_timesteps_files, satisfying_files = check_zip_files_for_total_timesteps(zip_src_folder, required_timesteps)

    if missing_timesteps_files:
        print("ZIP files missing the '_total_timesteps': 900000 field:")
        for file in missing_timesteps_files:
            print(file)
    else:
        print("All ZIP files contain the '_total_timesteps': 900000 field.")

    if satisfying_files:
        print("\nZIP files containing the '_total_timesteps': 900000 field:")
        for file in satisfying_files:
            print(file)
        print(f"\nNumber of ZIP files containing the '_total_timesteps': 900000 field: {len(satisfying_files)}")
