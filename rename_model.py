import os
import re
import shutil
from generate_dict_by_json import extract_info_and_convert_to_dict, process_folder

def copy_and_rename_zip_files_with_dict(src_folder, dest_folder, rename_dict):
    # Use regex to extract the dd-dd-dd-dddd pattern
    pattern = r'(\d{2}-\d{2}-\d{2}-\d{4})'
    
    # Create destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Traverse all files in the source folder
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
                        print(f"Copying '{old_file_path}' to '{new_file_path}'")  # Debug statement
                        shutil.copy2(old_file_path, new_file_path)
                        print(f"Copied and renamed '{file_name}' to '{new_file_name}'")
                    else:
                        print(f"Pattern '{extracted_string}' not found in rename_dict")
                else:
                    print(f"No match found in file name: {file_name}")

# Example usage
src_folder = r'D:\trg1vr\sumo-rl-main\sumo-rl-main\models'
dest_folder = r'D:\trg1vr\sumo-rl-main\sumo-rl-main\renamed_models'

# Generate the rename dictionary
cleaned_json_folder = 'D:\\trg1vr\\sumo-rl-main\\sumo-rl-main\\cleaned_result_in_json'
result_list = process_folder(cleaned_json_folder)
rename_dict = {list(d.keys())[0]: list(d.values())[0] for d in result_list}

# Display the rename dictionary for debugging purposes
print("Rename Dictionary:")
for key, value in rename_dict.items():
    print(f"{key}: {value}")

# Copy and rename zip files
copy_and_rename_zip_files_with_dict(src_folder, dest_folder, rename_dict)
