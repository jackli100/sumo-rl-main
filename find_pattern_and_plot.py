from json_tool import convert_csv_to_json, clean_whitespace_in_json,process_json_folder




if __name__ == "__main__":
    csv_src_folder = r'D:\trg1vr\sumo-rl-main\sumo-rl-main\result-evaluate'
    json_dest_folder = 'D:\\trg1vr\\sumo-rl-main\\sumo-rl-main\\result-evaluate_in_json'
    convert_csv_to_json(csv_src_folder, json_dest_folder)
    json_cleaned_folder = 'D:\\trg1vr\\sumo-rl-main\\sumo-rl-main\\cleaned_result_evaluate_in_json'
    process_json_folder(json_dest_folder,json_cleaned_folder)
    