import os
import pickle
import random

def merge_dict_pkls(folder_path, output_name):
    merged_dict = {}
    output_path = os.path.join(folder_path, output_name)

    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl") and filename != output_name:
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                try:
                    data = pickle.load(f)
                    print(filename)
                    print(len(data))
                    if isinstance(data, dict):
                        merged_dict.update(data)
                except Exception as e:
                    print(" error occur")

    keys = list(merged_dict.keys())
    random.shuffle(keys)
    shuffled_dict = {k: merged_dict[k] for k in keys}

    with open(output_path, 'wb') as f:
        pickle.dump(shuffled_dict, f)

    print(f"✅ finish：{output_path}， {len(shuffled_dict)} keys")


base_dir = '/home/user/database/gy/GSGP/STA/dataset/processed_data'


for split in ['train', 'val', 'test']:
    folder = os.path.join(base_dir, split)
    output_file = f"{split}.pkl"
    merge_dict_pkls(folder, output_file)
