import math
import pickle
import random
import os

all_dataset_ori = {}
all_dataset = {}
cell_type = "net_info"
target_name = ""
dataset_path = "../dataset/processed_data/{}.pkl".format(cell_type)
target_path = "../dataset/processed_data/"

use_ratio = 1
train_ratio = 0.7
test_ratio = 0.15
val_ratio = 0.15



with open(dataset_path, "rb") as file:
    temp_list = pickle.load(file)
    all_dataset_ori.update(temp_list)
file.close()

for cell in all_dataset_ori:
            
    if 'inNet' in all_dataset_ori[cell]:
        inNetInfo = all_dataset_ori[cell]['inNet']
        for item in inNetInfo:
            if item[-1] != 0:
                first_non_zero = next((i for i in range(4, 14) if item[i] != 0), None)
                str_dict = str(first_non_zero)
                all_dataset[cell] = all_dataset_ori[cell]


print("Total original samples get:", len(all_dataset))
keys = list(all_dataset.keys())
random.shuffle(keys)
subset_size = int(len(keys) * use_ratio)  
subset_keys = keys[:subset_size]  
subset_data = {k: all_dataset[k] for k in subset_keys} 

print(f"Subset ({use_ratio}) size: {len(subset_data)}")


train_size = int(len(subset_keys) * train_ratio)
test_size = int(len(subset_keys) * test_ratio)
val_size = len(subset_keys) - train_size - test_size 

train_keys = subset_keys[:train_size]
test_keys = subset_keys[train_size:train_size + test_size]
val_keys = subset_keys[train_size + test_size:]

train_data = {k: subset_data[k] for k in train_keys}
test_data = {k: subset_data[k] for k in test_keys}
val_data = {k: subset_data[k] for k in val_keys}


print(f"Train size (70% of 50%): {len(train_data)}")
print(f"Test size (15% of 50%): {len(test_data)}")
print(f"Validation size (15% of 50%): {len(val_data)}")

os.makedirs(target_path + 'train', exist_ok=True)
os.makedirs(target_path + 'test', exist_ok=True)
os.makedirs(target_path + 'val', exist_ok=True)

print(f"Train size (70% of 50%): {len(train_data)}")
print(f"Validation size (15% of 50%): {len(val_data)}")

with open(target_path + 'train/{}_{}.pkl'.format(cell_type, target_name), 'wb') as f:
    pickle.dump(train_data, f)
with open(target_path + 'val/{}_{}.pkl'.format(cell_type, target_name), 'wb') as f:
    pickle.dump(val_data, f)
with open(target_path + 'test/{}_{}.pkl'.format(cell_type, target_name), 'wb') as f:
    pickle.dump(test_data, f)

print("all done!")



