import json
import os
import numpy as np
import random

def json_ext(path, flag):
    if flag is 'train':
        dataset_path = os.path.join(path, 'train', 'mix')# read Train mixed speech name
        json_path = os.path.join(path, 'Json', 'train')
        os.makedirs(json_path, exist_ok=True)
    else:
        dataset_path = os.path.join(path, 'cv', 'mix')# read CV mixed speech name 
        json_path = os.path.join(path, 'Json', 'cv')
        os.makedirs(json_path, exist_ok=True)
    data_dir = os.listdir(dataset_path)
    file_num = len(data_dir)
    random.shuffle(data_dir)
    data_list = []

    for i in range(file_num):
        file_name = data_dir[i]
        file_name = os.path.splitext(file_name)[0]
        data_list.append(file_name)

    with open(os.path.join(json_path, 'files.json'), 'w') as f :
        json.dump(data_list, f, indent=4)

file_path = 'xxx'
json_ext(file_path, flag='train')
json_ext(file_path, flag='cv')