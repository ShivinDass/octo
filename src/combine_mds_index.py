import json
import os
from pathlib import Path
from copy import deepcopy

parent_path = Path('/mnt/hdd2/libero/mpt_dataset/libero_w15_with_subopt_128x128/')

index = parent_path / 'index.json'

f = open(index, 'r')
data = json.load(f)

for d in data['shards']:
    print(d['raw_data']['basename'])
    print(int(d['raw_data']['basename'].split('/')[-2].split('_')[-1]))

exit(0)



indices_json = [parent_path / data_name / 'index.json' for data_name in os.listdir(parent_path)]

for path in indices_json:
    assert path.exists(), f"Path {path} does not exist"

parent_index_path = parent_path / 'index.json'
parent_data = {
    'shards': [],
    'version': 2
}

for json_path in indices_json:
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for sample in data['shards']:
        correct_path_sample = deepcopy(sample)

        new_dir = str(json_path).replace('/index.json', '').replace(
            str(parent_path),
            ''
        )[1:]


        old_raw_data_path = sample['raw_data']['basename']
        new_raw_data_path = os.path.join(new_dir, old_raw_data_path)
        correct_path_sample['raw_data']['basename'] = new_raw_data_path

        if sample['zip_data'] is not None:
            old_zip_data_path = sample['zip_data']['basename']
            new_zip_data_path = os.path.join(new_dir, old_zip_data_path)
            correct_path_sample['zip_data']['basename'] = new_zip_data_path

        parent_data['shards'].append(deepcopy(correct_path_sample))
        del correct_path_sample

print(len(parent_data['shards']), 'samples in total')

with open(parent_index_path, 'w') as f:
    json.dump(parent_data, f)