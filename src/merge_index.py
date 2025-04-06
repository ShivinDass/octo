import os
import json
from pathlib import Path

from ipdb import set_trace as bp
from IPython import embed

from tqdm import tqdm
from copy import deepcopy

from ipdb import set_trace as bp
from IPython import embed

TRAIN=True
# TRAIN=False

def main():
    # MOUNT = "xfs"
    MOUNT = "nfs"
    if TRAIN:
        out_root = Path(f"/mnt/{MOUNT}/home/alaakh/store/oxe/mpt_dataset/train")
    else:
        out_root = Path(f"/mnt/{MOUNT}/home/alaakh/store/oxe/mpt_dataset/val")

    job_id = int(os.environ.get('JOB_ID', None))

    all_datasets = os.listdir(out_root)
    ds_name = all_datasets[job_id]

    # ds_name = FLAGS.config.dataset_kwargs.name
    dataset_dir = out_root / ds_name

    indices_json = list(dataset_dir.rglob('index.json'))
    parent_json_path = dataset_dir / 'index.json'
    if parent_json_path.exists():
        raise FileExistsError('Delete the old `index.json` file then re-run the code')


    parent_data = {
        'shards': [],
        'version': 2
    }

    for json_path in tqdm(indices_json):
        assert json_path.exists()

        with open(json_path, 'r') as f:
            data_child = json.load(f)

        assert data_child.keys() == parent_data.keys()

        for sample in data_child['shards']:
            correct_path_sample = deepcopy(sample)

            new_dir = str(json_path).replace('/index.json', '').replace(
                str(dataset_dir),
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

    parent_json_path = dataset_dir / 'index.json'

    if parent_json_path.exists():
        raise FileExistsError('Delete the old `index.json` file then re-run the code')

    with open(parent_json_path, 'w') as f:
        json.dump(parent_data, f)


    print('=> Done')

if __name__ == "__main__":
    main()