"""
tf.data.Dataset based dataloader for the BridgeData format, meaning
TFRecords with one trajectory per example. See the BridgeDataset class
below for more details.

Written by Kevin Black (kvablack@berkeley.edu).
"""

from typing import Iterable, List, Optional, Union, Mapping
import dlimp as dl
import json
import numpy as np
import tensorflow as tf
from functools import partial
from octo.data import obs_transforms

def apply_frame_transforms(
        dataset: tf.data.Dataset,
        train: bool,
        frame_transform_kwargs: Union[dict, Mapping[str, dict]] = {},
        num_parallel_calls: int = tf.data.AUTOTUNE,
    )-> tf.data.Dataset:

    resize_shapes = frame_transform_kwargs.get('resize_size', {})
    augment_kwargs = frame_transform_kwargs.get('image_augment_kwargs', {})
    # Image augmentations: TODO (They currently don't work)
    # To fix it either use ensure_shape or use the augmentations from tf_augmentations.py
    # if train:
    #     def aug(frame: dict):
    #         seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
    #         frame['observation'] = obs_transforms.augment(frame['observation'], seed=seed, augment_kwargs=augment_kwargs)
    #         return frame

    #     dataset = dataset.map(aug, num_parallel_calls=num_parallel_calls)

    def resize_image(data):
        for k in resize_shapes.keys():
            if f'image_{k}' in data['observation']:
                data['observation'][f'image_{k}'] = dl.transforms.resize_image(data['observation'][f'image_{k}'], size=resize_shapes[k])
        return data
    dataset = dataset.map(resize_image, num_parallel_calls=num_parallel_calls)

    # add 1 extra dimension to observations
    def add_extra_dim(data):
        for k in data['observation'].keys():
            if isinstance(data['observation'][k], dict):
                continue
            data['observation'][k] = tf.expand_dims(data['observation'][k], axis=0)
        return data
    
    dataset = dataset.map(add_extra_dim, num_parallel_calls=num_parallel_calls)

    return dataset

def safe_decode_image(image, image_size=(256, 256), dtype=tf.uint8):
    assert image.dtype == tf.string
    if tf.strings.length(image) == 0:
        return tf.zeros((*image_size, 3), dtype=dtype)
    return tf.io.decode_image(image, expand_animations=False, channels=3, dtype=dtype)

def get_dtype_from_key(key):
    if 'mask' in key:
        return tf.bool
    elif 'image' in key:
        return tf.uint8
    elif 'timestep' in key:
        return tf.int32
    elif ('dataset_name' in key) or ('language' in key):
        return tf.string
    else:
        return tf.float32

def get_all_dataset_keys(path):
    dataset = tf.data.TFRecordDataset(path, num_parallel_reads=tf.data.AUTOTUNE)
    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        all_dataset_keys = list(example.features.feature.keys())

    return all_dataset_keys

class RetrievedOctoDataset:
    """
    Args:
        data_paths: List of paths to the data files. If a list of list of paths
            is provided, the data will be sampled from each sub-list according
            to "sample_weights".
        batch_size: Batch size.
    """

    def __init__(
        self,
        data_paths: List[Union[str, List[str]]],
        load_keys: List[str],
        dataset_statistics_path: str,
        seed: int = 0,
        sample_weights: Optional[List[float]] = None,
        batch_size: int = 256,
        train=True,
        shuffle_buffer_size: int = 10000,
        frame_transform_kwargs: Union[dict, Mapping[str, dict]] = {},
        **kwargs,
    ):
        if isinstance(data_paths[0], str):
            data_paths = [data_paths]
        
        if isinstance(load_keys, str) and load_keys=='all':
            load_keys = get_all_dataset_keys(data_paths[0][0])

        print("\n==> Loading keys", load_keys, f"from {data_paths}")

        self.PROTO_TYPE_SPEC = {
            k: get_dtype_from_key(k) for k in load_keys
        }

        if sample_weights is None:
            # default to uniform distribution over sub-lists
            sample_weights = [1 / len(data_paths)] * len(data_paths)
        assert len(data_paths) == len(sample_weights), f"data_paths({len(data_paths)}) and sample_weights({len(sample_weights)}) must have the same length"
        assert np.isclose(sum(sample_weights), 1.0)

        # construct a dataset for each sub-list of paths
        datasets = []
        for sub_data_paths in data_paths:
            datasets.append(self._construct_tf_dataset(sub_data_paths, seed=seed))

        if train:
            # shuffle and repeat each sub-dataset, allocating the shuffle buffer
            # by sample_weights
            for i in range(len(datasets)):
                datasets[i] = (
                    datasets[i]
                    .shuffle(int(shuffle_buffer_size * sample_weights[i]), seed + i)
                    .repeat()
                )

        dataset = tf.data.Dataset.sample_from_datasets(
            datasets, sample_weights,# seed=seed, stop_on_empty_dataset=train
        ).shuffle(shuffle_buffer_size)

        if train:
            # dataset = dataset.enumerate()
            dataset = apply_frame_transforms(
                    dataset, train, frame_transform_kwargs, num_parallel_calls=tf.data.AUTOTUNE)
        
        if train:
            dataset = dataset.batch(
                batch_size,
                num_parallel_calls=tf.data.AUTOTUNE,
                drop_remainder=True,
                deterministic=False,
            )
        else:
            dataset = dataset.batch(
                batch_size,
                num_parallel_calls=tf.data.AUTOTUNE,
                drop_remainder=False,
                deterministic=True,
            )

        self.tf_dataset = dataset

        with tf.io.gfile.GFile(dataset_statistics_path, "r") as f:
            self.dataset_statistics = json.load(f)

    def _construct_tf_dataset(self, paths: List[str], seed) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of paths.
        The dataset yields a dictionary of tensors for each transition.
        """

        # yields raw serialized examples
        dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=tf.data.AUTOTUNE).shuffle(len(paths), seed)

        # yields trajectories
        dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset

    def _decode_example(self, example_proto):
        # decode the example proto according to PROTO_TYPE_SPEC
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in self.PROTO_TYPE_SPEC.keys()
        }

        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_tensors = {
            key: tf.io.parse_tensor(parsed_features[key], dtype) \
                if key not in ['observation/image_primary', 'observation/image_wrist'] else safe_decode_image(parsed_features[key])
            for key, dtype in self.PROTO_TYPE_SPEC.items()
        }

        def parse_string_into_nested_dict(parsed_tensors):
            '''
                parsed tensors can have keys in the form of "a/b/c".
                This function will convert them into nested dictionaries splitting by "/".
            '''
            nested_dict = {}
            for key, value in parsed_tensors.items():
                keys = key.split('/')
                temp_dict = nested_dict
                for k in keys[:-1]:
                    if k not in temp_dict:
                        temp_dict[k] = {}
                    temp_dict = temp_dict[k]
                temp_dict[keys[-1]] = value
            return nested_dict

        parsed_tensors = parse_string_into_nested_dict(parsed_tensors)

        return {
            key: value for key, value in parsed_tensors.items()
        }

    def iterator(self):
        return self.tf_dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()


def print_nested_dict(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print('\t' * indent, k)
            print_nested_dict(v, indent + 1)
        else:
            try:
                print('\t' * indent, k, v.shape)
            except:
                pass

if __name__=='__main__':
    import os
    data_dir = '/mnt/hdd1/baselines'
    
    method_name = 'flow'
    target_train_path = os.path.join(data_dir, 'target_data', 'easy_pick_dataset_n10_h8_prechunk', 'train', 'out.tfrecord')
    prior_train_path = os.path.join(data_dir, 'retrieved_data', 'easy_pick_dataset_n10_h8_prechunk_th0.005', method_name, 'out.tfrecord')

    # dataset = tf.data.TFRecordDataset(prior_train_path, num_parallel_reads=tf.data.AUTOTUNE)
    # for raw_record in dataset.take(1):
    #     example = tf.train.Example()
    #     example.ParseFromString(raw_record.numpy())
    #     all_dataset_keys = list(example.features.feature.keys())
    
    data_paths = [[prior_train_path], [target_train_path]]
    sample_weights = [0.5, 0.5]

    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    
    frame_transform_kwargs = {
        'resize_size': {'primary': (256, 256), 'wrist': (128, 128)},
        'image_augment_kwargs': {'primary': workspace_augment_kwargs, 'wrist': wrist_augment_kwargs}
    }
    dataset = RetrievedOctoDataset(
                            data_paths,
                            train=True,
                            sample_weights=sample_weights, 
                            seed=0, 
                            batch_size=256, 
                            load_keys='all',
                            frame_transform_kwargs=frame_transform_kwargs
                        )
    
    iter = dataset.iterator()
    for i, data in enumerate(iter):
        print_nested_dict(data)
        exit(0)

        names = {}
        for name in data['dataset_name']:
            n = name.decode('utf-8')
            if n not in names:
                names[n] = 0
            names[n] += 1

        print(names)
        print()
        if (i+1) % 3 == 0:
            break