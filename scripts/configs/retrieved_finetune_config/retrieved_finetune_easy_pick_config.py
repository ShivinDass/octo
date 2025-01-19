from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
import os

def get_config(config_string="br,0.05"):
    method_name, percent = config_string.split(',')
    assert method_name in ['br', 'flow', 'action']

    mode, task = "full", "language_conditioned"
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    data_dir = '/mnt/hdd1/baselines'    
    # method_name = 'flow'
    target_train_path = os.path.join(data_dir, 'target_data/easy_pick_dataset_n10_h8_prechunk/train', 'out.tfrecord')
    prior_train_path = os.path.join(data_dir, f'retrieved_data/easy_pick_dataset_n10_h8_prechunk_th{percent}/{method_name}/out.tfrecord')


    FINETUNING_KWARGS = {
        "data_paths": [[prior_train_path], [target_train_path]],
        "sample_weights": [0.5, 0.5], 
        "load_keys": 'all',
        "dataset_statistics_path": "/home/shivin/tensorflow_datasets/easy_pick_dataset_n10/0.1.0/dataset_statistics_933870c48b51bb7e232bf2c96502b7f741fe1aa26495ddce694e6e809590bf28.json",

        # "name": "bridge_dataset",
        # "data_dir": "/home/shivin/tensorflow_datasets",#"/mnt/hdd1/traj_data",
        # "image_obs_keys": {"primary": "image_0", "wrist": None},
        # "state_obs_keys": ["state", None],
        # "language_key": "language_instruction",
        # "action_proprio_normalization_type": "normal",
        # # All actions are relative deltas, except for the last one (gripper) which is absolute
        # # Specifying this is only necessary if you want to predict > 1 step into the future
        # "absolute_action_mask": [False, False, False, False, False, False, True],
        # "action_normalization_mask": [True, True, True, True, True, True, False],
        # # standardize_fn is dynamically loaded from a file
        # # for example: "experiments/kevin/custom_standardization_transforms.py:aloha_dataset_transform"
        # "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:bridge_dataset_transform",
        # # If the default data loading speed is too slow, try these:
        # # "num_parallel_reads": 8,  # for reading from disk / GCS
        # # "num_parallel_calls": 16,  # for initial dataset construction
    }

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    elif mode == "frozen_transformer":
        frozen_keys = ("octo_transformer.BlockTransformer_0.*",)
    else:
        raise ValueError("Invalid mode")

    max_steps = FieldReference(50000)
    window_size = FieldReference(default=2)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=128,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=int(max_steps.get()//5),
        save_interval=int(max_steps.get()//5),
        save_dir='/home/shivin/foundation_models/experiments',
        seed=42,
        wandb=dict(
            project="octo_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=int(max_steps.get()//25),
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
    )

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
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
            "wrist": (128, 128),  # wrist camera is at 128x128
        },
        image_augment_kwargs=[
            workspace_augment_kwargs,
            wrist_augment_kwargs,
        ],
    )
    # If the default data loading speed is too slow, try these:
    config[
        "frame_transform_threads"
    ] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    config["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
