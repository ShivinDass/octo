from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder


# def get_config(config_string="full,multimodal"): # old
def get_config(config_string="full,language_conditioned"): # new
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # Fill this in for your own dataset!

    # There should be two image keys
    # first image key should be the third-person view (None if not used)
    # and second image key should be the wrist view (None if not used)

    FINETUNING_KWARGS = {
        # "name": "bridge_dataset",
        # "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        # "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:bridge_dataset_transform",
        # "image_obs_keys": {"primary": "image_0", "wrist": None},
        # "state_obs_keys": ["state"],

        # "name": "utaustin_mutex",
        # "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        # "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:utaustin_mutex_dataset_transform",
        # "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        # # "state_obs_keys": ["state", None],
        # "state_obs_keys": ["state"],

        # "name": "austin_sailor_dataset_converted_externally_to_rlds",
        # "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        # "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:austin_sailor_dataset_transform",
        # "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        # # "state_obs_keys": ["state", None],
        # "state_obs_keys": ["state"],

        # "name": "austin_sirius_dataset_converted_externally_to_rlds",
        # "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        # "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:austin_sirius_dataset_transform",
        # "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        # # "state_obs_keys": ["state", None],
        # "state_obs_keys": ["state"],

        # "name": "austin_buds_dataset_converted_externally_to_rlds",
        # "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        # "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:austin_buds_dataset_transform",
        # "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        # # "state_obs_keys": ["state", None],
        # "state_obs_keys": ["state"],


        # "name": "jaco_play",
        # "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        # "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:jaco_play_dataset_transform",
        # "image_obs_keys": {"primary": "image", "wrist": "image_wrist"},
        # "state_obs_keys": ["state_eef", None, "state_gripper"],
        # # "state_obs_keys": ["state_eef"], ### why not this?


        # "name": "viola",
        # "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        # "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:viola_dataset_transform",
        # "image_obs_keys": {"primary": "agentview_rgb", "wrist": "eye_in_hand_rgb"},
        # "state_obs_keys": ["joint_states", "gripper_states"],
        # # "state_obs_keys": ["state_eef"], ### why not this?



        ################################################################
        #################### FOR RUNNING REAL WORLD ####################
        ################################################################ 
        
        # "exp_save_dir": "first_set",
        # "name": (
        #     "utaustin_mutex-"
        #     "austin_sailor_dataset_converted_externally_to_rlds-"
        #     "austin_sirius_dataset_converted_externally_to_rlds-"
        #     "austin_buds_dataset_converted_externally_to_rlds-"
        #     "easy_pick_dataset_n5_1"
        # ),
        # "data_dir": "/mnt/nfs/home/alaakh/store/oxe/mpt_dataset/train",
        # "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:austin_buds_dataset_transform",
        # "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        # # "state_obs_keys": ["state", None],
        # "state_obs_keys": ["state"],

        "name": "second_set",
        "all_train_datasets": (
            "austin_buds_dataset_converted_externally_to_rlds-"
            "austin_sailor_dataset_converted_externally_to_rlds-"
            "austin_sirius_dataset_converted_externally_to_rlds-"
            "bc_z-"
            "berkeley_autolab_ur5-"
            "berkeley_cable_routing-"
            "berkeley_fanuc_manipulation-"
            "bridge_dataset-"
            "cmu_stretch-"
            "dlr_edan_shared_control_converted_externally_to_rlds-"
            "fractal20220817_data-"
            "furniture_bench_dataset_converted_externally_to_rlds-"
            "iamlab_cmu_pickup_insert_converted_externally_to_rlds-"
            "jaco_play-"
            "nyu_door_opening_surprising_effectiveness-"
            "nyu_franka_play_dataset_converted_externally_to_rlds-"
            "roboturk-"
            "stanford_hydra_dataset_converted_externally_to_rlds-"
            "taco_play-"
            "toto-"
            "ucsd_kitchen_dataset_converted_externally_to_rlds-"
            "utaustin_mutex-"
            "viola-"
            "deo_in_pouch_dataset_first10_128x128"
            # "easy_pick_dataset_n5_1"
        ),
        "data_dir": "/mnt/nfs/home/alaakh/store/oxe/mpt_dataset/train",
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:austin_buds_dataset_transform",
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        # "state_obs_keys": ["state", None],
        "state_obs_keys": ["state"],

        # "name": "easy_pick_dataset_n5_1",
        # "data_dir": "/mnt/nfs/home/alaakh/store/oxe/mpt_dataset/train",
        # "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:custom_dataset_transform", 
        # "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        # "state_obs_keys": ["state"],

        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        "absolute_action_mask": [False, False, False, False, False, False, True],
        "action_normalization_mask": [True, True, True, True, True, True, False],    
    }

    FINETUNING_VAL_KWARGS = {
        # "name": "bridge_dataset",
        # "data_dir": "./tests/debug_dataset",
        # "data_dir": "/mnt/xfs/home/alaakh/store/oxe/traj_data",
        # "data_dir": "/mnt/xfs/home/alaakh/store/oxe/train_val_splits/train",
        # "data_dir": "/mnt/xfs/home/alaakh/store/oxe/train_val_splits/val",
        # "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/val",

        # "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:bridge_dataset_transform",
        # "image_obs_keys": {"primary": "image_0", "wrist": None},
        # "state_obs_keys": ["state", None],
        # "language_key": "language_instruction",
        # "action_proprio_normalization_type": "normal",
        # "absolute_action_mask": [False, False, False, False, False, False, True],

        # "name": "easy_pick_dataset_n10",
        # "data_dir": "/mnt/xfs/home/alaakh/store/oxe/real_data",

        # easy pick last
        # "name": "easy_pick_dataset_n5_2",
        # "data_dir": "/mnt/xfs/home/alaakh/store/oxe/real_data_splits/easy_pick",

        # deo
        "name": "deo_in_pouch_dataset_last10_128x128",
        "data_dir": "/mnt/xfs/home/alaakh/store/oxe/real_data_splits/deo_in_pouch",

        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:custom_dataset_transform", 
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        "state_obs_keys": ["state"],
        # "state_obs_keys": ["state", None],
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        "absolute_action_mask": [False, False, False, False, False, False, True],
        "action_normalization_mask": [True, True, True, True, True, True, False],
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

    # max_steps = FieldReference(50_000)
    # max_steps = FieldReference(10_000)
    # max_steps = FieldReference(5_000)
    # max_steps = FieldReference(3_800)
    # max_steps = FieldReference(2_000)
    max_steps = FieldReference(5_000)
    # max_steps = FieldReference(5)
    # window_size = FieldReference(default=1) # old
    window_size = FieldReference(default=1) # new

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        job_id=placeholder(int),
        folder_name=placeholder(str),

        # num_workers=8,
        # num_workers=32,
        # num_workers=48,
        num_workers=64,

        batch_size=256,
        mini_batch_size=128,
        val_batch_size=32,
        mini_val_batch_size=32,
        # batch_size=128,
        shuffle_buffer_size=10000,

        num_steps=max_steps,
        bob_steps=100,
        # bob_steps=4,
        # bob_steps=2,

        # candidate_size=1,
        candidate_size=0.125,

        # num_steps=30,
        # bob_steps=2,
        # candidate_size=0.01,

        log_interval=100,
        # eval_interval=5000,
        # save_interval=5000,
        eval_interval=int(0.05 * max_steps.get()),
        save_interval=int(0.05 * max_steps.get()),
        # save_dir=placeholder(str),
        save_dir='/mnt/xfs/home/alaakh/src/octo_dir/octo/exps/debug',
        seed=42,
        wandb=dict(
            project="octo_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=FINETUNING_KWARGS,
        val_dataset_kwargs=FINETUNING_VAL_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                # warmup_steps=2000,
                warmup_steps=int(0.05 * max_steps.get()),
                # warmup_steps=int(0.1 * max_steps.get()),
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

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        future_action_window_size=3,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
        # If the default data loading speed is too slow, try these:
        # num_parallel_calls=16,  # for less CPU-intensive ops
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
            # "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
            "primary": (128, 128),  # workspace (3rd person) camera is at 256x256
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

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
