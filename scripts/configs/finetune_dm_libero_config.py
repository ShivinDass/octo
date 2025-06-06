from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
import copy

def get_config(config_string="full,language_conditioned"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # Fill this in for your own dataset!

    # There should be two image keys
    # first image key should be the third-person view (None if not used)
    # and second image key should be the wrist view (None if not used)
    # data_path = "/home/shivin/tensorflow_datasets"
    data_path = "/mnt/hdd2/libero/mpt_dataset"
    FINETUNING_TRAIN_KWARGS = {
        # "name": "libero90_128x128",
        # "name": "libero90_horizon30_128x128",
        "name": placeholder(str), #"libero90_horizon15_128x128",
        "data_dir": f"{data_path}",
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        "state_obs_keys": ["state"],
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        # All actions are relative deltas, except for the last one (gripper) which is absolute
        # Specifying this is only necessary if you want to predict > 1 step into the future
        "absolute_action_mask": [False, False, False, False, False, False, True],
        "action_normalization_mask": [True, True, True, True, True, True, False],
        # standardize_fn is dynamically loaded from a file
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:custom_dataset_transform",
    }

    FINETUNING_VAL_KWARGS = copy.deepcopy(FINETUNING_TRAIN_KWARGS)
    FINETUNING_VAL_KWARGS["name"] = placeholder(str) # target_task_name
    FINETUNING_VAL_KWARGS["data_dir"] = f"{data_path}/libero_val_first5_128x128/"

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

    max_steps = FieldReference(2000)
    window_size = FieldReference(default=1)

    config = dict(
        use_proprio=True,
        action_chunks=8,
        loss_type="mse", #"l1",

        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),

        ## added
        job_id=placeholder(int),
        folder_name=placeholder(str),
        num_workers=32, # fewer workers (half of # of cpus typically)

        ## added
        batch_size=512,
        mini_batch_size=512,
        val_batch_size=512,
        mini_val_batch_size=512,

        shuffle_buffer_size=10000,
        num_steps=max_steps,

        # added
        bob_steps=100,
        candidate_size=1.0,#0.2,

        log_interval=100,
        eval_interval=int(max_steps.get()//5),
        save_interval=int(max_steps.get()//5),
        save_dir=f'/mnt/hdd2/dm_experiments/',
        seed=42,
        wandb=dict(
            project="octo_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=FINETUNING_TRAIN_KWARGS,
        val_dataset_kwargs=FINETUNING_VAL_KWARGS,

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
            trajs_for_metrics=5,
            trajs_for_viz=2,
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
        future_action_window_size=7,
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
            "primary": (128, 128),  # workspace (3rd person) camera is at 256x256
            "wrist": (128, 128),  # wrist camera is at 128x128
        },
        image_augment_kwargs=dict(
            # primary=workspace_augment_kwargs,
            # wrist=wrist_augment_kwargs,
        ),
    )
    # If the default data loading speed is too slow, try these:
    config["frame_transform_threads"] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
