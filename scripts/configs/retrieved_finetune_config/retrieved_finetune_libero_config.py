from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
import os

# task_name="kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_h8_prechunk"
# task_name="kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_h8_prechunk"
def get_config(config_string="br,0.1"):
    method_name, percent, task_name = config_string.split(',')
    assert method_name in ['br', 'flow', 'action', 'language']

    mode, task = "full", "language_conditioned"
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    data_dir = '/mnt/hdd2/baselines/'    

    target_train_path = os.path.join(data_dir, f'target_data_chunk8/{task_name}/train', 'out.tfrecord')
    prior_train_path = os.path.join(data_dir, f'/mnt/hdd2/baselines/retrieved_data_chunk8/{task_name}/{task_name}_th{percent}/{method_name}/out.tfrecord')

    FINETUNING_KWARGS = {
        "data_paths": [[prior_train_path], [target_train_path]],
        "sample_weights": None, 
        "load_keys": 'all',
        "dataset_statistics_path": "/home/shivin/tensorflow_datasets/libero90/0.1.0/dataset_statistics_9abb65a9c7829f52c81741919ae39f05baf55b6a5aab3f0ddd897947d3b283e5.json",
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

    max_steps = FieldReference(10_000)
    window_size = FieldReference(default=1)

    config = dict(
        use_proprio=True,
        action_chunks=4,
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=128,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=int(max_steps.get()//5),
        save_interval=int(max_steps.get()),
        save_ckpts=[10000],
        save_dir='/mnt/hdd2/libero/experiments/',
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
