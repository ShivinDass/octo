configs = [
    {
        "name": "utaustin_mutex",
        "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:utaustin_mutex_dataset_transform",
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        "state_obs_keys": ["state"],
    },

    {
        "name": "austin_sailor_dataset_converted_externally_to_rlds",
        "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:austin_sailor_dataset_transform",
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        "state_obs_keys": ["state"],
    },

    {
        "name": "austin_sirius_dataset_converted_externally_to_rlds",
        "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:austin_sirius_dataset_transform",
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        "state_obs_keys": ["state"],      
    },

    {
        "name": "austin_buds_dataset_converted_externally_to_rlds",
        "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:austin_buds_dataset_transform",
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        "state_obs_keys": ["state"],
    },

    {
        "name": "jaco_play",
        "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:jaco_play_dataset_transform",
        "image_obs_keys": {"primary": "image", "wrist": "image_wrist"},
        "state_obs_keys": ["state_eef", None, "state_gripper"],
    },

    {
        "name": "viola",
        "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:viola_dataset_transform",
        "image_obs_keys": {"primary": "agentview_rgb", "wrist": "eye_in_hand_rgb"},
        "state_obs_keys": ["joint_states", "gripper_states"],
    },

    {
        "name": "taco_play",
        "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:taco_play_dataset_transform",
        "image_obs_keys": {"primary": "rgb_static", "wrist": "rgb_gripper"},
        "state_obs_keys": ["state_eef", None, "state_gripper"],
    },

    {
        "name": "fractal20220817_data",
        "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:rt1_dataset_transform",
        "image_obs_keys": {"primary": "image", "wrist": None},
        "state_obs_keys": ["base_pose_tool_reached", "gripper_closed"],
    },

    {
        "name": "stanford_hydra_dataset_converted_externally_to_rlds",
        "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:stanford_hydra_dataset_transform",
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
    },

    
    {
        "name": "nyu_franka_play_dataset_converted_externally_to_rlds",
        "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:nyu_franka_play_dataset_transform",
        "image_obs_keys": {"primary": "image", "secondary": "image_additional_view", "wrist": None},
        "state_obs_keys": ["eef_state", None, None],
    },

    {
        "name": "ucsd_kitchen_dataset_converted_externally_to_rlds",
        "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:ucsd_kitchen_dataset_transform",
        "image_obs_keys": {"primary": "image", "wrist": None},
        "state_obs_keys": ["joint_state", None],
    },

    {
        "name": "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:iamlab_pick_insert_dataset_transform",
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        "state_obs_keys": ["joint_state", "gripper_state"],
    },

    {
        "name": "furniture_bench_dataset_converted_externally_to_rlds",
        "data_dir": "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train",
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:furniture_bench_dataset_transform",
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        "state_obs_keys": ["state"],
    },

]