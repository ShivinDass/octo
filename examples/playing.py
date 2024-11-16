import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from octo.utils.visualization_lib import write_video

import cv2
import numpy as np
from tqdm import tqdm

# ============= Data =============
print('\n' + '='*10 +'> Data Details <' + '='*10)
builder = tfds.builder_from_directory(builder_dir='gs://gresearch/robotics/bridge/0.1.0/')
ds = builder.as_dataset(split='train[:1]')

episode = next(iter(ds))
steps = list(episode['steps'])
images = [cv2.resize(np.array(step['observation']['image']), (256, 256)) for step in steps]

# write_video(np.array(images), '../visuals/playing.mp4', fps=10)
language_instruction = steps[0]['observation']['natural_language_instruction'].numpy().decode()
print(language_instruction)
print(len(steps))
print(steps[0].keys())
print(type(steps[0]['action']['world_vector']))

# ============= Model =============
print('\n' + '='*10 +'> Model Details <' + '='*10)
from octo.model.octo_model import OctoModel

model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
print(model.get_pretty_spec())


# ============= Inference =============
print('\n' + '='*10 +'> Inference Details <' + '='*10)
task = model.create_tasks(texts=[language_instruction])
WINDOW_SIZE = 2
N_RNG=5

pred_actions_list = [[] for _ in range(N_RNG)]
true_actions = []
for step in tqdm(range(0, len(images)-WINDOW_SIZE+1)):
    input_images = np.stack(images[step:step+WINDOW_SIZE])[None]
    observation = {
        'image_primary': input_images,
        'timestep_pad_mask': np.full((1, input_images.shape[1]), True, dtype=bool)
    }

    for i in range(N_RNG):
        actions = model.sample_actions(
            observation, 
            task, 
            unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
            rng=jax.random.PRNGKey(i))
        action = actions[0]

        pred_actions_list[i].append(action)

    # print(steps[step+WINDOW_SIZE-1]['action']['world_vector'].shape, steps[step+WINDOW_SIZE-1]['action']['rotation_delta'].shape, steps[step+WINDOW_SIZE-1]['action']['open_gripper'].shape)
    true_actions.append(np.concatenate(
        (
            steps[step+WINDOW_SIZE-1]['action']['world_vector'],
            steps[step+WINDOW_SIZE-1]['action']['rotation_delta'],
            np.array([steps[step+WINDOW_SIZE-1]['action']['open_gripper']]).astype(np.float32)
        ), axis=-1
    ))

# ============= Plotting =============
print('\n' + '='*10 +'> Plotting Details <' + '='*10)

import matplotlib.pyplot as plt

ACTION_DIM_LABELS = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'grasp']

# build image strip to show above actions
img_strip = np.concatenate(np.array(images[::3]), axis=1)

# set up plt figure
figure_layout = [
    ['image'] * len(ACTION_DIM_LABELS),
    ACTION_DIM_LABELS
]
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplot_mosaic(figure_layout)
fig.set_size_inches([45, 10])

# plot actions
pred_actions = np.array(pred_actions_list).squeeze()
true_actions = np.array(true_actions).squeeze()
for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
  # actions have batch, horizon, dim, in this example we just take the first action for simplicity
#   axs[action_label].plot(pred_actions[:, 0, action_dim], label='predicted action')
  for i in range(1, N_RNG):
    axs[action_label].plot(pred_actions[i, :, 0, action_dim], label=f'predicted action {i}')
  axs[action_label].plot(true_actions[:, action_dim], label='ground truth')
  axs[action_label].set_title(action_label)
  axs[action_label].set_xlabel('Time in one episode')

axs['image'].imshow(img_strip)
axs['image'].set_xlabel('Time in one episode (subsampled)')
plt.legend()
plt.savefig('../visuals/playing.png')