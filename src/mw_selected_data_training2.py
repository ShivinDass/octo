from torch.utils.data import Dataset, DataLoader
import glob, os
import h5py
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax_lm.metaworld.metaworld_model_jax import create_mw_policy
from jax_lm.metaworld.eval_metaworld import eval_metaworld_sim

def load_dm_mask(path, topk):
    iter_num = len(os.listdir(path))
    all_dms = []
    count_dms = 0

    n_trajs = 8245
    for iter_id in range(iter_num):
        iter_path = os.path.join(path, f'iter_{iter_id}')
        
        try:
            # filtered_dw = np.load(os.path.join(iter_path, 'data_weights.npy'))[:1000000]
            filtered_dm = np.load(os.path.join(iter_path, 'datamodels.npy'))[1000000:1000000+n_trajs]
            all_dms.append(filtered_dm)
            count_dms += 1
        except:
            pass
    all_dms = np.array(all_dms)
    avg_dm = np.mean(all_dms, axis=0, where=all_dms!=0)
    avg_dm = np.nan_to_num(avg_dm)

    print(count_dms, np.sum(avg_dm!=0), n_trajs)
    # assert n_trajs == np.sum(avg_dm!=0)

    # Option 1
    threshold = np.sort(avg_dm, axis=0)[int(topk*n_trajs)]
    selected_dm = avg_dm <= threshold
    print(threshold, np.sum(selected_dm))

    return selected_dm

class MetaworldDataset(Dataset):

    def __init__(self, data_folder, path_masks=None):
        super().__init__()

        self.hdf5_files = sorted(glob.glob(os.path.join(data_folder, "**", "*.h5"), recursive=True))
        if path_masks is not None:
            filter_files = []
            for i, file in enumerate(self.hdf5_files):
                if path_masks[i]:
                    filter_files.append(file)
            self.hdf5_files = filter_files

        print(len(self.hdf5_files), "files found")
        
        self.actions = []
        self.observations = []
        for file in tqdm.tqdm(self.hdf5_files):
            with h5py.File(file, 'r') as f:
                self.actions.extend(f['actions'][:])
                self.observations.extend(f['observations'][:])

        # Convert to numpy arrays
        self.actions = np.array(self.actions)
        self.observations = np.array(self.observations)
        
    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        action = self.actions[idx]
        observation = self.observations[idx]

        return action, observation

def train_policy(seed):
    #load pre-trained model
    dm_path = "/home/shivin/libero_experiments/experiments/metaworld/pick-place-wall_dm"
    topk = 0.1
    selected_mask = load_dm_mask(dm_path, topk=topk)

    data_folder = "/home/shivin/libero_experiments/data/metaworld/prior/"
    dataset = MetaworldDataset(data_folder, path_masks=selected_mask)
    print(dataset.actions.shape, dataset.observations.shape)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

    policy = create_mw_policy(seed=seed, obs_dim=39)
    # policy, state = create_train_state(seed=0, lr=3e-4)

    eval_results = []
    for epoch in tqdm.tqdm(range(1000)):
        total_loss = 0
        for batch in dataloader:
            
            # convert from torch tensor to jax
            actions = jnp.array(batch[0].numpy())
            observations = jnp.array(batch[1].numpy())

            policy, loss = policy.update(observations, actions)
            # state, loss = train_step(policy.state, observations, actions)
            total_loss += loss.mean()

        if (epoch+1) % 200 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
            success_rate = eval_metaworld_sim(policy)['prob']
            print(f"Success Rate: {success_rate}")
            eval_results.append(success_rate)

    return eval_results

if __name__ == "__main__":
    results = []
    for i in range(3):
        results.append(train_policy(seed=i))

    print('mean:', np.mean(results, axis=0))
    print('std:', np.std(results, axis=0))
    