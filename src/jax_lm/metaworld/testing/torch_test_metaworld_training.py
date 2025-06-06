from torch.utils.data import Dataset, DataLoader
import glob, os
import h5py
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from src.metaworld.testing.metaworld_model_torch import create_mw_policy
from eval_metaworld import eval_metaworld_sim
import torch

class MetaworldDataset(Dataset):

    def __init__(self, data_folder):
        super().__init__()

        filter_func = lambda x: 'pick-place-wall' in x

        self.hdf5_files = glob.glob(os.path.join(data_folder, "**", "*.h5"), recursive=True)
        self.hdf5_files = list(filter(filter_func, self.hdf5_files))

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

if __name__ == "__main__":
    #load pre-trained model

    data_folder = "/home/shivin/libero_experiments/data/metaworld/prior/demonstration"
    dataset = MetaworldDataset(data_folder)
    print(dataset.actions.shape, dataset.observations.shape)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

    policy = create_mw_policy(seed=0)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        betas=(0.9, 0.999),
    )

    for epoch in range(1000):
        total_loss = 0
        for batch in dataloader:
            
            actions = batch[0].float().to('cuda')
            observations = batch[1].float().to('cuda')

            optimizer.zero_grad(set_to_none=True)
            out = policy(observations)
            loss = policy.loss(out, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch+1) % 250 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
            print(eval_metaworld_sim(policy))