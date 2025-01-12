import torch as ch
import os
import torchvision
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from math import ceil
import augmax
from augmax.base import PyTree as AugPyTree

# random shift 2px
# random flip lr
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465]) * 255
CIFAR_STD = np.array((0.2470, 0.2435, 0.2616)) * 255
IMAGE_UPPER_BD = (jnp.ones((32, 32, 3)) * 255 - CIFAR_MEAN) / CIFAR_STD
IMAGE_LOWER_BD = (jnp.zeros((32, 32, 3)) * 255 - CIFAR_MEAN) / CIFAR_STD

class CutOut(augmax.imagelevel.ImageLevelTransformation):
    def __init__(self, cutout_size=(8, 8), p=0.5, input_types=[augmax.InputType.IMAGE]):
        assert input_types == [augmax.InputType.IMAGE]
        super().__init__(input_types)
        self.cutout_size = cutout_size
        self.probability = p

    def apply(self, rng, inputs: AugPyTree, input_types: AugPyTree, invert=False):
        assert input_types == augmax.InputType.IMAGE

        key1, key2 = jax.random.split(rng)
        do_apply = jax.random.bernoulli(key1, self.probability)
        def transform_single(raw_image):
            H, W, C = raw_image.shape
            cx, cy = self.cutout_size
            cutout = jnp.zeros((cx, cy, C))
            x, y = jax.random.randint(key2, (2,), (H - cx), (W - cy))
            image = jax.lax.dynamic_update_slice(raw_image, cutout, (x, y, 0))
            current = jnp.where(do_apply, image, raw_image)
            return current

        return jax.tree_map(transform_single, inputs)

AUGMAX_AUGS = augmax.Chain(
    augmax.HorizontalFlip(0.5),
    augmax.RandomCrop(width=32, height=32),
)

def ds_to_jax(ds):
    X = np.array(ds.data).astype(np.float32) #  np ndarray uint8
    X = (X - CIFAR_MEAN) / CIFAR_STD

    idxs = np.arange(X.shape[0])
    Y = np.array(ds.targets).astype(np.int32) # list of ints

    # now convert to jax and put on gpu
    gpu_dev = jax.devices('gpu')[0]
    X = jax.device_put(jnp.array(X), gpu_dev)
    Y = jax.device_put(jnp.array(Y), gpu_dev)
    return idxs, (X, Y)

def augment_X(X, inds, seed):
    rng = jax.random.PRNGKey(seed)
    # sub_rngs = jax.random.split(rng, X.shape[0]) [TODO: edited here]
    sub_rngs = jax.random.split(rng, 50_000)[inds]
    # pad to 36x36
    X = jnp.pad(X, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='reflect')
    vmapped_transform = jax.vmap(AUGMAX_AUGS)
    aug_X = vmapped_transform(sub_rngs, X)
    return aug_X

ds = torchvision.datasets.CIFAR10(root='/tmp/data', train=True, download=True)
ds_test = torchvision.datasets.CIFAR10(root='/tmp/data', train=False, download=True)

ixs, (X, Y) = ds_to_jax(ds)
Y = jax.nn.one_hot(Y, 10) * 10 # Roughly one-hot after softmax

# First, randomly shuffle the training data
rand_perm = np.random.RandomState(0).permutation(len(X))
X = X[rand_perm]
Y = Y[rand_perm]
ixs = ixs[rand_perm]

ixs_test, (X_test, Y_test) = ds_to_jax(ds_test)
Y_test = jax.nn.one_hot(Y_test, 10) * 10 # Roughly one-hot after softmax

# Make a random permutation with seed 0
rand_perm = np.random.RandomState(0).permutation(len(X_test))
X_test = X_test[rand_perm]
Y_test = Y_test[rand_perm]
ixs_test = ixs_test[rand_perm]

n_train = len(X)
n_test = len(X_test)

def loaders_and_order_for_seed(seed, bs, epochs, shuffle_train=True, use_nchw=False):
    assert n_train % bs == 0
    assert n_test % bs == 0

    train_indices = []
    for epoch_ii in range(epochs):
        if shuffle_train:
            shuffle_key = jax.random.PRNGKey(seed + epoch_ii)
            this_epoch_indices = jax.random.permutation(shuffle_key, ixs)
        else:
            this_epoch_indices = ixs

        train_indices.append(this_epoch_indices)

    train_indices = jnp.concatenate(train_indices)

    def train_batcher(i):
        s, e = i * bs, (i + 1) * bs
        slc = slice(s, e)
        bseed = seed + i
        batch_ixs = train_indices[slc]
        sel_X, sel_Y = X[batch_ixs], Y[batch_ixs]

        augmenter = jax.tree_util.Partial(augment_X, seed=bseed)
        return batch_ixs, (sel_X, sel_Y, augmenter)

    def test_batcher(i):
        s, e = i * bs, (i + 1) * bs
        slc = slice(s, e)
        X_batch = X_test[slc]
        Y_batch = Y_test[slc]
        return np.array(ixs_test[slc]), (X_batch, Y_batch, None)

    train_batcher.dataset_size = len(X)
    test_batcher.dataset_size = len(X_test)
    return (train_batcher, train_batcher.dataset_size * epochs // bs), \
           (test_batcher, test_batcher.dataset_size // bs)

if __name__ == '__main__':
    tf, vf = loaders_and_order_for_seed(0, 10, 2, True, True)
    x = tf(3)[1][0]
    print(x[0])
    # vf(4)