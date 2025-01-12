#############################################
#            Setup/Hyperparameters          #
#############################################
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable, Tuple
from jax import lax
import numpy as np
from flax.linen.pooling import pool

TEMP = 2.0
def lse_pool(inputs, window_shape, strides=None, padding='VALID', count_include_pad=True):
    inputs = jnp.exp(inputs * TEMP)
    y = pool(inputs, 0.0, lax.add, window_shape, strides, padding)
    if count_include_pad:
        y = y / np.prod(window_shape)
    else:
        div_shape = inputs.shape[:-1] + (1,)
        if len(div_shape) - 2 == len(window_shape):
            div_shape = (1,) + div_shape[1:]
        y = y / pool(jnp.ones(div_shape), 0.0, lax.add, window_shape, strides, padding)
    return jnp.log(y) / TEMP

# Hyp dictionary as provided
my_pool = lse_pool
hyp = {
    'widths': {
        'block1': 128,
        'block2': 384,
        'block3': 384,
    },
    'batchnorm_momentum': 0.4,
    'scaling_factor': 1/9.,
}

class Flatten(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Input is (N, H, W, C) due to NHWC format.
        # Flatten to (N, H*W*C)
        return x.reshape((x.shape[0], -1))

class BatchNorm(nn.BatchNorm):
    def __call__(self, x, training=True):
        return super().__call__(x, use_running_average=not training)

class ConvGroup(nn.Module):
    channels_out: int
    batchnorm_momentum: float

    @nn.compact
    def __call__(self, x, training=True):
        # x is NHWC
        x = nn.Conv(self.channels_out, 
                    kernel_size=(3, 3), 
                    strides=1, 
                    padding=1, 
                    use_bias=False)(x)
        x = my_pool(x, window_shape=(2, 2), strides=(2, 2)) 
        x = BatchNorm(momentum=self.batchnorm_momentum, use_scale=False, use_bias=True)(x, training)
        x = nn.gelu(x)
        x = nn.Conv(self.channels_out, 
                    kernel_size=(3, 3), 
                    strides=1, 
                    padding=1, 
                    use_bias=False)(x)
        x = BatchNorm(momentum=self.batchnorm_momentum, use_scale=False, use_bias=True)(x, training)
        x = nn.gelu(x)
        return x

class Net(nn.Module):
    hyp: dict

    @nn.compact
    def __call__(self, x, training=True):
        # x is assumed to come from the dataloader in NHWC format: (N, H, W, C)
        widths = self.hyp['widths']
        batchnorm_momentum = self.hyp['batchnorm_momentum']
        scaling_factor = self.hyp['scaling_factor']

        whiten_kernel_size = 2
        whiten_width = 2 * 3 * (whiten_kernel_size**2)

        x = nn.Conv(whiten_width, 
                    kernel_size=(whiten_kernel_size, whiten_kernel_size), 
                    padding=0, 
                    use_bias=True)(x)
        x = nn.gelu(x)
        x = ConvGroup(widths['block1'], batchnorm_momentum)(x, training)
        x = ConvGroup(widths['block2'], batchnorm_momentum)(x, training)
        x = ConvGroup(widths['block3'], batchnorm_momentum)(x, training)
        x = my_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = Flatten()(x)  # Now a fully flattened vector per image.
        x = nn.Dense(10, use_bias=False)(x)
        return x * scaling_factor

class TTANet(Net):
    def __call__(self, x, tta_level=0, training=False):
        """
        Infer function with test-time augmentation (TTA).
        images: (N, H, W, C) - NHWC format
        Returns logits (N, num_classes).
        """
        def infer_basic(inputs):
            # When not training, we use batchnorm running averages, so no need for mutable here.
            return super(TTANet, self).__call__(inputs, training=training)

        def infer_mirror(inputs):
            # Flip along width dimension (axis=2 for NHWC)
            flipped = jnp.flip(inputs, axis=2)
            return 0.5 * infer_basic(inputs) + 0.5 * infer_basic(flipped)

        def infer_mirror_translate(inputs):
            logits = infer_mirror(inputs)
            # Pad height and width dimensions (axes 1 and 2 for NHWC)
            padded = jnp.pad(inputs, ((0,0),(1,1),(1,1),(0,0)), mode='reflect')
            inputs_translate_list = [
                padded[:, 0:32, 0:32, :],
                padded[:, 2:34, 2:34, :]
            ]
            logits_translate_list = [infer_mirror(x) for x in inputs_translate_list]
            logits_translate = jnp.mean(jnp.stack(logits_translate_list), axis=0)
            return 0.5 * logits + 0.5 * logits_translate

        infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
        return infer_fn(x)

#############################################
#       Whitening Conv Initialization       #
#############################################

def get_patches(x, patch_shape):
    N, H, W, C = x.shape
    h, w = patch_shape
    out_h = H - h + 1
    out_w = W - w + 1

    patches_list = []
    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, i:i+h, j:j+w, :]
            patch = jnp.transpose(patch, (0, 3, 1, 2))
            patches_list.append(patch)

    # (out_h*out_w, N, C, h, w)
    patches = jnp.stack(patches_list, axis=0)
    # Rearrange to (N*out_h*out_w, C, h, w)
    patches = patches.transpose(1,0,2,3,4).reshape(N*out_h*out_w, C, h, w)
    return patches.astype(jnp.float32)

def get_whitening_parameters(patches):
    n, c, h, w = patches.shape
    patches_flat = patches.reshape(n, -1)  # (n, c*h*w)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n  # (c*h*w, c*h*w)

    eigenvalues, eigenvectors = jnp.linalg.eigh(est_patch_covariance)
    eigenvalues = jnp.flip(eigenvalues, axis=0)
    eigenvectors = eigenvectors.T.reshape(c*h*w, c, h, w)
    eigenvectors = jnp.flip(eigenvectors, axis=0)

    return eigenvalues.reshape(-1, 1, 1, 1), eigenvectors

def init_whitening_conv(weight, train_set, eps=5e-4):
    kh, kw, _, _ = weight.shape
    patches = get_patches(train_set, patch_shape=(kh, kw))
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / jnp.sqrt(eigenvalues + eps)
    initialized_weight = jnp.concatenate([eigenvectors_scaled, -eigenvectors_scaled], axis=0)
    initialized_weight = jnp.transpose(initialized_weight, (2, 3, 1, 0))
    return initialized_weight


#############################################
#               Model Constructor           #
#############################################

def make_net(seed, images_init):
    rng = jax.random.PRNGKey(seed)
    model = TTANet(hyp=hyp)
    dummy_input = jnp.ones((1,32,32,3), jnp.float32)

    # model.init returns a dictionary with params and batch_stats (if any).
    variables = model.init(rng, dummy_input, training=True)
    params = variables['params']
    batch_stats = variables['batch_stats']

    whitened_weights = init_whitening_conv(params['Conv_0']['kernel'], images_init)
    # Update params with whitened weights
    def initializer_fn(path, x):
        if 'Conv_0' in str(path[0]) and 'kernel' in str(path[-1]):
            return whitened_weights
        elif 'kernel' in str(path[-1]) and 'conv' in str(path[0]).lower():
            a, b, c, d = x.shape
            new_d = min(c, d)
            w = jnp.zeros((a, b, new_d, new_d))
            for i in range(new_d):
                w = w.at[a // 2, b // 2, i, i].set(1.0)
            return x.at[:, :, :new_d, :new_d].set(w)
        return x
    params = jax.tree_util.tree_map_with_path(initializer_fn, params)

    return model, {'params': params, 'batch_stats': batch_stats}



