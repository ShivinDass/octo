import jax
import jax.numpy as jnp
from domains.cifar_model import make_net
from functools import partial
from metagradients.optimizers.sgd import SGDTrainState
from functools import cache
from domains.cifar_ds import loaders_and_order_for_seed
from metagradients.dataloading import naive_batch_maker
from domains.calculate_lds import calculate_lds

MODEL_SEED = 0
DATA_SEED = 0
TEST_SAMPLE = 0
DROP_FRAC = 0.01
LDS_SEED = 0

EPOCHS = 13
BATCH_SIZE = 1000
LR = 10.0
MOMENTUM = 0.85
WEIGHT_DECAY = 0.0153
BIAS_SCALER = 64.0
WHITEN_BIAS_EPOCHS = 3
TTA_LEVEL = 2

def get_lr(step, base_lr, total_train_steps):
    frac = step / total_train_steps
    (x0, y0), (x1, y1), (x2, y2) = (0.0, 0.2), (0.23, 1.0), (1.0, 0.07)
    return jnp.minimum(base_lr * (y0 + (y1 - y0) / (x1 - x0) * (frac - x0)),
                        base_lr * (y1 + (y2 - y1) / (x2 - x1) * (frac - x1)))

@cache
def model_maker(seed, train_loader):
    images_init = jnp.zeros((5000, 32, 32, 3), jnp.float32)
    for i in range(5):
        _, (images, _, _) = train_loader(i)
        images_init = images_init.at[i*1000:(i+1)*1000].set(images)
    model, model_state = make_net(seed, images_init)
    return jax.tree_util.Partial(model.apply), model_state

# missing only: initial_params=params, train_its=train_its, 
def make_loaders_and_data_weights(data_seed):
    cifar_loaders_for_seed = partial(loaders_and_order_for_seed, bs=BATCH_SIZE, epochs=EPOCHS)
    loaders = cifar_loaders_for_seed(data_seed)
    (train_ba_fn, train_its), (val_ba_fn, val_its) = loaders
    assert train_its > 0

    # make dataweights
    num_datapoints = 0
    for it in range(train_its):
        ixs = train_ba_fn(it)[0]
        num_datapoints = int(max(num_datapoints, ixs.max() + 1))

    data_weights = jax.numpy.ones((num_datapoints,), dtype=jnp.float32)
 
    train_batcher = partial(naive_batch_maker, get_batch=train_ba_fn,
                            minibs=BATCH_SIZE, num_batches=train_its)
    val_batcher = partial(naive_batch_maker, get_batch=val_ba_fn,
                          minibs=BATCH_SIZE, num_batches=val_its)
    return train_batcher, val_batcher, data_weights, train_its, val_its

def make_cifar_optimizer(params, train_its):
    return SGDTrainState.create(params=params, 
                                lr=jax.tree_util.Partial(get_lr, base_lr=LR, total_train_steps=train_its), 
                                momentum=MOMENTUM, 
                                weight_decay=WEIGHT_DECAY, 
                                bias_scaler=BIAS_SCALER)

def main():
    ret = calculate_lds(MODEL_SEED, DATA_SEED, TEST_SAMPLE, BATCH_SIZE, DROP_FRAC,
                        LDS_SEED, make_loaders_and_data_weights, model_maker,
                        make_cifar_optimizer)
    print(ret)

if __name__ == '__main__':
    main()

