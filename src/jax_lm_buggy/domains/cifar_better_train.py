import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from math import ceil
from domains.cifar_ds import loaders_and_order_for_seed
from domains.cifar_model import make_net, hyp
from functools import partial
from metagradients.optimizers.sgd import SGDTrainState
from optax import softmax_cross_entropy

jax.config.update('jax_enable_x64', False)

#############################################
#      Hyperparameters          #
#############################################
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

# Disable float64

def main(seed=0):
    kilostep_scale = 1024 * (1 + 1 / (1 - MOMENTUM))
    lr = LR / kilostep_scale
    wd = WEIGHT_DECAY * BATCH_SIZE / kilostep_scale

    ((train_fn, train_num_its), (val_fn, val_num_its)) = loaders_and_order_for_seed(seed, BATCH_SIZE, EPOCHS)
    total_train_steps = train_num_its

    # Initialize whitening conv
    images_init = jnp.zeros((5000, 32, 32, 3), jnp.float32)
    for i in range(5):
        _, (images, _, _) = train_fn(i)
        images_init = images_init.at[i*1000:(i+1)*1000].set(images)
    model, model_state = make_net(seed, images_init)
    all_state = SGDTrainState.create(**model_state,
                                     lr=jax.tree_util.Partial(get_lr, base_lr=lr, total_train_steps=total_train_steps), 
                                     momentum=MOMENTUM, 
                                     weight_decay=wd, 
                                     bias_scaler=BIAS_SCALER, 
                                     max_lr=lr)
    
    def loss_fn(vars, inputs, labels):
        labels = jax.nn.softmax(labels)
        logits, new_vars = model.apply(vars, inputs, training=True, mutable=['batch_stats'])
        new_batch_stats = new_vars['batch_stats']
        per_example_loss = softmax_cross_entropy(logits, labels)
        return jnp.sum(per_example_loss), (logits, new_batch_stats)

    @jax.jit
    def eval_step(vars, images, labels):
        logits = model.apply(vars, images, tta_level=TTA_LEVEL, training=False)
        preds = jnp.argmax(logits, axis=1)
        correct = jnp.sum(preds == jnp.argmax(labels, axis=1))
        total = labels.shape[0]
        return correct, total

    def evaluate(_all_state, val_fn, val_num_its):
        correct = 0
        total = 0
        for i in range(val_num_its):
            idx, (images, labels, _) = val_fn(i)
            vars = {'params': _all_state.params, 'batch_stats': _all_state.batch_stats}
            correct, total = eval_step(vars, images, labels)
            correct += correct
            total += total
        return (correct / total).item()

    @jax.jit
    def train_step(_all_state, inputs, labels, step):
        gradder = jax.value_and_grad(loss_fn, has_aux=True)
        vars = {'params': _all_state.params, 'batch_stats': _all_state.batch_stats}
        (loss, (logits, new_batch_stats)), grads = gradder(vars, inputs, labels)
        new_all_state = _all_state.apply_grads(grads['params'], bs_updates=new_batch_stats)
        return new_all_state, loss, logits

    current_steps = 0
    steps_per_epoch = train_num_its / ceil(EPOCHS)  # approximate steps per epoch

    for epoch_i in range(ceil(EPOCHS)):
        # One epoch of training
        for i in range(int(steps_per_epoch)):
            idx, (images, labels, augment_X) = train_fn(current_steps)
            images = augment_X(images, inds=idx)
            all_state, loss, logits = train_step(all_state, images, labels, current_steps)
            current_steps += 1

        # Evaluation after the epoch
        train_acc = jnp.mean((jnp.argmax(logits, axis=1) == jnp.argmax(labels, axis=1)).astype(jnp.float32))
        train_loss = loss
        val_acc = evaluate(all_state, val_fn, val_num_its)
        print(f"Epoch {epoch_i}: train_acc={train_acc:.4f}, train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    # TTA evaluation
    tta_val_acc = evaluate(all_state, val_fn, val_num_its)
    print(f"Final TTA val acc: {tta_val_acc:.4f}")
    return tta_val_acc

if __name__ == "__main__":
    final_acc = main()
    print("Final accuracy:", final_acc)
