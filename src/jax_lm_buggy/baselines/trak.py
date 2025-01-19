import torch as ch
from trak import TRAKer
from trak import modelout_functions
from abc import ABC, abstractmethod
from typing import Iterable
from torch import Tensor

def get_avg_olg(logits, shifted_labels):
    num_samples, sequence_len, vocab_size = logits.shape
    num_y = num_samples * sequence_len
    logits_flat = logits.view(num_y, vocab_size)
    labels_flat = shifted_labels.view(num_y)
    bindex = ch.arange(num_y).to(logits_flat.device, non_blocking=True)
    ps = ch.nn.functional.softmax(logits_flat, dim=1)[bindex, labels_flat]
    mask = (labels_flat != -100).to(dtype=ps.dtype, non_blocking=True)
    ps = (ps * mask).view(num_samples, sequence_len)
    mean_p = ps.sum(-1) / mask.view(num_samples, -1).sum(-1)
    return (1 - mean_p).to(dtype=ch.float32)

def trak_out(logits, shifted_labels):
    num_samples, sequence_len, vocab_size = logits.shape
    num_y = num_samples * sequence_len
    logits_flat = logits.view(num_y, vocab_size)
    labels_flat = shifted_labels.view(num_y)

    bindex = ch.arange(num_y).to(logits_flat.device, non_blocking=True)

    logits_correct = logits_flat[bindex, labels_flat]

    cloned_logits = logits_flat.clone()
    cloned_logits[bindex, labels_flat] = ch.tensor(-ch.inf).to(logits_flat.device, non_blocking=True, dtype=logits_flat.dtype)
    logits_correct = logits_correct.view(num_samples, sequence_len)
    cloned_logits = cloned_logits.view(num_samples, sequence_len, vocab_size)
    margins = (logits_correct - cloned_logits.logsumexp(dim=-1))
    factor = (shifted_labels != -100).to(dtype=margins.dtype, non_blocking=True)
    margins = margins * factor
    avg_margin = margins.sum(dim=-1) / factor.sum(dim=-1)
    return avg_margin # margins.sum(dim=-1)

class LMModelOutput(modelout_functions.AbstractModelOutput):
    """Margin for text classification models. This assumes that the model takes
    in input_ids, token_type_ids, and attention_mask.

    .. math::

        \\text{logit}[\\text{correct}] - \\log\\left(\\sum_{i \\neq
        \\text{correct}} \\exp(\\text{logit}[i])\\right)

    """
    def __init__(self, temperature=1.0) -> None:
        super().__init__()
        self.softmax = ch.nn.Softmax(-1)
        self.loss_temperature = temperature

    @staticmethod
    def get_output(
        model,
        weights: Iterable[Tensor],
        buffers: Iterable[Tensor],
        input_ids: Tensor,
        labels: Tensor,
    ) -> Tensor:
        logits = ch.func.functional_call(model, (weights,), (input_ids[None, ...], labels[None, ...]))
        logits = logits[0]
        res = trak_out(logits, labels)
        assert res.shape[0] == 1
        return res[0]

    @staticmethod
    def get_out_to_loss_grad(model, weights, buffers, batch: Iterable[Tensor]) -> Tensor:
        input_ids, labels = batch
        logits = ch.func.functional_call(model, (weights,), (input_ids, labels))[0]
        res = get_avg_olg(logits, labels)
        res = res[..., None]
        return res

from tqdm import tqdm

def depad(x, y):
    is_not_pad = x != 0
    cumsum = is_not_pad.cumsum(-1)
    max_len = ch.max(cumsum)
    zx = x[:, :max_len]
    zy = y[:, :max_len]
    return zx, zy


def trak_matrix(model, state_dicts, ds_train, ds_test, task, hasher):
    model.eval()
    train_set_size = len(ds_train)

    if task == 'wikitext':
        bs = 16
    elif task == 'ift':
        bs = 2
    else:
        raise ValueError(f'Unknown task: {task}')

    dataloader_train = ch.utils.data.DataLoader(ds_train, batch_size=bs,
                                                shuffle=False)
    dataloader_test = ch.utils.data.DataLoader(ds_test, batch_size=bs,
                                               shuffle=False)

    trak_task = 'image_classification' if task == 'CIFAR' else LMModelOutput

    traker = TRAKer(model=model, task=trak_task, train_set_size=train_set_size,
                    save_dir=f'./trak_results/{hasher}')
    checkpoints = state_dicts

    for model_id, checkpoint in enumerate(checkpoints):
        traker.load_checkpoint(checkpoint, model_id=model_id)
        for batch in tqdm(dataloader_train):
            # batch should be a tuple of inputs and labels
            batch = depad(*batch)
            batch = [b.to(model.embedder.weight.device, non_blocking=True) for b in batch]
            x, _ = batch
            traker.featurize(batch=batch, num_samples=x.shape[0])

        traker.finalize_features()

    for model_id, checkpoint in enumerate(checkpoints):
        traker.start_scoring_checkpoint(checkpoint=checkpoint,
                                        model_id=model_id,
                                        exp_name='test',
                                        num_targets=len(ds_test))

        for batch in tqdm(dataloader_test):
            batch = [b.to(model.embedder.weight.device, non_blocking=True) for b in batch]
            batch = depad(*batch)
            traker.score(batch=batch, num_samples=len(batch[0]))

    scores = traker.finalize_scores(exp_name='test')
    return scores


