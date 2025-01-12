import torch
from .model import GPTConfig, GPT
import os

def save_model(model, out_dir, model_args, *, optimizer=None, iter_num=0,
               best_val_loss=float('-inf'), config=None):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

def load_model(init_from, *, block_size, meta_vocab_size=None, out_dir=None,
               device='cuda', dropout=0., **model_args):
    '''
    guide: 
    - init_from: scratch - need all arguments other than out_dir
    - init_from: resume - need just block_size and out_dir
    - init_from: gpt2 - need just block_size and init_from
    '''
    iter_num = 0
    best_val_loss = float('-inf')

    model_args.update({
        'block_size': block_size,
        'dropout': dropout,
    })

    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        assert meta_vocab_size is not None
        model_args['vocab_size'] = meta_vocab_size
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]

        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint.get('iter_num', iter_num)
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value

    model.to(device)
    return model, iter_num, best_val_loss, model_args
