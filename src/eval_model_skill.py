import os
import pandas as pd
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
from utils import _compute_all_possible_actions
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from typing import Tuple, Dict, Any
from jared_models.nanoGPT import GPT, GPTConfig

import logging
logger = logging.getLogger("jaredLogger")
logger.setLevel("DEBUG")
file_log =logging.FileHandler("CLLM.log")
logger.addHandler(file_log)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_log.setFormatter(formatter)
# ----------------------------------------------------------------------------- Deepmind imports
import config as config_lib
import data_loader
from jared_data_loader import build_data_loader_parallel
MOVE_TO_ACTION, ACTION_TO_MOVE = _compute_all_possible_actions()
# ----------------------------------------------------------------------------- Deepmind imports
def calcModelSkill(checkpoint_name: str = "ckpt600000.pt", eval_iters: int = 20, writeFilePath: str = "/workspace/searchless_chess/src/out/performance_sweep.txt")->None:

    out_dir = 'out'
    vocab_size = 1968 # number of possible legal moves in chess fen
    eval_interval = 4000
    log_interval = 100
    eval_only = False # if True, script exits right after the first eval
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
    # wandb logging
    wandb_log = False # disabled by default
    wandb_project = 'owt'
    wandb_run_name = 'gpt2' # 'run' + str(time.time())
    # data
    dataset = 'lichess_hf_dataset'
    gradient_accumulation_steps = 2 # used to simulate larger batch sizes
    batch_size = 1024 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 78 # number of tokens.
    # model
    n_layer = 8
    n_head = 8
    n_embd = 512
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate = 0.0003 # max learning rate
    max_iters = 600000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
    min_lr = 0.00003 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True # use PyTorch 2.0 to compile the model to be faster (jared is temporarily setting to false for debugging purposes.)
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('jared_configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
        
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    ## here's where we need to insert the deepmind stuff
    policy="behavioral_cloning"
    num_return_buckets=128
    train_data = config_lib.DataConfig(
        batch_size=batch_size,
        shuffle=True,
        worker_count=1,  # 0 disables multiprocessing.
        num_return_buckets=num_return_buckets,
        policy=policy,
        split='train'
    )
    test_data = config_lib.DataConfig(
        batch_size=batch_size,
        shuffle=True,
        worker_count=1,  # 0 disables multiprocessing.
        num_return_buckets=num_return_buckets,
        policy=policy,
        split='test',
    )
    # print("reached this line")

    if ddp:
        start_time = time.time()
        train_loader=build_data_loader_parallel(config=train_data, rank=ddp_rank, world_size=ddp_world_size)
        print(f"Created Train loader in  {time.time() - start_time} seconds")
        start_time = time.time()
        test_loader=build_data_loader_parallel(config=test_data, rank=ddp_rank, world_size=ddp_world_size)
        print(f"Created Test loader in  {time.time() - start_time} seconds")
    else:
        train_loader=data_loader.build_data_loader(config=train_data).__iter__()
        test_loader=data_loader.build_data_loader(config=test_data).__iter__()
    loader_iterators: Dict[str, Any] = {}

    def get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        def shift_right(targets: torch.Tensor) -> torch.Tensor:
            bos_column: torch.Tensor = torch.zeros(
                (targets.size(0), 1),
                dtype=targets.dtype,
                device=targets.device  # Ensure bos_column is on the same device
            )
            inputs: torch.Tensor = torch.cat([bos_column, targets[:, :-1]], dim=1)
            return inputs

        loader = train_loader if split == 'train' else test_loader

        try:
            batch: torch.Tensor = next(loader_iterators[split])
        except KeyError:
            # Initialize the iterator if not already done
            loader_iterators[split] = iter(loader)
            batch = next(loader_iterators[split])
        except StopIteration:
            # Re-initialize the iterator if needed
            loader_iterators[split] = iter(loader)
            batch = next(loader_iterators[split])

        # Move batch to device
        if device != "cpu":
            
            targets_tensor: torch.Tensor = batch.to(device, non_blocking=True)
            inputs_tensor: torch.Tensor = shift_right(targets_tensor)
        else: 
            targets_tensor = batch[0]
            bos_column = np.zeros((targets_tensor.shape[0], 1), dtype=targets_tensor.dtype)  # Create the first column of zeros
            inputs_tensor = np.hstack((bos_column, targets_tensor[:, :-1]))  # Horizontally stack the new column with the right-shifted arra
            targets_tensor = torch.tensor(targets_tensor, dtype=torch.int64)
            inputs_tensor = torch.tensor(inputs_tensor, dtype=torch.int64)

        return inputs_tensor, targets_tensor

    ## init models now. you will need to carefully adjust your model architecture to make sure that the attention mechanisms mask properly. 
    ## you mask everything except the last column for behavioral cloning.

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9


    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=vocab_size, dropout=dropout) # start with model_args from command line
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, checkpoint_name)
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
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
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

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])




    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(verbose: bool = False):
        out = {}
        model.eval()
        losses = torch.zeros(eval_iters)
        accs = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # print(f"starting iter {k}")
            X, Y = get_batch('test')
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            preds = torch.argmax(logits, dim=-1)
            preds = preds[:, -1].detach().cpu()
            gt_best_moves = Y[:,-1].detach().cpu()
            total_correct = sum(preds == gt_best_moves)
            total_moves = len(preds)
            batch_acc = total_correct/total_moves
            accs[k]=batch_acc
            print(f"iter {k}, acc: {batch_acc}%, loss: {loss.item()}")
            if verbose: 
                for i in range(len(preds)):
                    print(f"Best Move: {ACTION_TO_MOVE[int(gt_best_moves[i])]}, Chosen Move: {ACTION_TO_MOVE[int(preds[i])]}")

        model.train()
        return losses.mean(), accs.mean()*100

    # losses, mean_acc = estimate_loss(verbose=True)
    losses, mean_acc = estimate_loss()
    print(f"mean loss for {checkpoint_name}: {losses}, % accuracy: {mean_acc}%")
    with open(writeFilePath, 'a') as f:
        f.write(f"{checkpoint_name}, {losses.item()}, {mean_acc}\n")
    f.close()
    logger.info(f"{checkpoint_name}, {losses.item()}, {mean_acc}")

if __name__=="__main__":
    ################ to evaluate loss and accuracy for the final model
    checkpoint_name = "ckpt96000_causal.pt"
    writeFilePath="./out/performance_96000.txt"
    eval_iters = 1
    calcModelSkill(checkpoint_name=checkpoint_name, eval_iters=eval_iters, writeFilePath=writeFilePath)
    
    ################ to evaluate loss and accuracy of model trained across all epochs
    # ckpts = [i for i in range(4000, 602000, 8000)]
    # eval_iters = 20
    # for c in ckpts:
    #     # if c>=10000:
    #         checkpoint_name = "ckpt" + str(c) + ".pt"
    #         if os.path.exists(os.path.join("./out/", checkpoint_name)):
    #             print(F"evaluating model {checkpoint_name}")
    #             calcModelSkill(checkpoint_name=checkpoint_name, eval_iters=eval_iters)
    
    ############## to evaluate across all checkpoints for causal 
    # ckpts = [i for i in range(4000, 602000, 8000)]
    # writefilepath = "/workspace/searchless_chess/src/out/performance_sweep_causal.txt"
    # eval_iters = 20
    # for c in ckpts:
    #     # if c>=10000:
    #         checkpoint_name = "ckpt" + str(c) + "_causal.pt"
    #         if os.path.exists(os.path.join("./out/", checkpoint_name)):
    #             print(F"evaluating model {checkpoint_name}")
    #             calcModelSkill(checkpoint_name=checkpoint_name, eval_iters=eval_iters, writeFilePath=writefilepath)