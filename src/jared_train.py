"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import pandas as pd
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from typing import Tuple, Dict, Any
from jared_models.nanoGPT import GPT, GPTConfig
# from jared import JaredPT # type: ignore
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
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
vocab_size = 1968 # number of possible legal moves in chess fen
eval_interval = 4000
log_interval = 100
eval_iters = 100
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
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
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
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
    targets_tensor: torch.Tensor = batch.to(device, non_blocking=True)

    # Shift right to create inputs
    inputs_tensor: torch.Tensor = shift_right(targets_tensor)

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
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop (first batch fetched above so we could register loss mask with torch for greater efficiency.)
X, Y = get_batch(split="train") # fetch the very first batch 

# print(f"input tensor is {X}\n\n\n")
# print(f"target tensor is {Y}")
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
start = time.time()
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num == 20:
        end = time.time()
        print(f"time to run 20 epochs: {end-start}")
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # logger.info(f"iter: {iter_num}, train/loss: {losses['train']}, val/loss: {losses['val']}, lr: {lr}, mfu: {running_mfu*100}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0: # we're not saving the 0th model
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                save_file = "ckpt" + str(iter_num) + ".pt"
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, save_file))
                ################################## Play against stockfish level 0
                # num_games_per_ckpt=100
                # stockfish_skill=0
                # stockfish_play_time = 0.1
                # config_play = {

                #     "RUN_FOR_ANALYSIS": True,
                #     "player_one_recording_name": save_file,
                #     "player_two_recording_name": "stockfish",
                #     "MAX_MOVES": 1000,
                #     "recording_file": "logs/determine.csv"
                # }
                # csv_file_path = (
                #     f"logs/{config_play['player_one_recording_name']}_vs_{config_play['player_two_recording_name']}"
                # )
                # csv_file_path = csv_file_path.replace(
                #     ".", "_"
                # )  # filenames can't have periods in them. Useful for e.g. gpt-3.5 models
                # csv_file_path += ".csv"
                # player_one = NanoGptPlayer(model_name=config_play['player_one_recording_name'])
                # player_two = StockfishPlayer(skill_level=stockfish_skill, play_time=stockfish_play_time)
                # play_game(player_one=player_one, 
                #         player_two=player_two, 
                #         config_options = config_play,
                #         max_games = num_games_per_ckpt
                #         )
                # ##
                # df = pd.read_csv(csv_file_path)
                # levels = range(1)
                # results = calcWinRate(df=df, stockfish_range=levels)
                
                # logger.info(f"iter: {iter_num}, train/loss: {losses['train']}, val/loss: {losses['val']}, lr: {lr}, mfu: {running_mfu*100}, W/L/D vs. Stockfish Level 0: {results['Stockfish 0'][0]}/{results['Stockfish 0'][1]}/{results['Stockfish 0'][2]}")

                ##################################
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # if iter_num < 10:
        #     print("Batch")
        #     print(X)
        #     print("y")
        #     print(Y)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        logger.info(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
