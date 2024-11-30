from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, AutoConfig
from torch.distributed import init_process_group, destroy_process_group
import torch
from torch.utils.data import DataLoader
import math
from language_data_loader import build_data_loader_language, LlamaLoader
from config_language import LanguageDataConfig
from test import decode
from torch.optim import Adam
from contextlib import nullcontext
import yaml
import time
import logging
from typing import Tuple, Dict, Any, List
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disabling Autotokenizer parallelism so we can do distributed training.

from yaml import CLoader as Loader
from torch.nn.parallel import DistributedDataParallel as DDP

# def get_batch(split: str, loader: DataLoader)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     try:
        

@torch.no_grad()
def estimate_loss(model: AutoModelForCausalLM, eval_iters: int, train_loader: DataLoader, test_loader: DataLoader, ctx=None) -> dict:
    
    out = {}
    model.eval()  # Set model to evaluation mode
    for split, loader in [('train', train_loader), ('val', test_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            seq, attn_mask, loss_mask = next(loader)
            
            seq = seq.to(model.device)
            attn_mask = attn_mask.to(model.device)
            loss_mask=loss_mask.to(model.device)

            with ctx if ctx else torch.no_grad():  # Use ctx if provided, else default no_grad context
                outputs = model(**{"input_ids": seq, "attention_mask": attn_mask}) 
                logits = outputs.logits
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  # Use 'none' to apply loss mask
                loss = loss_fn(logits.view(-1, logits.size(-1)), seq.view(-1))  # Shape: (batch_size * seq_len)

                # Apply the loss mask
                loss = loss.view(seq.size(0), seq.size(1))  # Reshape to (batch_size, seq_len)
                loss = loss * ~loss_mask  # Mask out non-target tokens (logical False = 1, logical True = 0
                loss = loss.sum() / (~loss_mask).sum()  # Normalize over unmasked tokens
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()  # Set model back to training mode
    return out


# Ensure this runs within each DDP process
def create_optimizer(model: AutoModelForCausalLM, config: dict) -> Adam:
    # Extract optimizer configuration
    weight_decay = config['weight_decay']
    learning_rate = config['learning_rate']
    betas = (config['beta1'], config['beta2'])
    device_type = config['device_type']  # GPU/CPU setup
    
    # Check device placement for parameters
    if device_type == 'cuda':
        # Ensure parameters are on the correct device
        model = model.to(torch.cuda.current_device())
    else:
        model = model.to(device_type)
    
    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay
    )
    return optimizer



        
# test_data_loader(loader=test_iter, iters=max_iters)
# this just iterates repeatedly through the loader iters times. Confirming it won't error out.
def test_data_loader(loader: LlamaLoader, iters: int)->None:
    print(f"type loader is {type(loader)}, len loader is: {len(loader)}")
    for i in range(iters):
        print(f"loading batch {i}")
        _, _, _, = next(loader)
    print(F"all loads repeated successfully")
    
def load_dataloader(config: dict, tokenizer: PreTrainedTokenizer, split: str) -> DataLoader:
    # set up config used to create dataloader
    world_size = config["ddp_world_size"]
    local_rank = config["ddp_local_rank"]
    train_data = LanguageDataConfig(
        batch_size= config["batch_size"],
        tokenizer=tokenizer,
        shuffle=config["shuffle"],
        worker_count=config["worker_count"],  # 0 disables multiprocessing.
        num_return_buckets=config["num_return_buckets"],
        policy=config["policy"],
        split=split,
    )
    # build and return dataloader
    print(f"building data loader with world size = {world_size} and local rank = {local_rank}")
    data_iter = build_data_loader_language(config=train_data, world_size=world_size, rank=local_rank)
    return data_iter

# learning rate decay scheduler (cosine with warmup)
def get_lr(it:int, config: dict)->float:
    # 1) linear warmup for warmup_iters steps
    if it < config['warmup_iters']:
        return config['learning_rate'] * it / config['warmup_iters']
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config['lr_decay_iters']:
        return config['min_lr']
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])
        
def setupLogger(config: dict, logger_name: str = "jaredLogger", log_level: str = "DEBUG") -> logging.Logger:
    """
    Sets up and returns a logger with a specified log file, logger name, and log level.

    Args:
        log_file (str): The path to the log file where logs will be written.
        logger_name (str): The name of the logger. Default is "jaredLogger".
        log_level (str): The logging level (e.g., "DEBUG", "INFO"). Default is "DEBUG".

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the directory for the log file exists
    log_file = config["log_path"]
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create or get the logger instance
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers if function is called multiple times
    if not logger.handlers:
        # Create a file handler
        file_log = logging.FileHandler(log_file)

        # Define the log format
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_log.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(file_log)

    return logger

        
def training(config: dict) -> None:
    # set variables
    logger = setupLogger(config=config)
    max_iters = config["max_iters"]
    out_dir = config["out_dir"]
    decay_lr = config["decay_lr"]
    log_interval = config["log_interval"]
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}['bfloat16']
    ctx = nullcontext() if config['device_type'] == 'cpu' else torch.amp.autocast(device_type=config['device_type'], dtype=ptdtype)
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # load model, tokenizer, dataloader
    # this tokenizer won't match exactly to begin with in terms of size because it is GPT4NeoXTokenizer and I'm using a pythia model. similar size of vocab size but not identical, because there are certain reserved tokens the model doesn't care about.
    model = AutoModelForCausalLM.from_pretrained(config["model_load_dir"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_load_dir"])
    
    # decrease context length of model.
    # current_positional_embeddings = model.model.get_input_embeddings().weight
    # new_positinal_embeddings = current_positional_embeddings[:config["max_context_length"]].clone()
    # model.model.embed_tokens = torch.nn.Embedding(num_embeddings=config["max_context_length"], embedding_dim=model.config.hidden_size) # num embeddings = context length, embedding_dim = embedding dimension
    # model.model.embed_tokens.weight.data = new_positinal_embeddings
    
    # set up distributed training if that's what we're doing:
    ddp = int(os.environ.get('RANK', -1)) != -1  # Is this a DDP run?
    if ddp:
        # Initialize process group for DDP
        init_process_group(backend=config['backend'])

        # Fetch environment variables
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        config["ddp_world_size"] = ddp_world_size
        config["ddp_local_rank"] = ddp_local_rank
        # Set the device based on local rank
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)

        # Move the model to the correct device
        model.to(device)
        
        print(f"Using Multiprocessing: ddp world size is {ddp_world_size}, local rank is {ddp_local_rank}")
        logger.info(f"world size is {ddp_world_size}, local rank is {ddp_local_rank}")
    else:
        # Single GPU or CPU setup
        device = config['device']
        model.to(device)
        config["ddp_world_size"] = 0
        config["ddp_local_rank"] = 0
        master_process = True
        print(f"setting up on device {device}")
    # set up ddp 
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    
    ## create dataloader
    train_loader = load_dataloader(config=config, tokenizer=tokenizer, split="train")
    train_iter = LlamaLoader(loader=train_loader)
    model.resize_token_embeddings(len(tokenizer)) # must resize the length of the modelstoken embeddings because we've added tokens to the tokenizer
    ## create optimizer 
    optimizer = create_optimizer(model, config)
    print(optimizer)
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    print(f"successfully wrapped model with DDP object for parallel processing")

    # seq, attn_mask, loss_mask = next(data_iter)
    # print(f"seq is {seq}")
    # pathvar = "/workspace/searchless_chess/src/Llama/languagedata/custom_tokenizer/"
    # print(decode(pathvar, seq))
    t0 = time.time()
    local_iter_num = 0
    best_val_loss = float('inf')
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    iter_num = 0
    
    learning_rate = config["learning_rate"]
    eval_interval = config["eval_interval"]
    always_save_checkpoint = config["always_save_checkpoint"]
    
    test_loader = load_dataloader(config=config, tokenizer=tokenizer, split="test")
    test_iter = LlamaLoader(loader=test_loader)
    ## unit tests to make sure the dataloaders can repeat over dataset and not raise stopIteration errors
    # test_data_loader(loader=test_iter, iters=max_iters)
    # print(f"made it through test loader")
    # test_data_loader(loader=train_iter, iters=max_iters)
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    seq, attn_mask, loss_mask = next(train_iter)
    seq = seq.to(device)
    attn_mask = attn_mask.to(device)
    loss_mask=loss_mask.to(device)
    logger.info("starting train loop")
    while iter_num < max_iters:
        print(f"beginning iteration {iter_num}")
        # Set learning rate for the current iteration
        if decay_lr:
            lr = get_lr(iter_num, config)
        else:
            lr = learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluation and logging every eval_interval steps
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model=model, eval_iters = config['eval_iters'], train_loader=train_iter, test_loader=test_iter, ctx=ctx)  # Placeholder for evaluation function
            train_loss, val_loss = losses['train'], losses['val']
            print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            
            if logger:
                logger.info(f"iter: {iter_num}, train/loss: {train_loss}, val/loss: {val_loss}, lr: {lr}, mfu: {running_mfu * 100}")
        

            # Checkpoint saving
            if val_loss < best_val_loss or always_save_checkpoint:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "tokenizer_config": tokenizer.save_pretrained(out_dir),  # Save tokenizer
                    "iteration": iter_num,  # Save iteration info
                }
                os.makedirs(out_dir, exist_ok=True)
                save_file = "ckpt" + str(iter_num) + ".pt"
                torch.save(checkpoint, os.path.join(out_dir, save_file))
                print(f"Checkpoint saved to {os.path.join(out_dir, save_file)}")

        # Training step (including gradient accumulation)
        # logger.info(f"reached this line in iter {iter_num}")
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                # Pass input through the model
                outputs = model(**{"input_ids": seq, "attention_mask": attn_mask}) # I think this will output gibberish now becaus I haven't trained the model to understand my new tokens yet.
                logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size). torch.argmax(logits, dim=-1) gets the token it thinks is most likely to follow the initial i tokens. 

                # Compute loss
                # Let's assume `labels` is a tensor of the same shape as `seq`, where the target tokens are stored.
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  # Use 'none' to apply loss mask
                loss = loss_fn(logits.view(-1, logits.size(-1)), seq.view(-1))  # Shape: (batch_size * seq_len)

                # Apply the loss mask
                loss = loss.view(seq.size(0), seq.size(1))  # Reshape to (batch_size, seq_len)
                loss = loss * ~loss_mask  # Mask out non-target tokens (logical False = 1, logical True = 0
                loss = loss.sum() / (~loss_mask).sum()  # Normalize over unmasked tokens
                loss = loss / gradient_accumulation_steps  # Normalize loss for accumulated gradients


            # Get the next batch 
            # immediately async prefetch next batch while model is doing the forward pass on the GPU

            seq, attn_mask, loss_mask = next(train_iter)
            seq = seq.to(device)
            loss_mask = loss_mask.to(device)
            attn_mask = attn_mask.to(device)
            
            # Backward pass with gradient scaling for mixed precision
            scaler.scale(loss).backward()

        if config["grad_clip"] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            loss_value = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = 0 # TODO: replace with mfu function later.
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            logger.info(f"iter {iter_num}: loss {loss_value:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            
        iter_num += 1
        local_iter_num += 1

    print("Training complete.")
        
    
    # destroy subprocess at end
    if ddp:
        destroy_process_group()
    

    
if __name__ == "__main__":

    config_file = "/workspace/searchless_chess/src/config_pythia.yaml"
    # config_file = "/workspace/searchless_chess/src/config_language.yaml"
    with open(config_file, "r") as stream:
        config = yaml.load(stream=stream, Loader=Loader)

    training(config=config)
    
    # local_dir = "./Llama/llama-3.2-1B"
    # model = AutoModelForCausalLM.from_pretrained(local_dir)
    # tokenizer = AutoTokenizer.from_pretrained(local_dir)

