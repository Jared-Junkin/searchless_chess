from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from torch.distributed import barrier, init_process_group, destroy_process_group
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
import math
import wandb
from language_data_loader import LlamaLoader
from config_language import LanguageDataConfig
from test import decode
from torch.optim import Adam
from contextlib import nullcontext
import yaml
import time
import logging
from typing import Tuple, Callable, Any
import os
from hooks import *
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disabling Autotokenizer parallelism so we can do distributed training.
# os.environ["WANDB_MODE"] = "offline" # disabling wandb attempting to sync with my account, which doesn't exist.
from yaml import CLoader as Loader
from torch.nn.parallel import DistributedDataParallel as DDP
import traceback
from hooks import HookManager, forward_step_hook, log_batch_details_hook

### Jared Imports
from typing import Optional
def run_with_error_handling(
    func: Callable[..., None], 
    *args: Any, 
    log_path: Optional[str]="/workspace/searchless_chess/src/pythia/logs/errors.log",
    **kwargs: Any
) -> None:
    """
    Runs a function and logs any errors before exiting.
    
    Args:
        func (Callable[..., None]): The function to run.
        *args (Any): Positional arguments to pass to the function.
        **kwargs (Any): Keyword arguments to pass to the function.
        log_path: (Optional[str]): full path to the logfile (doesn't need to exist already) where error messages will get logged.
    """
    logger_errors = setupLogger(config={"log_path": log_path}) # file doesn't need to exist already.
    try:
        func(*args, **kwargs)
    except Exception as e:
        # Capture the full traceback as a formatted string
        full_traceback = traceback.format_exc()
        logger_errors.error(f"{str(e)}\n{full_traceback}\n")

        # Print for debugging
        print(f"{str(e)}\n{full_traceback}")
        
        exit(1)

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
    print(f"setting up logfile from {log_file}")
    
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

### End Jared Imports




@torch.no_grad()
def estimate_loss(model: AutoModelForCausalLM, eval_iters: int, train_loader: DataLoader,test_loader: DataLoader, tokenizer, logger, iter_num:int,  ctx=None) -> dict:
    
    out = {}
    model.eval()  # Set model to evaluation mode
    for split, loader in [('train', train_loader), ('val', test_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            seq, attn_mask, loss_mask, _, _ = next(loader)
            
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
                loss = loss * loss_mask  # Mask out non-target tokens (logical False = 1, logical True = 0
                loss = loss.sum() / (~loss_mask).sum()  # Normalize over unmasked tokens
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    # with ctx if ctx else torch.no_grad():
        
    #     seq, attn_mask, loss_mask,_,_ = next(loader)
    #     seq = seq.to(model.device)
    #     loss_mask = loss_mask.to(model.device)
    #     attn_mask = attn_mask.to(model.device)
        
    #     outputs = model(**{"input_ids": seq, "attention_mask": attn_mask}) # I think this will output gibberish now becaus I haven't trained the model to understand my new tokens yet.
    #     logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size). torch.argmax(logits, dim=-1) gets the token it thinks is most likely to follow the initial i tokens. 

    #     predicted_tokens = torch.argmax(logits, dim=-1)
    #     preds = tokenizer.decode(predicted_tokens[0])
    #     indices=(~loss_mask.int()).argmax(dim=1)
    #     logger.info(f"iter: {iter_num}, predicted_token: {tokenizer.decode(predicted_tokens[0][indices[0]])}, neighborhood: {tokenizer.decode(predicted_tokens[0][indices[0]-3:indices[0]+3])}, \n all_preds: {preds} \n\n\n")
        
                        ## to generate (inference time):
        # input_dict = {"input_ids": seq[0].unsqueeze(0), "attention_mask": attn_mask[0].unsqueeze(0)}
        # num_tokens = input_dict["input_ids"].shape[1]
        # tokens_gen = model.module.generate(**input_dict, max_length=len(seq[0]) + 7)
        # generated_tokens = outputs[0][num_tokens:]
        # txt = tokenizer.decode(generated_tokens)
        # logger.info(f"iter: {iter_num} predictions: {txt}")
    logger.info(f"done evals.")
    model.train()  # Set model back to training mode
    ### temporary code in here as a sanity check.
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
        _, _, _,_,_ = next(loader)
    print(F"all loads repeated successfully")
    
# def load_dataloader(config: dict, tokenizer: PreTrainedTokenizer, split: str) -> DataLoader:
#     # set up config used to create dataloader
#     world_size = config["ddp_world_size"]
#     local_rank = config["ddp_local_rank"]
#     train_data = LanguageDataConfig(
#         batch_size= config["batch_size"],
#         tokenizer=tokenizer,
#         tokenizer_save_path=config["out_dir"],
#         shuffle=config["shuffle"],
#         worker_count=config["worker_count"],  # 0 disables multiprocessing.
#         num_return_buckets=config["num_return_buckets"],
#         policy=config["policy"],
#         split=split,
#     )
#     # build and return dataloader
#     print(f"building data loader with world size = {world_size} and local rank = {local_rank}")
#     data_iter = build_data_loader_language(config=train_data, world_size=world_size, rank=local_rank)
#     return data_iter

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
        

def set_ddp_params(config: dict) -> Tuple[dict, str, int]:
    """
    Sets up Distributed Data Parallel (DDP) parameters.

    Args:
        config (dict): Configuration dictionary with at least a 'device' and 'backend' key.

    Returns:
        Tuple[dict, str, int]: Updated config, the device string, and whether DDP is active (1 for True, 0 for False).
    """
    # Determine if this is a DDP run
    ddp = int(os.environ.get('RANK', -1)) != -1  # Is this a DDP run?

    if ddp:
        # Initialize process group for DDP
        init_process_group(backend=config['backend'])

        # Fetch environment variables
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])

        # Set DDP-specific config parameters
        config["ddp_world_size"] = ddp_world_size
        config["ddp_local_rank"] = ddp_local_rank
        config["ddp_rank"] = ddp_rank

        # Set the device based on local rank
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)

        master_process = ddp_rank == 0 
        print(f"Using Multiprocessing: ddp world size is {ddp_world_size}, local rank is {ddp_local_rank}. Master Process: {master_process}")
    else:
        # Single GPU or CPU setup
        device = config['device']
        config["ddp_world_size"] = 0
        config["ddp_local_rank"] = 0
        config["ddp_rank"] = 0
        master_process = True
        ddp_local_rank=0

        print(f"Setting up on device {device}")

    return config, device, ddp, ddp_local_rank, master_process

def training(config: dict) -> None:
    # set variables
    logger = setupLogger(config=config)
    for key, value in config.items():
        logger.info(f"{key}: {value}\n")
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
    
    
    
    # hook_manager = HookManager()
    # loghook = LogBatchHook()
    # hook_manager.register_hook(loghook, context = {iter_num: int,
    #                                                 seq: torch.Tensor,
    #                                                 loss: torch.Tensor,
    #                                                 loss_mask: torch.Tensor,
    #                                                 logits: torch.Tensor,
    #                                                 ctx,
    #                                                 master_process: bool,
    #                                                 log_interval: int,
    #                                                 logger: logging.Logger,
    #                                                 tokenizer: PreTrainedTokenizer
    #                                                 }
    #                             )

    
    
    # decrease context length of model.
    # current_positional_embeddings = model.model.get_input_embeddings().weight
    # new_positinal_embeddings = current_positional_embeddings[:config["max_context_length"]].clone()
    # model.model.embed_tokens = torch.nn.Embedding(num_embeddings=config["max_context_length"], embedding_dim=model.config.hidden_size) # num embeddings = context length, embedding_dim = embedding dimension
    # model.model.embed_tokens.weight.data = new_positinal_embeddings
    
    config, device, ddp, ddp_local_rank, master_process = set_ddp_params(config=config)
    model.to(device)
    ## create dataloader
    train_iter = LlamaLoader(config=config, tokenizer=tokenizer, split="train")
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
    running_mfu = -1.0
    iter_num = 0
    
    learning_rate = config["learning_rate"]
    eval_interval = config["eval_interval"]
    always_save_checkpoint = config["always_save_checkpoint"]
    
    test_iter = LlamaLoader(config=config, tokenizer=tokenizer, split="test")
    ## unit tests to make sure the dataloaders can repeat over dataset and not raise stopIteration errors
    # test_data_loader(loader=test_iter, iters=10*len(test_iter))
    # print(f"made it through test loader")
    # test_data_loader(loader=train_iter, iters=max_iters)
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    seq, attn_mask, loss_mask,_,_ = next(train_iter)
    seq = seq.to(device)
    attn_mask = attn_mask.to(device)
    loss_mask=loss_mask.to(device)
    logger.info(f"Starting Train loop. Batch size: {config['batch_size']}, learning rate: {learning_rate}, gradient steps: {gradient_accumulation_steps}, weight_decay: {decay_lr}, ddp_local_rank: {ddp_local_rank}")
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
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(model=model, eval_iters = config['eval_iters'], train_loader=train_iter, test_loader=test_iter, tokenizer=tokenizer, logger=logger, iter_num=iter_num, ctx=ctx)  # Placeholder for evaluation function
            train_loss, val_loss = losses['train'], losses['val']
            print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, master_process: {master_process}")
           
            
            if logger:
                logger.info(f"iter: {iter_num}, train/loss: {train_loss}, val/loss: {val_loss}, lr: {lr}, mfu: {running_mfu * 100}, master_process: {master_process}, seq: {seq.shape}, loss_mask: {loss_mask.shape}, attn_mask: {attn_mask.shape}")
        
            # Checkpoint saving
            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                checkpoint_dir = os.path.join(out_dir, f"ckpt{iter_num}")
                # save model
                if ddp:
                    model.module.save_pretrained(checkpoint_dir)
                else:
                    model.save_pretrained(checkpoint_dir)
                # save tokenizer
                tokenizer.save_pretrained(checkpoint_dir)
                # save optimizer
                # torch.save(optimizer.state_dict(), checkpoint_dir)
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "opt_state_dict.pt"))
                
                
            if config["wandb_log"]:
                print(f"logging wandb outside")
                wandb.log({"val_loss": best_val_loss, "train_loss": train_loss})
            else:
                print("not using wandb")

        # Training step (including gradient accumulation)
        # logger.info(f"reached this line in iter {iter_num}")
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                loss, logits, shifted_mask, shifted_labels, outputs, seq_tmp, attn_mask = forward_step_hook(seq=seq,
                                                                               loss_mask=loss_mask,
                                                                               attn_mask=attn_mask,
                                                                               model=model,
                                                                               gradient_accumulation_steps=gradient_accumulation_steps,
                                                                               method=config["attn_method"])

                if iter_num % eval_interval == 0 and master_process:
                    with torch.no_grad():
                        # log_batch_details_hook
                        log_batch_details_hook(
                            outputs=outputs,
                            shifted_mask=shifted_mask,
                            shifted_labels=shifted_labels,
                            logits=logits,
                            iter_num=iter_num,
                            loss=loss,
                            config=config,
                            tokenizer=tokenizer,
                            logger=logger,
                            attn_mask=attn_mask,
                            seq_tmp=seq_tmp
                        )
                       


            seq, attn_mask, loss_mask,_,_ = next(train_iter)
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
        if iter_num % log_interval == 0:
            loss_value = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = 0 # TODO: replace with mfu function later.
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            if master_process:
                logger.info(f"iter {iter_num}: loss {loss_value:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
                if config["wandb_log"]:
                    print(f"logging wandb outside loop")
                    wandb.log({"val_loss": best_val_loss, "train_loss": loss_value})
                else:
                    print(f"not using wandb outside loop")
        iter_num += 1
        local_iter_num += 1

    print("Training complete.")
        
    
    # destroy subprocess at end
    if ddp:
        destroy_process_group()
        


def sweep_hyperparameters(config: dict, sweep_runs: int = 10) -> None:
    """
    Perform a hyperparameter sweep by repeatedly modifying the config
    and calling the training function.
    
    Args:
        config (dict): The initial configuration dictionary.
        sweep_runs (int): Number of hyperparameter configurations to test.
    """
    # Define the sweep configuration for WandB
    sweep_config = {
        "name": "Hyperparameter Sweep",
        "method": "bayes",  # Use Bayesian optimization
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "batch_size": {"values": [16, 32, 64]},
            "learning_rate": {"distribution": "uniform", "min": 1e-5, "max": 1e-4},
            "weight_decay": {"distribution": "uniform", "min": 1e-5, "max": 1e-2},
            "gradient_accumulation_steps": {"values": [2, 4, 8]},
        },
    }

    # Initialize the WandB sweep
    sweep_id = wandb.sweep(sweep_config, project=config["wandb_project"])

    def train_with_wandb():
        """
        Inner function called by WandB agent for each sweep run.
        """
        # Initialize WandB and get the run-specific hyperparameters
        wandb.init(project=config["wandb_project"], name=config["wandb_run_name"])
        sweep_params = wandb.config

        # Update config with sweep parameters
        config["batch_size"] = sweep_params.batch_size
        config["learning_rate"] = sweep_params.learning_rate
        config["decay_lr"] = sweep_params.weight_decay
        config["wandb_log"] = True  # Enable WandB logging for this run
        config["gradient_accumulation_steps"] = sweep_params.gradient_accumulation_steps
        # Call the unmodified training function
        training(config)

    # Launch the sweep
    wandb.agent(sweep_id, function=train_with_wandb, count=sweep_runs)

    
if __name__ == "__main__":

    # config_file = "/workspace/searchless_chess/src/config_hypsweep.yaml"
    config_file = "/workspace/searchless_chess/src/config_pythia.yaml"
    with open(config_file, "r") as stream:
        config = yaml.load(stream=stream, Loader=Loader)

    run_with_error_handling(training, config=config, log_path=config["log_path"])
    # sweep_hyperparameters(config=config, sweep_runs=20)
    
    # local_dir = "./Llama/llama-3.2-1B"
    # model = AutoModelForCausalLM.from_pretrained(local_dir)
    # tokenizer = AutoTokenizer.from_pretrained(local_dir)
