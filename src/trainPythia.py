from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, BitsAndBytesConfig
from torch.distributed import barrier, init_process_group, destroy_process_group
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler
import time
import logging
from typing import Tuple, Callable, Any
import os
from hooks import *
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disabling Autotokenizer parallelism so we can do distributed training.
from yaml import CLoader as Loader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import traceback



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
    logger.info(f"done eval")
    model.train()  # Set model back to training mode
    ### temporary code in here as a sanity check.
    return out

import os
import torch
from torch.optim import Adam
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
    # this tokenizer won't match exactly to begin with in terms of size because it is GPT4NeoXTokenizer and I'm using a pythia model. similar size of vocab size but not identical, because there are certain reserved tokens the model doesn't care about
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_load_dir"])
    
    
    if "LORA" in config and config["LORA"]==True:
        # QLORA (quantization parameters)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16  # or torch.float16 depending on your hardware
        )
        # quantize model weights to 4 bits precision to save memory.
        model = AutoModelForCausalLM.from_pretrained(
            config["model_load_dir"],
            quantization_config=bnb_config,
        )
        model = prepare_model_for_kbit_training(model) # this will ensure gradients continue to flow through our quantized model.
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"]
        lora_config = LoraConfig(
            r=8,  # Low rank dimension
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)     # Inject LoRA adapters
        model.gradient_checkpointing_enable()
    else:
        model = AutoModelForCausalLM.from_pretrained(config["model_load_dir"])
    # model.gradient_checkpointing_enable()
    model.train()
        
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params:,}")

    
    # decrease context length of model.
    # current_positional_embeddings = model.model.get_input_embeddings().weight
    # new_positinal_embeddings = current_positional_embeddings[:config["max_context_length"]].clone()
    # model.model.embed_tokens = torch.nn.Embedding(num_embeddings=config["max_context_length"], embedding_dim=model.config.hidden_size) # num embeddings = context length, embedding_dim = embedding dimension
    # model.model.embed_tokens.weight.data = new_positinal_embeddings
    
    config, device, ddp, ddp_local_rank, master_process = set_ddp_params(config=config)
    model.to(device)

        
    ## create dataloader
    train_iter = LlamaLoader(training_config=config, tokenizer=tokenizer, split="train")
    # model.resize_token_embeddings(len(tokenizer)) # must resize the length of the modelstoken embeddings because we've added tokens to the tokenizer
    ## create optimizer 
    optimizer = create_optimizer(model, config)
        ## store initial params for F2 regularization penalty
    if config["F2_regularization"]:
        with torch.no_grad():
            initial_params = []
            for (name, p) in model.named_parameters():
                if p.requires_grad:
                    initial_params.append(p.view(-1))
            initial_params_flat = torch.cat(initial_params) 
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
    if config["tens_log"]:
        os.makedirs(config["tens_log_dir"], exist_ok=True)
        writer = SummaryWriter(log_dir=config["tens_log_dir"])
    
    learning_rate = config["learning_rate"]
    eval_interval = config["eval_interval"]
    always_save_checkpoint = config["always_save_checkpoint"]
    
    test_iter = LlamaLoader(training_config=config, tokenizer=tokenizer, split="test")
    ## unit tests to make sure the dataloaders can repeat over dataset and not raise stopIteration errors
    # test_data_loader(loader=test_iter, iters=10*len(test_iter))
    # print(f"made it through test loader")
    # test_data_loader(loader=train_iter, iters=max_iters)
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    seq, attn_mask, loss_mask,fen,_ = next(train_iter)
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
                

        # Training step (including gradient accumulation)
        # logger.info(f"reached this line in iter {iter_num}")
        ## this prints the batch data across each individual process. should be different.
        # print(f"Rank {dist.get_rank()} - First sample FEN: {fen[0]}")
        # dist.barrier()
        acc = 0
        loss_avg = 0
        avg_correct_prob = 0
        avg_chosen_prob = 0
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                loss, logits, shifted_mask, shifted_labels, outputs, seq_tmp, attn_mask, sequence_accuracy, mean_correct_prob, mean_chosen_prob = forward_step_hook(seq=seq,
                                                                               loss_mask=loss_mask,
                                                                               attn_mask=attn_mask,
                                                                               model=model,
                                                                               gradient_accumulation_steps=gradient_accumulation_steps,
                                                                               method=config["attn_method"])

                acc += sequence_accuracy
                avg_correct_prob += mean_correct_prob
                avg_chosen_prob += mean_chosen_prob
                loss_avg += loss
                if iter_num % eval_interval == 0 and master_process and micro_step==0: # making it so only one micro step can log
                    logger.info(f"master proces: {master_process}, iter num: {iter_num}, micro batch: {micro_step}")
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
            # Only apply F2 penalty on the final micro-step before backpropagating the full batch loss
            if micro_step == gradient_accumulation_steps - 1 and config["F2_regularization"]:
                print(f"doing f2 regularization")
                loss = F2_loss_hook(
                    model=model,
                    F2_lambda=config["f2_lambda"],
                    initial_params_flat=initial_params_flat,
                    loss=loss
                )
            
            # Backward pass with gradient scaling for mixed precision
            scaler.scale(loss).backward()
        acc /= gradient_accumulation_steps
        if config["grad_clip"] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        
        ## this code verifies that the gradients across all process are synced after the gradient accumulation steps.
        # param_name, param = list(model.named_parameters())[0]
        # grad_norm = param.grad.data.norm().item()
        # print(f"Rank {dist.get_rank()} - {param_name} grad norm after all microsteps: {grad_norm}")
        # dist.barrier()

        scaler.step(optimizer)
        scaler.update()
        
        # log to wand if we want to.
        # log_wand_hook(wand_log = config["wandb_log"],
        #               master_process=master_process,
        #               iter_num=iter_num,
        #               log_interval=log_interval,
        #               outputs=outputs,
        #               model=model,
        #               ddp=ddp)
        # flush the gradients as soon as we can, no need for this memory anymore
        # if config["tens_log"] and master_process and iter_num % (eval_interval//2) == 0:
        #     logger.info(f"logging tensorboard on iter {iter_num}")
        #     nbins = 512  # Set the desired number of bins
        #     # bin_range = (-1, 1)  # If you need a specific range, tensorboard handles bins internally

        #     # Log attention histograms
        #     for i, attn in enumerate(outputs.attentions):
        #         attn_float32 = attn.detach().cpu().to(torch.float32).numpy()
        #         writer.add_histogram(f"attention_layer_{i}", attn_float32, global_step=iter_num, bins=nbins)

        #     # Access transformer layers
        #     en_obj = enumerate(model.module.gpt_neox.layers) if ddp else enumerate(model.gpt_neox.layers)
        #     for i, layer in en_obj:
        #         # Query-Key-Value projections
        #         qkv_weights = layer.attention.query_key_value.weight
        #         qkv_grads = qkv_weights.grad

        #         hidden_size = qkv_weights.shape[1]
        #         q_weights = qkv_weights[:hidden_size, :].detach().cpu().float().numpy()
        #         k_weights = qkv_weights[hidden_size:2 * hidden_size, :].detach().cpu().float().numpy()
        #         v_weights = qkv_weights[2 * hidden_size:, :].detach().cpu().float().numpy()

        #         # Log weights
        #         writer.add_histogram(f"q_weights_layer_{i}", q_weights, global_step=iter_num, bins=nbins)
        #         writer.add_histogram(f"k_weights_layer_{i}", k_weights, global_step=iter_num, bins=nbins)
        #         writer.add_histogram(f"v_weights_layer_{i}", v_weights, global_step=iter_num, bins=nbins)

        #         # Dense weights
        #         dense_weights = layer.attention.dense.weight
        #         dense_weights_data = dense_weights.detach().cpu().float().numpy()
        #         writer.add_histogram(f"dense_weights_layer_{i}", dense_weights_data, global_step=iter_num, bins=nbins)

        #         # Log gradients if available
        #         if qkv_grads is not None:
        #             q_grads = qkv_grads[:hidden_size, :].detach().cpu().float().numpy()
        #             k_grads = qkv_grads[hidden_size:2 * hidden_size, :].detach().cpu().float().numpy()
        #             v_grads = qkv_grads[2 * hidden_size:, :].detach().cpu().float().numpy()

        #             writer.add_histogram(f"q_grads_layer{i}", q_grads, global_step=iter_num, bins=nbins)
        #             writer.add_histogram(f"k_grads_layer{i}", k_grads, global_step=iter_num, bins=nbins)
        #             writer.add_histogram(f"v_grads_layer{i}", v_grads, global_step=iter_num, bins=nbins)

        #         dense_grads = dense_weights.grad
        #         if dense_grads is not None:
        #             dense_grads_data = dense_grads.detach().cpu().float().numpy()
        #             writer.add_histogram(f"dense_grads_layer{i}", dense_grads_data, global_step=iter_num, bins=nbins)
        optimizer.zero_grad(set_to_none=True)
        

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0:
            loss_value = loss_avg
            # loss_value = loss.item() # not multiplying by # of gradient accumulation steps. want avg loss so I can get a good idea of how the loss changes
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = 0 # TODO: replace with mfu function later.
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            if master_process:
                # changed the probs in here because I'm using log-softmax now for memory reasons.
                logger.info(f"iter {iter_num}: loss {loss_value:.4f}, time {(dt/gradient_accumulation_steps)*1000:.2f}ms, seq_acc: {acc*100:.2f}%, gt. conf: {math.exp(avg_correct_prob/gradient_accumulation_steps):.4f}, ans. conf: {math.exp(avg_chosen_prob/gradient_accumulation_steps):.4f} diff: {(math.exp(avg_chosen_prob/gradient_accumulation_steps) - math.exp(avg_correct_prob/gradient_accumulation_steps)):.4f}")
                if config["wandb_log"]:
                    print(f"logging wandb outside loop")
                    # wandb.log({"val_loss": best_val_loss, "train_loss": loss_value})
                    wandb.log({"train_loss": loss_value, "seq_acc": acc*100, "time": dt/gradient_accumulation_steps*1000, "gt. conf:": avg_correct_prob/gradient_accumulation_steps, "ans. confg": avg_chosen_prob/gradient_accumulation_steps, "diff": (avg_chosen_prob/gradient_accumulation_steps - avg_correct_prob/gradient_accumulation_steps)})
                else:
                    print(f"not using wandb outside loop")
        iter_num += 1
        local_iter_num += 1

    print("Training complete.")
        
    
    # destroy subprocess at end
    if ddp:
        destroy_process_group()
        


def sweep_hyperparameters(config: dict, sweep_runs: int = 30) -> None:
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
        "metric": {"name": "train_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 0.00009, "max": 0.0009},
            "lr_decay_iters": {"values": [60000, 600000, 6000000]}, # in my current setup, this will control how quickly the learning rate decays. less iters means it decays more quickly.
            "gradient_accumulation_steps": {"values": [4, 8, 32, 80, 160]},
            "f2_lambda": {"values": [0.0006, 0.0001, 0.00006]}
        },
    }

    # Initialize the WandB sweep
    sweep_id = wandb.sweep(sweep_config, project=config["wandb_project"])

    def train_with_wandb():
        wandb.init(project=config["wandb_project"])  # initialize wandb
        sweep_params = wandb.config
        
        unique_run_name = (f"{config['wandb_run_name']}_lr{config['learning_rate']:.5e}"
                           f"_gas{config['gradient_accumulation_steps']}"
                           f"_lambda{config['f2_lambda']}"
                           f"_decay{config['lr_decay_iters']}_{int(time.time())}")

        wandb.run.name = unique_run_name
        
        config["learning_rate"] = sweep_params.learning_rate
        config["min_lr"] = sweep_params.learning_rate/10
        config["lr_decay_iters"] = sweep_params.lr_decay_iters
        config["gradient_accumulation_steps"] = sweep_params.gradient_accumulation_steps
        config["wandb_log"] = True
        config["f2_lambda"] = sweep_params.f2_lambda
        

        # Debug print
        print(f"Starting run: {unique_run_name}")
        training(config)

    # Launch the sweep
    wandb.agent(sweep_id, function=train_with_wandb, count=sweep_runs)




        # # Construct a unique run name for logging clarity
        # unique_run_name = (f"{config_run['wandb_run_name']}_lr{config_run['learning_rate']:.5e}"
        #                    f"_gas{config_run['gradient_accumulation_steps']}"
        #                    f"_lambda{config_run['f2_lambda']}"
        #                    f"_decay{config_run['lr_decay_iters']}_{int(time.time())}")
    
if __name__ == "__main__":

    # config_file = "/workspace/searchless_chess/src/config_pthia_hypsweep.yaml"  # config file for wandb hyperparameter sweep
    # config_file = "/workspace/searchless_chess/src/config_pythia.yaml"        # config for pythia training from scratch
    # config_file = "/workspace/searchless_chess/src/config_llama.yaml"         # config for llama training from scratch
    config_file = "/workspace/searchless_chess/src/config_pythia_finetune.yaml"

    with open(config_file, "r") as stream:
        config = yaml.load(stream=stream, Loader=Loader)

    run_with_error_handling(training, config=config, log_path=config["log_path"])
    # run_with_error_handling(sweep_hyperparameters, config=config, log_path=config["log_path"])
    