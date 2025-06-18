
from typing import List, Dict, Any, Tuple, Callable, Optional
from abc import ABC, abstractmethod
from language_data_loader import LlamaLoader
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import torch.distributed as dist     
from torch.distributed import init_process_group
import math
# abstract hook class
import numpy as np
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import traceback
import os
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


def set_ddp_params(config: dict) -> Tuple[dict, str, bool, int, bool]:
    """
    Initialise distributed (NCCL) and return the info your code expects.

    Returns
    -------
    config  : dict   – updated with ddp_world_size / ddp_local_rank / ddp_rank
    device  : str    – e.g. "cuda:3"  or  "cpu"
    ddp     : bool   – True if multi‑process run
    local   : int    – local rank  (0 in single‑GPU mode)
    master  : bool   – True only on global rank 0
    """
    # ---------- Are we in a torchrun / mpirun launch?
    ddp = int(os.environ.get("RANK", "-1")) != -1

    if ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend=config.get("backend", "nccl"))

        ddp_rank       = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])

        # save into cfg for later convenience
        config.update(
            ddp_world_size=ddp_world_size,
            ddp_local_rank=ddp_local_rank,
            ddp_rank=ddp_rank,
        )

        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(ddp_local_rank)   # important for FSDP

        master_process = ddp_rank == 0
        print(f"[FSDP] world={ddp_world_size}  "
              f"rank={ddp_rank}  local_rank={ddp_local_rank}  "
              f"master={master_process}")

    else:
        # ---- single‑GPU (or CPU) fallback
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        config.update(ddp_world_size=1, ddp_local_rank=0, ddp_rank=0)
        master_process = True
        print(f"[Single] device={device}")

    return config, device, ddp, config["ddp_local_rank"], master_process

class HookManager:
    def __init__(self)->None:
        self.hooks: Dict[str, Callable[..., Any]] = {}
    
    def register_hook(self, name: str, func: Dict[str, Callable[..., Any]])->None:
        self.hooks[name] = func
        
    def execute_hook(self, name: str, *args: Any, **kwargs: Any)->None:
        if name in self.hooks:
            self.hooks[name](*args, **kwargs)


# def log_wand_hook(wand_log: bool, master_process: bool, iter_num: int, log_interval: int, outputs: torch.Tensor, model: AutoModelForCausalLM, ddp: bool)->None:
#     def plot_hist_image(data: torch.Tensor, name: str)->None:
#         data_to_plot = data.detach().cpu().to(torch.float32).numpy().ravel()
#         plt.figure(figsize=(6,4))
#         sns.violinplot(x=data_to_plot)
#         plt.title(f"Distribution of {name}")
#         wandb.log({f"{name}": wandb.Image(plt)}, step=iter_num)
#         plt.close()
        
#     if wand_log and master_process and iter_num % log_interval == 0:

#         # Log attention histograms
#         for i, attn in enumerate(outputs.attentions):
#             plot_hist_image(attn, f"attention_layer_{i}")

#         # Access transformer layers under model
#         en_obj = enumerate(model.module.gpt_neox.layers) if ddp else enumerate(model.gpt_neox.layers)
#         for i, layer in en_obj:
#             # Query-Key-Value projections
#             qkv_weights = layer.attention.query_key_value.weight
#             qkv_grads = qkv_weights.grad

#             # Split into query, key, and value
#             hidden_size = qkv_weights.shape[1]
#             q_weights = qkv_weights[:hidden_size, :]
#             k_weights = qkv_weights[hidden_size:2 * hidden_size, :]
#             v_weights = qkv_weights[2 * hidden_size:, :]

#             q_grads = qkv_grads[:hidden_size, :] if qkv_grads is not None else None
#             k_grads = qkv_grads[hidden_size:2 * hidden_size, :] if qkv_grads is not None else None
#             v_grads = qkv_grads[2 * hidden_size:, :] if qkv_grads is not None else None

#             # Plot weights for KQV
#             dense_weights = layer.attention.dense.weight
#             dense_grads = dense_weights.grad
            
#             plot_hist_image(q_weights, f"q_weights_layer_{i}")
#             plot_hist_image(k_weights, f"k_weights_layer_{i}")
#             plot_hist_image(v_weights, f"v_weights_layer_{i}")
#             plot_hist_image(dense_weights, f"dense_weights_layer_{i}")
            
#             plot_hist_image(q_grads, f"q_grads_layer_{i}")
#             plot_hist_image(k_grads, f"k_grads_layer_{i}")
#             plot_hist_image(v_grads, f"v_grads_layer_{i}")
#             plot_hist_image(dense_grads, f"dense_grads_layer_{i}")
            

            
            

def get_batch_hook():
    pass

def generate_best_move_hook(model: AutoModelForCausalLM, 
                            tokenizer: AutoTokenizer,
                            temperature: float = 1.0,
                            input_ids: torch.Tensor = None,
                            fen_tokens: np.ndarray = None, 
                            legal_move_tokens = None, 
                            prompt_components: List[np.ndarray] = None, 
                            )->None:
        if input_ids is None: # in this case, we have to construct the prompt
            assert prompt_components and fen_tokens and legal_move_tokens
        
            prompt_tokens = np.concatenate(
                    [
                        prompt_components[0],
                        fen_tokens,
                        prompt_components[1],
                        legal_move_tokens,
                        prompt_components[2]
                    ]
                )
                
            # Now convert to torch tensors
            input_ids = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(model.device)
            
            
        attention_mask = torch.zeros_like(input_ids, dtype=torch.long) 
        attention_mask[input_ids!=0] = 1 # prevent model from attending to padding tokens

        # Now use model.generate() to produce the next tokens
        # Let's say we want to generate up to 4 tokens for the best move
        generated_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4,  # This will produce up to 4 new tokens after the prompt
            temperature=temperature,
            do_sample=True,       # or False if you want greedy
            top_k=50,             # optional, depends on your decoding strategy
            top_p=0.95            # optional, depends on your decoding strategy
        )
                # The generated output includes the original prompt plus the new tokens
        all_tokens = generated_outputs[0].cpu().numpy()

        # The new tokens are the last 4 tokens in `all_tokens[len(prompt_tokens):]`
        best_move_ids = all_tokens[input_ids.shape[-1]:]  
        best_move = tokenizer.decode(best_move_ids)
                
        return best_move

'''
    # optionally impose an element-wise F2 penalty if prevent weights shifting too far from initial distribution
    if F2_penalty and F2_lambda is not None and initial_params is not None:
        current_params = []
        for (name, p) in model.named_parameters():
            if p.requires_grad:
                current
'''

def F2_loss_hook(model: AutoModelForCausalLM, 
                 initial_params_flat: torch.Tensor, 
                 F2_lambda: float,
                 loss: float)->float:
    current_params = []
    for (name, p) in model.named_parameters():
        if p.requires_grad:
            current_params.append(p.view(-1))
    current_params = torch.cat(current_params)
    kl_loss = torch.nn.functional.mse_loss(current_params, initial_params_flat, reduction='sum')
    loss = loss + F2_lambda * kl_loss
    return loss
    
    
# def forward_step_train(seq: torch.Tensor, loss_mask: torch.Tensor, model: AutoModelForCausalLM, attn_mask: torch.Tensor):
#     seq_tmp = seq.detach().clone()
#     seq_tmp[loss_mask == 1] = 1  # Replace target tokens in the input with pad (1)
#     outputs = model(input_ids = seq_tmp, attention_mask=attn_mask)
#     logits = outputs.logits
def fast_forward_step_hook_grad(seq: torch.Tensor,
                           loss_mask: torch.Tensor,
                           attn_mask: torch.Tensor,
                           model: AutoModelForCausalLM,
                           gradient_accumulation_steps: int,
                           method: str = "dont_attend_to_prev_answers",
                           attentions: bool = True,
                           alpha: float = 0.5  # <-- weight for all-or-nothing penalty
                          ) -> torch.Tensor:
    """
    Modified to add an auxiliary "all-or-nothing" penalty if any token is wrong.
      - 'alpha' controls how harshly we penalize an imperfect sequence.
    """

    # 1) Forward pass
    outputs = model(input_ids=seq, attention_mask=attn_mask, output_attentions=False)
    logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)

    # 2) Shift labels by one so model[i] matches seq[i+1]
    shifted_labels = seq[:, 1:].clone()
    shifted_mask   = loss_mask[:, 1:]
    logits         = logits[:, :-1, :]  # align predictions to labels

    # 3) Mark non-target positions with -100 (ignore_index)
    shifted_labels[shifted_mask == 0] = -100

    # 4) Standard token-level cross-entropy (provides partial-credit gradient)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    ce_loss = loss_fn(logits.reshape(-1, logits.size(-1)), shifted_labels.view(-1))

    # 5) Compute the "all-or-nothing" correctness check (non-differentiable indicator)
    with torch.no_grad():
        # Argmax over vocab for each token
        preds = logits.argmax(dim=-1)  # shape: (batch_size, seq_len-1)

        # For each sample & token: check if correct OR label is -100 (masked)
        token_correct_or_ignored = (preds == shifted_labels) | (shifted_labels == -100)
        # fully_correct_mask[i] = True iff *every* token in sample i is correct (or -100)
        fully_correct_mask = token_correct_or_ignored.all(dim=1).float()  # shape: (batch_size,)

    # 6) Auxiliary penalty: alpha if not fully correct, else 0
    #    e.g. L_aux(i) = alpha * (1 - c_i)
    #    Then average over the batch
    aux_loss = alpha * (1.0 - fully_correct_mask).mean()

    # 7) Combine the two losses
    total_loss = ce_loss + aux_loss

    # 8) Scale by gradient_accumulation_steps
    total_loss = total_loss / gradient_accumulation_steps

    return total_loss
def fast_forward_step_hook(seq: torch.Tensor,
                      loss_mask: torch.Tensor,
                      model: AutoModelForCausalLM,
                      gradient_accumulation_steps: int,
                      method: str = "dont_attend_to_prev_answers",
                      attentions: bool = True
                      )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_tmp = seq
    outputs = model(input_ids=seq_tmp, output_attentions=False)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    # Shift labels by one
    shifted_labels = seq[:, 1:].clone()      # The model at position i predicts seq[i+1]
    shifted_mask = loss_mask[:, 1:]          # Shift mask as well
    logits = logits[:, :-1, :]               # Align logits so they match shifted_labels
    

    # Set non-target positions to -100
    shifted_labels[shifted_mask == 0] = -100

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), shifted_labels.view(-1))
    loss = loss / gradient_accumulation_steps
    return loss
# this must be run on the master process
def eval_hook(seq: torch.Tensor,
                      loss_mask: torch.Tensor,
                      model: AutoModelForCausalLM,
                      )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_tmp = seq # 
    
    ######### jared's sanity check:
    # # Step 1: Find the index of the 0th target token
    # target_indices = torch.where(loss_mask[0])[0]  # Indices of target tokens in the sequence
    # if target_indices.numel() == 0:
    #     raise ValueError("No target tokens found in the sequence.")
    # first_target_index = target_indices[0].item()

    # # Step 2: Slice seq_tmp[0] and attn_mask[0] up to the 0th target token's index
    # sliced_seq_tmp = seq_tmp[0, :first_target_index]
    # sliced_attn_mask = attn_mask[0, :first_target_index]

    # # Step 3: Initialize autoregressive generation variables
    # generated_tokens = []  # Store generated tokens
    # max_generation_steps = 7  # Generate 7 tokens

    # for step in range(max_generation_steps):
    #     # Add batch dimension for input to model
    #     sliced_seq_tmp = sliced_seq_tmp.unsqueeze(0)  # Shape: (1, seq_len)
    #     sliced_attn_mask = sliced_attn_mask.unsqueeze(0)  # Shape: (1, seq_len)

    #     # Pass current sequence through the model
    #     outputs = model(input_ids=sliced_seq_tmp, attention_mask=sliced_attn_mask)
    #     logits = outputs.logits  # Shape: (1, seq_len, vocab_size)

    #     # Get the predicted token for the last position
    #     logit_for_last_token = logits[0, -1, :]  # Last token's logits
    #     pred_token = torch.argmax(logit_for_last_token, dim=-1)  # Predicted token ID
    #     generated_tokens.append(pred_token.item())  # Append to generated tokens list

    #     # Step 4: Update seq_tmp and attn_mask for next step
    #     sliced_seq_tmp = torch.cat([sliced_seq_tmp.squeeze(0), pred_token.unsqueeze(0)], dim=0)  # Add token to seq_tmp
    #     sliced_attn_mask = torch.cat([sliced_attn_mask.squeeze(0), torch.tensor([1]).to(sliced_attn_mask.device)], dim=0)  # Add attention mask

    # # Step 5: Print the resulting 7 tokens
    # print("Pred 0", generated_tokens)
    ####################################### 
    
    outputs = model(input_ids=seq_tmp, output_attentions=False)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    # Shift labels by one
    shifted_labels = seq[:, 1:].clone()      # The model at position i predicts seq[i+1]
    shifted_mask = loss_mask[:, 1:]          # Shift mask as well
    logits = logits[:, :-1, :]               # Align logits so they match shifted_labels
    

    # Set non-target positions to -100
    shifted_labels[shifted_mask == 0] = -100

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), shifted_labels.view(-1))
    # calculate the sequence accuracy.
    pred = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
    valid_positions = (shifted_labels != -100)
    sequence_accuracy = ((pred == shifted_labels) | ~valid_positions).all(dim=1).float().mean()
    
    # calculate the mean probability of the actual best move
    probs = torch.log_softmax(logits, dim=-1) # more memory efficient than softmax
    correct_probs = probs[valid_positions, shifted_labels[valid_positions]]
    mean_correct_prob = correct_probs.mean()
    
    # calculate the mean probability of the chosen move
    chosen_probs = probs[valid_positions, pred[valid_positions]]
    mean_chosen_prob = chosen_probs.mean()
    return loss, logits, shifted_mask, shifted_labels, outputs, seq_tmp, sequence_accuracy, mean_correct_prob, mean_chosen_prob


def forward_step_hook(seq: torch.Tensor,
                      loss_mask: torch.Tensor,
                      attn_mask: torch.Tensor,
                      model: AutoModelForCausalLM,
                      gradient_accumulation_steps: int,
                      method: str = "attend_to_prev_answers",
                      attentions: bool = True
                      )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    if method == "dont_attend_to_prev_answers":
        seq_tmp = seq.detach().clone()
        seq_tmp[loss_mask == 1] = 1  # Replace target tokens in the input with pad (0)
    elif method == "attend_to_prev_answers":
        seq_tmp = seq # 
        # attn_mask = (seq_tmp > 1).long().to(seq.device) # if we're attending to previous answers, we should modify attention mask to do so. (really all this should be done in the dataloader, but I'm short on time.)
    else:
        raise NotImplementedError(f"Method {method} is not a valid input. See hooks.py ~ forward_step_hook for details.")
    
    ######### jared's sanity check:
    # # Step 1: Find the index of the 0th target token
    # target_indices = torch.where(loss_mask[0])[0]  # Indices of target tokens in the sequence
    # if target_indices.numel() == 0:
    #     raise ValueError("No target tokens found in the sequence.")
    # first_target_index = target_indices[0].item()

    # # Step 2: Slice seq_tmp[0] and attn_mask[0] up to the 0th target token's index
    # sliced_seq_tmp = seq_tmp[0, :first_target_index]
    # sliced_attn_mask = attn_mask[0, :first_target_index]

    # # Step 3: Initialize autoregressive generation variables
    # generated_tokens = []  # Store generated tokens
    # max_generation_steps = 7  # Generate 7 tokens

    # for step in range(max_generation_steps):
    #     # Add batch dimension for input to model
    #     sliced_seq_tmp = sliced_seq_tmp.unsqueeze(0)  # Shape: (1, seq_len)
    #     sliced_attn_mask = sliced_attn_mask.unsqueeze(0)  # Shape: (1, seq_len)

    #     # Pass current sequence through the model
    #     outputs = model(input_ids=sliced_seq_tmp, attention_mask=sliced_attn_mask)
    #     logits = outputs.logits  # Shape: (1, seq_len, vocab_size)

    #     # Get the predicted token for the last position
    #     logit_for_last_token = logits[0, -1, :]  # Last token's logits
    #     pred_token = torch.argmax(logit_for_last_token, dim=-1)  # Predicted token ID
    #     generated_tokens.append(pred_token.item())  # Append to generated tokens list

    #     # Step 4: Update seq_tmp and attn_mask for next step
    #     sliced_seq_tmp = torch.cat([sliced_seq_tmp.squeeze(0), pred_token.unsqueeze(0)], dim=0)  # Add token to seq_tmp
    #     sliced_attn_mask = torch.cat([sliced_attn_mask.squeeze(0), torch.tensor([1]).to(sliced_attn_mask.device)], dim=0)  # Add attention mask

    # # Step 5: Print the resulting 7 tokens
    # print("Pred 0", generated_tokens)
    ####################################### 
    
    outputs = model(input_ids=seq_tmp, attention_mask=attn_mask, output_attentions=False)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    # Shift labels by one
    shifted_labels = seq[:, 1:].clone()      # The model at position i predicts seq[i+1]
    shifted_mask = loss_mask[:, 1:]          # Shift mask as well
    logits = logits[:, :-1, :]               # Align logits so they match shifted_labels
    

    # Set non-target positions to -100
    shifted_labels[shifted_mask == 0] = -100

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), shifted_labels.view(-1))
    loss = loss / gradient_accumulation_steps
    
    # calculate the sequence accuracy.
    pred = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
    valid_positions = (shifted_labels != -100)
    sequence_accuracy = ((pred == shifted_labels) | ~valid_positions).all(dim=1).float().mean()
    
    # calculate the mean probability of the actual best move
    probs = torch.log_softmax(logits, dim=-1) # more memory efficient than softmax
    correct_probs = probs[valid_positions, shifted_labels[valid_positions]]
    mean_correct_prob = correct_probs.mean()
    
    # calculate the mean probability of the chosen move
    chosen_probs = probs[valid_positions, pred[valid_positions]]
    mean_chosen_prob = chosen_probs.mean()
    return loss, logits, shifted_mask, shifted_labels, outputs, seq_tmp, attn_mask, sequence_accuracy, mean_correct_prob, mean_chosen_prob


def log_batch_details_hook(outputs: torch.Tensor,
                           shifted_mask: torch.Tensor,
                           shifted_labels: torch.Tensor,
                           logits: torch.Tensor,
                           iter_num: int,
                           loss: torch.Tensor,
                           config: dict,
                           tokenizer: AutoTokenizer,
                           logger: logging.Logger,
                           attn_mask: torch.Tensor,
                           seq_tmp: torch.Tensor,
                           )->None:
    ## temporarily removing attended tokens from this because it is extremely memory inefficient.
    # attention_weights = outputs.attentions  # List of attention tensors
    # padding_mask = (attn_mask == False).unsqueeze(1).unsqueeze(2)
    # for layer_idx, layer_attention in enumerate(attention_weights): # this will verify that 0 attention is being given to padding tokens. 
    #     padding_attention = layer_attention * padding_mask
    #     sum_padding_attention = padding_attention.sum()
    #     logger.info(f"Layer {layer_idx}: Padding attention sum: {sum_padding_attention}")
    
    # attended_tokens = torch.argmax(attention_weights[0][0, 0], dim=-1)

    # We have already computed:
    # shifted_labels = seq[:, 1:].clone()
    # shifted_mask = loss_mask[:, 1:]
    # logits = logits[:, :-1, :] (done before computing the loss)
    #
    # Now we must extract the indices of the target tokens from shifted_mask.
    
    token_indices = torch.nonzero(shifted_mask)  # Indices of tokens we care about
    row_indices = token_indices[:, 0]
    col_indices = token_indices[:, 1]

    # best_moves are the ground-truth tokens from shifted_labels at these positions
    best_moves = shifted_labels[row_indices, col_indices]

    # predicted_tokens are the model's predictions at these positions
    full_predicted_tokens = torch.argmax(logits, dim=-1)  # (batch, seq_len)
    predicted_tokens = full_predicted_tokens[row_indices, col_indices]

    # Compute probabilities for ground truth and chosen answers
    # Extract the logits for these positions
    row_logits = logits[row_indices, col_indices, :]  # (num_targets, vocab_size)
    row_probs = torch.softmax(row_logits, dim=-1)

    # Ground truth probabilities
    ground_truth_probs = []
    chosen_answer_probs = []
    grouped_best_moves = []
    grouped_predicted_tokens = []
    grouped_ground_truth_probs = []
    grouped_chosen_answer_probs = []

    unique_rows = torch.unique(row_indices)
    for row in unique_rows:
        current_row_mask = (row_indices == row)
        
        row_best_moves = best_moves[current_row_mask]  # ground truth tokens for this row
        row_preds = predicted_tokens[current_row_mask] # predicted tokens for this row

        # Extract row-specific probabilities
        row_start = torch.nonzero(current_row_mask).min()
        row_end = torch.nonzero(current_row_mask).max()

        # row_probs for this specific row
        # Instead of slicing by range, we can just index directly:
        # The positions in row_indices, col_indices corresponding to this row are directly mask-selected
        row_positions = torch.where(current_row_mask)[0]
        current_row_probs = row_probs[row_positions, :]

        # Ground truth probabilities for each token
        gt_probs = current_row_probs[torch.arange(current_row_probs.size(0)), row_best_moves]
        chosen_probs = current_row_probs[torch.arange(current_row_probs.size(0)), row_preds]

        grouped_best_moves.append(row_best_moves.tolist())
        grouped_predicted_tokens.append(row_preds.tolist())
        grouped_ground_truth_probs.append(gt_probs.tolist())
        grouped_chosen_answer_probs.append(chosen_probs.tolist())
    log_batch_info(
        iter_num=iter_num,
        loss=loss,
        predicted_tokens=grouped_predicted_tokens,
        best_moves=grouped_best_moves,
        ground_truth_probs=grouped_ground_truth_probs,
        chosen_answer_probs=grouped_chosen_answer_probs,
        config=config,
        tokenizer=tokenizer,
        logger=logger,
        attended_tokens=[],
        attn_mask=attn_mask,
        seq=seq_tmp[0]  # seq_tmp is the input to the model (prompt+pad), for logging
    )
       
def log_batch_info(iter_num, loss, predicted_tokens, best_moves, ground_truth_probs, chosen_answer_probs, config, tokenizer, logger, attended_tokens, attn_mask, seq):
    # Helper function for conditional formatting of probabilities
    def format_prob(prob):
        return f"{prob:.4e}" if prob < 0.0001 else f"{prob:.4f}"
    
    # Compute average probabilities for each move, avoiding division by 0
    def compute_average_prob(probs, num_tokens):
        probs_tensor = torch.tensor(probs)
        return probs_tensor.sum() / max(num_tokens, 1)  # Avoid division by 0

    # Compute accuracy as the percentage of moves where all tokens match
    batch_size = len(best_moves)
    accuracy = sum(
        torch.equal(torch.tensor(predicted_tokens[i]), torch.tensor(best_moves[i])) 
        for i in range(batch_size)
    ) / batch_size * 100  # Accuracy in percentage

    avg_best_move_probs = [
        compute_average_prob(ground_truth_probs[i], len(best_moves[i]))
        for i in range(batch_size)
    ]
    avg_chosen_move_probs = [
        compute_average_prob(chosen_answer_probs[i], len(predicted_tokens[i]))
        for i in range(batch_size)
    ]

    # Log header line with loss and accuracy
    logger.info(
        f"Iter: {iter_num}, "
        f"Loss = {loss:.4f}, "
        f"Accuracy (all tokens match): {accuracy:.2f}%, "
        f"Mean best move prob: {torch.mean(torch.tensor(avg_best_move_probs)):.4f}, "
        f"Mean chosen move prob: {torch.mean(torch.tensor(avg_chosen_move_probs)):.4f}"
    )
    print(f"logging batch details")
    logger.info(f"Attended tokens for sample 0: {attended_tokens}")
    logger.info(f"Attention mask for sample 0: {attn_mask[0]}")
    token_indices = torch.nonzero(attn_mask[0])
    logger.info(f"Seq is {seq}")
    # logger.info(f"Seq is {seq[token_indices].squeeze(1)}")

    # Header for the table columns
    logger.info(f"{'Best Move':<20}{'Chosen Move':<20}{'Best Move Prob':<20}{'Chosen Move Prob':<20}")

    # Table rows for each sample in the batch
    for i in range(batch_size):
        best_move_tokens = best_moves[i]
        predicted_move_tokens = predicted_tokens[i]
        
        best_move_str = tokenizer.decode(best_move_tokens)
        chosen_move_str = tokenizer.decode(predicted_move_tokens)
        best_move_prob = avg_best_move_probs[i]  # Average probability for the best move
        chosen_move_prob = avg_chosen_move_probs[i]  # Average probability for the chosen move
        
        logger.info(
            f"{best_move_str:<20}{chosen_move_str:<20}{format_prob(best_move_prob):<20}{format_prob(chosen_move_prob):<20}"
        )

# class LogBatchHook(Hook):
#     """
#     A hook to log detailed batch results, including the loss, best move statistics,
#     predicted moves, and their associated probabilities during training.

#     Arguments:
#         iter_num (int): The current iteration number in the training loop.
#         seq (torch.Tensor): The tensor containing the sequence of tokens for each sample in the batch.
#         loss (torch.Tensor): The loss value for the current batch.
#         loss_mask (torch.Tensor): A mask indicating which tokens in the sequence contribute to the loss.
#         logits (torch.Tensor): The model's raw logits for each token in the sequence (before softmax).
#         ctx (Any): A context manager for handling resources or computations (e.g., for distributed training).
#         master_process (bool): A flag indicating whether the current process is the master process (useful in distributed setups).
#         log_interval (int): The frequency (in iterations) at which to log batch results.
#         logger (logging.Logger): The logger instance used for logging batch results.
#         tokenizer (AutoTokenizer): The tokenizer used to decode token indices into human-readable text.

#     Purpose:
#         This hook is used to log the current batch's training information, including:
#         - The current iteration and loss.
#         - The percentage of times the best move was predicted correctly.
#         - The mean probability of the best move for the batch.
#         - A table showing the best move, the chosen move, and their associated probabilities for each sample in the batch.

#     Outputs:
#         Logs information to the provided logger. Specifically, it logs:
#         - The iteration number, loss, percent of best moves chosen, and mean best move probability.
#         - A formatted table showing the best move, the chosen move, and their corresponding probabilities for each token in the batch.
#     """
    
#     # Helper function for conditional formatting of probabilities
#     def format_prob(self, prob: float)->str:
#         return f"{prob:.4e}" if prob < 0.0001 else f"{prob:.4f}"
    
#     def __call__(
#         self,
#         iter_num: int,
#         seq: torch.Tensor,
#         loss: torch.Tensor,
#         loss_mask: torch.Tensor,
#         logits: torch.Tensor,
#         ctx,
#         master_process: bool,
#         log_interval: int,
#         logger: logging.Logger,
#         tokenizer: AutoTokenizer
#     )->None:
#         print(f"Successfully entered hook")
#         if iter_num % log_interval == 0 and master_process:
#             print(f"master process")
#             with ctx:
#                     # back out best move from each sample
#                     token_indices = torch.nonzero(~loss_mask) # find false values in loss mask (target tokens for prediction)
#                     row_indices = token_indices[:,0]
#                     col_indices = token_indices[:,1]
#                     best_moves = seq[row_indices, col_indices] # gives us all best moves (loss_mask[row_indices, col_indices]=all false values)
                    
#                     ## to back predicted tokens out
#                     predicted_tokens = torch.argmax(logits, dim=-1) # get predicted tokens
#                     predicted_tokens = predicted_tokens[row_indices, col_indices]
                    
#                     ## back out probabilities associated iwth best move, move chosen
#                     # ground_truth_logits = logits[row_indices, col_indices, best_moves] # get the logit associated with each best move (batch_size,)
#                     token_probs = torch.softmax(logits[row_indices, col_indices], dim=-1) # get the probability associated with each logit (batch_size, vocab_size)
#                     ground_truth_probs = token_probs[torch.arange(token_probs.size(0)), best_moves] # probabilities associated with each best move
#                     chosen_answer_probs = token_probs[torch.arange(token_probs.size(0)), predicted_tokens] # probabilities associated with each move chosen
#                     # Header line with loss and percent best move chosen
#                     logger.info(
#                         f"Iter: {iter_num}, "
#                         f"Loss = {loss:.4f}, "
#                         f"Percent best move chosen: {sum(predicted_tokens == best_moves) / predicted_tokens.shape:.2%}"
#                         f"Mean best move prob: {torch.mean(ground_truth_probs)}"
#                     )

#                     # Header for the table columns
#                     logger.info(f"{'Best Move':<20}{'Chosen Move':<20}{'Best Move Prob':<20}{'Chosen Move Prob':<20}")

#                     # Table rows for each sample in the batch
#                     for i in range(predicted_tokens.shape):
#                         best_move = tokenizer.decode(best_moves[i])
#                         chosen_move = tokenizer.decode(predicted_tokens[i])
#                         best_move_prob = ground_truth_probs[i].item()
#                         chosen_move_prob = chosen_answer_probs[i].item()
#                         logger.info(
#                             f"{best_move:<20}{chosen_move:<20}{self.format_prob(best_move_prob):<20}{self.format_prob(chosen_move_prob):<20}"
#                         )
#         print(f"successfully terminated hook")