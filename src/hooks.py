
from typing import List, Dict, Any, Tuple, Callable
from abc import ABC, abstractmethod
import torch
from transformers import PreTrainedTokenizer, AutoModelForCausalLM
import logging
# abstract hook class


class HookManager:
    def __init__(self)->None:
        self.hooks: Dict[str, Callable[..., Any]] = {}
    
    def register_hook(self, name: str, func: Dict[str, Callable[..., Any]])->None:
        self.hooks[name] = func
        
    def execute_hook(self, name: str, *args: Any, **kwargs: Any)->None:
        if name in self.hooks:
            self.hooks[name](*args, **kwargs)

def forward_step_hook(seq: torch.Tensor,
                      loss_mask: torch.Tensor,
                      attn_mask: torch.Tensor,
                      model: AutoModelForCausalLM,
                      gradient_accumulation_steps: int,
                      )->None:
    seq_tmp = seq.detach().clone()
    seq_tmp[loss_mask == 1] = 0  # Replace target tokens in the input with pad (0)

    outputs = model(input_ids=seq_tmp, attention_mask=attn_mask, output_attentions=True)
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
    
    return loss, logits, shifted_mask, shifted_labels



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
#         tokenizer (PreTrainedTokenizer): The tokenizer used to decode token indices into human-readable text.

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
#         tokenizer: PreTrainedTokenizer
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