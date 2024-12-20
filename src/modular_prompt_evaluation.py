from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import os
from typing import List
import  torch
from torch.distributed import barrier, init_process_group, destroy_process_group
# from language_data_loader import load_dataloader
from contextlib import nullcontext
import heapq
from hooks import forward_step_hook
from language_data_loader import LlamaLoader
from yaml import CLoader as Loader
import data_loader
import torch.nn.functional as F
import config as config_lib
import chess
import chess.engine
import numpy as np
import utils
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disabling Autotokenizer parallelism so we can do distributed training.
prompt_str="You are a chess grandmaster. This is the board position in FEN notation: 5r2/p5kp/3p2p1/4p3/2Bb2PP/PPnP4/3B1P2/n2K1R2 w - - 3 29'. The legal moves are: 'd1e1', 'd1c1', 'd2c3'. Which of these is the best move? Best move: "
# Define prompts
# PROMPTS = {
#     "prompt_1": [
#         "You are a chess grandmaster. This is the board position in FEN notation: ",
#         "{fen}",
#         "The legal moves are: ",
#         "{moves}",
#         "Which of these is the best move? Best move: "
#     ]
# }
PROMPTS = {
    "prompt_1": [
        "You are a chess grandmaster. This is the board position in FEN notation: ",
        "{fen}",
        "The legal moves are: ",
        "{moves}",
        "Which of these is the best move? Best move: "
    ],
    # "prompt_2": [
    #     "[White 'Magnus Carlsen'] [Black 'Stockfish'] Board position: {fen}, Legal Moves: {moves}, Best Move: "
    # ],
    # "prompt_3": [
    #     "You are a chess grandmaster. This is the board in fen (Forsyth-Edwards notation). It is your move: ",
    #     "{fen}",
    #     "Please select the best move from this list: ",
    #     "{moves}",
    #     ".Please ONLY PLAY MOVES LISTED HERE. ANY move not in here is illegal. Best move: "
    # ],
    # "prompt_4": [
    #     "You are analyzing a competitive chess game. The current board position is represented in FEN notation: ",
    #     "{fen}",
    #     ". The legal moves available are: ",
    #     "{moves}",
    #     ". Based on the position, decide which move is the best. Best move: "
    # ],
    # "prompt_5": [
    #     "[FEN '{fen}'] Legal Moves: {moves}. Based on the current board, determine the best move from the provided options. Best Move: "
    # ],
    # "prompt_6": [
    #     "As a world-class chess engine, your task is to analyze the following board position and select the best move. Board in FEN: ",
    #     "{fen}",
    #     ". Legal moves available: ",
    #     "{moves}",
    #     ". Choose the strongest move from the list. Best move: "
    # ],
    # "prompt_7": [
    #     "You are a chess grandmaster. This is the board position in FEN notation: ",
    #     "{fen}",
    #     "The legal moves are: ",
    #     "{moves}",
    #     "Which of these is the best move? Best move: \\n1. "
    # ],
    # "prompt_8": [
    #     "You are a chess grandmaster. This is the board position in FEN notation: ",
    #     "{fen}",
    #     "The legal moves are: ",
    #     "{moves}",
    #     "Which of these is the best move? Best move: \n1. "
    # ],
}
def edit_distance(s1, s2):
    return sum(a != b for a, b in zip(s1, s2)) + abs(len(s1) - len(s2))

def beam_search(logits: torch.Tensor, loss_mask: torch.Tensor, k: int = 5)->None: # pass in loss_mask[:, 1:] because we have already shifted logits
    
    # step 1. for all indices we care about predicting, get the top k most likely tokens (those are in batch_indices)
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k)
    batch_logits, batch_indices = [], []
    for i in range(logits.shape[0]):
        batch_logits.append(values[i][loss_mask[i]])
        batch_indices.append(indices[i][loss_mask[i]])
    
    return batch_logits, batch_indices
        
def getBeamsFromMatrix(probs: torch.Tensor, indices: torch.Tensor, num_beams:int=5)->List[str]:
    def getNeighbors(moves: List[int])->List[List[int]]:
        neighbors =[]
        for dex in range(len(moves)):
            moves_new = moves.copy()
            if moves_new[dex]+1<len(moves):
                moves_new[dex]+=1
            neighbors.append(moves_new)
        return neighbors
    
    ret = []
    pq=[]
    visited = []
    move_probs = []
    most_likely_move_prob = torch.prod(probs.gather(1, torch.tensor([0]*num_beams).unsqueeze(1)))
    heapq.heappush(pq, (-most_likely_move_prob, [0]*num_beams)) # taking negative because pq's sort in ascending order by default
    for _ in range(num_beams):
        move_prob, best_move = heapq.heappop(pq)
        visited.append(best_move)

        neighbors = getNeighbors(moves=best_move)
        # add each neighbor to queue
        for neighbor in neighbors: 
            if neighbor not in visited:
                move_prob = torch.prod(probs.gather(1, torch.tensor(neighbor).unsqueeze(1)))
                heapq.heappush(pq, (-move_prob, neighbor))
        # add our next best path to the list we're returning
        move_tokens = indices.gather(1, torch.tensor(best_move, dtype=torch.int64).unsqueeze(1))
        ret.append(move_tokens.T[0][:-1])
        move_probs.append(move_prob)
    
    return ret, move_probs
def prompt_evaluation(
    model: AutoModelForCausalLM,
    data_iter: LlamaLoader,
    tokenizer,
    device: str,
    model_name: str,
    num_batches: int = 2,
    strip_away_characters: bool = False
) -> None:
    print(f"Entered function")
    # Calculate exhaustive list of all valid SAN moves
    file_path = "prompt_statistics.txt"
    all_moves = utils._compute_all_possible_actions()
    valid_san_moves = set(all_moves[0].keys())

    # Initialize stockfish helper to annotate how good our moves are
    stockfish_path = "/usr/games/stockfish"
    board = chess.Board()



    # Iterate over all prompt templates
    for prompt_name, prompt_template in PROMPTS.items():
        # Dictionary to store statistics with descriptive keys
        statistics = {
            "prompt_name": prompt_name,
            "model_name": model_name,
            "Generous? ": strip_away_characters,
            "total_prompts": 0,
            "valid_san_moves": 0,
            "legal_moves": 0,
            "first_move_chosen": 0,
            "best_moves": 0,
            "offByOneOrZero": 0,
            "offByOneOrZeroOrLegalMove": 0
        }
        print(f"Prompt template is {prompt_template}")

        # Iterate over all batches we've decided to load
        for i in range(num_batches):
            print(f"starting batch {i}")
            seq, attn_mask, loss_mask, fen_batch, best_move_batch = next(data_iter)
            with torch.no_grad():
                seq=seq.to(model.device)
                attn_mask = attn_mask.to(model.device)
                loss_mask = loss_mask.to(model.device)
                loss, logits, shifted_mask, shifted_labels, outputs, seq_tmp, attn_mask, sequence_accuracy, mean_correct_prob, mean_chosen_prob = forward_step_hook(seq=seq,
                                                                            loss_mask=loss_mask,
                                                                            attn_mask=attn_mask,
                                                                            model=model,
                                                                            gradient_accumulation_steps=1, # don't care about this input here.
                                                                            method=config["attn_method"])
                logits_beam, indices_beam = beam_search(logits=logits, loss_mask=loss_mask[:, 1:])
                token_indices = torch.nonzero(shifted_mask)  # Indices of tokens we care about
                row_indices = token_indices[:, 0]
                col_indices = token_indices[:, 1]

                unique_rows = torch.unique(row_indices)
                full_predicted_tokens = torch.argmax(logits, dim=-1)  # (batch, seq_len)
                predicted_tokens = full_predicted_tokens[row_indices, col_indices]
                grouped_predicted_tokens = []
                for row in unique_rows:
                    current_row_mask = (row_indices == row)
                    row_preds = predicted_tokens[current_row_mask] # predicted tokens for this row
                    grouped_predicted_tokens.append(tokenizer.decode(row_preds[:-1]))
                    # best_move_str = tokenizer.decode(row_preds[:-1])
            i = 0
            for fen, best_move in zip(fen_batch, best_move_batch):
                output = grouped_predicted_tokens[i]
                tokens, move_probs = getBeamsFromMatrix(probs=logits_beam[i], indices=indices_beam[i], num_beams=5)
                candidates = [tokenizer.decode(move) for move in tokens]
                board.set_fen(fen)
                legal_moves = [str(m) for m in board.legal_moves]
                # prompt_str = "".join(prompt_template).format(fen=fen, moves=", ".join(legal_moves))
                # inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
                # num_tokens = inputs["input_ids"].shape[1]

                # # Generate candidate moves
                # if strip_away_characters:
                #     candidates = output.split(" ")
                # else:
                #     candidates = getBeamsFromMatrix(probs=logits_beam[i], indices=indices_beam[i], num_beams=5)

                # Update statistics based on conditions
                if any(candidate.strip() in valid_san_moves for candidate in candidates):
                    statistics["valid_san_moves"] += 1
                # else:
                #     print(f"@@@@@@@@@@@@@@@ {output} is not a valid san move")
                if any(candidate.strip() in legal_moves for candidate in candidates):
                    statistics["legal_moves"] += 1
                if any(candidate.strip() == best_move for candidate in candidates):
                    statistics["best_moves"] += 1
                if any(candidate.strip() == legal_moves[0] for candidate in candidates):
                    statistics["first_move_chosen"] += 1
                if edit_distance(output, best_move) < 2:
                    statistics["offByOneOrZero"] += 1
                if edit_distance(output, best_move) < 2 or any(candidate.strip() in legal_moves for candidate in candidates):
                    statistics["offByOneOrZeroOrLegalMove"] += 1
                # if any(candidate.strip())
                statistics["total_prompts"] += 1
                print(f"chosen move: {output}, best move: {best_move}, edit distance: {edit_distance(output, best_move)}, candidates: {candidates}, probs: {move_probs}")
                # print(
                #     f"Best move: {best_move}, prediction: {output}, legal move? {output in legal_moves}"
                # )
                i+=1

        # Write statistics to a YAML file
        with open(file_path, "a") as f:
            for key, value in statistics.items():
                f.write(f"{key}: {value}\n")
            f.write("-" * 40 + "\n")
            f.close()
        print(f"Statistics saved to {file_path}")

if __name__ == "__main__":
    # config_file = "/workspace/searchless_chess/src/config_llama.yaml"
    # model_load_path = "./Llama/llama-3.2-1B"
    # model_name = "Llama3.2-1B"
    config_file = "/workspace/searchless_chess/src/config_pythia.yaml"
    model_load_path = "/workspace/searchless_chess/src/pythia/ckpts_ft/ckpt68000"
    model_name = "pythia-160m"
    with open(config_file, "r") as stream:
        config = yaml.load(stream=stream, Loader=Loader)

    ## hacky test code to make sure I can load the models I've saved in train_pythia
    ddp = int(os.environ.get('RANK', -1)) != -1  # Is this a DDP run?
    if ddp:
        
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])

        # Set DDP-specific config parameters
        config["ddp_world_size"] = ddp_world_size
        config["ddp_local_rank"] = ddp_local_rank
        config["ddp_rank"] = ddp_rank
    else:
        device = config['device']
        config["ddp_world_size"] = 0
        config["ddp_local_rank"] = 0
        config["ddp_rank"] = 0

    # config_path = "/workspace/searchless_chess/src/pythia/ckpts/ckpt500"
    device = 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_load_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    data_iter = LlamaLoader(training_config=config, tokenizer=tokenizer, split="train")
    if ddp:
        init_process_group(backend=config['backend'])
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}['bfloat16']
    ctx = nullcontext() if config['device_type'] == 'cpu' else torch.amp.autocast(device_type=config['device_type'], dtype=ptdtype)
    with ctx if ctx else torch.no_grad():  # Use ctx if provided, else default no_grad context
        prompt_evaluation(model, data_iter,tokenizer=tokenizer, model_name=model_name, device=device, strip_away_characters=True, num_batches=20)
    if ddp:
        destroy_process_group()