from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import os
import  torch
from torch.distributed import barrier, init_process_group, destroy_process_group
# from language_data_loader import load_dataloader
from contextlib import nullcontext
from language_data_loader import LlamaLoader
from yaml import CLoader as Loader
import data_loader
import config as config_lib
import chess
import chess.engine
import numpy as np
import utils
from hooks import generate_best_move_hook

os.environ["TOKENIZERS_PARALLELISM"] = "false" # disabling Autotokenizer parallelism so we can do distributed training.
prompt_str="You are a chess grandmaster. This is the board position in FEN notation: 5r2/p5kp/3p2p1/4p3/2Bb2PP/PPnP4/3B1P2/n2K1R2 w - - 3 29'. The legal moves are: 'd1e1', 'd1c1', 'd2c3'. Which of these is the best move? Best move: "
# Define prompts
PROMPTS = {
    "prompt_1": [
        "You are a chess grandmaster. This is the board position in FEN notation: ",
        "{fen}",
        "The legal moves are: ",
        "{moves}",
        "Which of these is the best move? Best move: "
    ]
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

@torch.no_grad()
def prompt_evaluation(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, data_iter: LlamaLoader, num_batches: int=2) -> None:
    # calculate exhaustive list of all valid san moves
    file_path = "prompt_statistics.txt"
    all_moves = utils._compute_all_possible_actions()
    valid_san_moves = set(all_moves[0].keys())
    
    # initialize stockfish helper to annotate how good our moves are.
    stockfish_path = "/usr/games/stockfish"
    helper = chess.engine.SimpleEngine.popen_uci(stockfish_path) # probably will need to make sure my helper has the same color as my agent
    board = chess.Board()
    
    # iterate over all prompt templates
    for prompt_name, prompt_template in PROMPTS.items():
        print(f"prompt template is {prompt_template}")
        batch_statistics = {
            "num_moves": 0,
            "Valid_San": 0,
            "Legal_Move": 0,
            "Best_Move": 0,
            "First_move": 0,
            "Best_move_is_first_move": 0
        }
                


        # iterate over all batches we've decided to load
        for _ in range(num_batches):

            try:
                seq, attn_mask, loss_mask, fen_batch, best_move_batch = next(data_iter)
            except StopIteration:
                print(f"Out of data")
                batch_statistics["Best_Move"]/=batch_statistics["num_moves"]
                batch_statistics["Valid_San"]/=batch_statistics["num_moves"]
                batch_statistics["First_move"]/=batch_statistics["num_moves"]
                batch_statistics["Best_move_is_first_move"]/=batch_statistics["num_moves"]
                batch_statistics["Legal_Move"]/=batch_statistics["num_moves"]
                with open(file_path, "a") as f:
                    f.write(f"Prompt: {prompt_name}\n")
                    f.write(f"Statistics:{batch_statistics}")
                f.close()
                helper.close()
                return

            # seq[attn_mask==0] = 0 # remove target tokens ahead of generation
            # best_move = generate_best_move_hook(input_ids=seq, model=model, tokenizer=tokenizer)

            i=0
            for fen, best_move in zip(fen_batch, best_move_batch):
                board.set_fen(fen)
                                
                # get legal moves for AI prompt
                # seq, attn_mask, loss_mask, fen_batch, best_move_batch = next(data_iter)
                legal_moves = [str(m) for m in board.legal_moves]
                

                seq[i][attn_mask[i]==0] = 0 # remove target tokens ahead of generation
                sample = seq[i][seq[i]!=0]
                prediction = generate_best_move_hook(input_ids=sample.unsqueeze(0), model=model, tokenizer=tokenizer)
                
                
                # get best moves so we can evaluate how good our agent is
                # Get best moves so we can evaluate how good our agent is
                best_moves_san = []
                info = helper.analyse(board, chess.engine.Limit(depth=10), multipv=len(legal_moves))  # Get Stockfish's analysis for all legal moves
                perspective = chess.WHITE if board.turn else chess.BLACK  # Determine if this is a black position or a white position.

                for item in info:
                    # Check for invalid scores
                    if item['score'] is None or item['score'].pov(perspective) is None:
                        print(f"Skipping item with invalid score: {item}")
                        continue  # Skip this item if the score is invalid
                    
                    # Convert the first move in the principal variation to SAN
                    san_move = board.san(item['pv'][0])

                    # Assess the score
                    score = item['score'].pov(perspective)
                    if score.is_mate():  # Check if the position is a mate
                        mate_value = score.mate()  # Get the mate distance (+i or -i)
                        if mate_value > 0:  # Winning mate
                            numeric_score = 0
                        else:  # Losing mate
                            numeric_score = 1000
                    else:  # Centipawn score
                        numeric_score = score.score()

                    # Append the SAN move and numeric score
                    best_moves_san.append((san_move, numeric_score))

                candidates = [prediction]
                
                
            

                # candidates = decoded_output.split(" ")
                batch_statistics["num_moves"]+=1
                if any(candidate.strip() in valid_san_moves for candidate in candidates): # outputted a valid SAN move
                    batch_statistics["Valid_San"]+=1
                if any(candidate.strip() in legal_moves for candidate in candidates): # AI has outputted valid SAN that is also a legal move in the current board configuration
                    batch_statistics["Legal_Move"] += 1
                   
                if any(candidate.strip() == best_move for candidate in candidates): # AI has outputted the best move correctly
                    batch_statistics["Best_Move"] += 1  
                if any(candidate.strip() == legal_moves[0] for candidate in candidates):    # AI has chosen the first legal move that was listed
                    batch_statistics["First_move"] += 1
                if legal_moves[0] == best_move:
                    batch_statistics["Best_move_is_first_move"]+=1
                i+=1
                print(f"done sample {i}. Player's move is {candidates[0]}.  best move is {best_move}")

            print("done batch")
        batch_statistics["Best_Move"]/=batch_statistics["num_moves"]
        batch_statistics["Valid_San"]/=batch_statistics["num_moves"]
        batch_statistics["First_move"]/=batch_statistics["num_moves"]
        batch_statistics["Best_move_is_first_move"]/=batch_statistics["num_moves"]
        batch_statistics["Legal_Move"]/=batch_statistics["num_moves"]
        with open(file_path, "a") as f:
            f.write(f"Prompt: {prompt_name}\n")
            f.write(f"Statistics:{batch_statistics}")
        f.close()
        helper.close()
        return
if __name__ == "__main__":
    config_file = "/workspace/searchless_chess/src/config_pythia.yaml"
    with open(config_file, "r") as stream:
        config = yaml.load(stream=stream, Loader=Loader)
    # model = AutoModelForCausalLM.from_pretrained(config["model_load_dir"])
    # tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_load_dir"])
    # prompt_evaluation(model, tokenizer)


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

    config_path = "/workspace/searchless_chess/src/pythia/ckpts/ckpt52000"
    model = AutoModelForCausalLM.from_pretrained(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config_path)
    data_iter = LlamaLoader(config=config, tokenizer=tokenizer, split="test", repeat=False)
    if ddp:
        init_process_group(backend=config['backend'])
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}['bfloat16']
    ctx = nullcontext() if config['device_type'] == 'cpu' else torch.amp.autocast(device_type=config['device_type'], dtype=ptdtype)
    with ctx if ctx else torch.no_grad():  # Use ctx if provided, else default no_grad context
        prompt_evaluation(model, tokenizer, data_iter, num_batches=10)
    if ddp:
        destroy_process_group()