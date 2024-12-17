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
    "prompt_2": [
        "[White 'Magnus Carlsen'] [Black 'Stockfish'] Board position: {fen}, Legal Moves: {moves}, Best Move: "
    ],
    "prompt_3": [
        "You are a chess grandmaster. This is the board in fen (Forsyth-Edwards notation). It is your move: ",
        "{fen}",
        "Please select the best move from this list: ",
        "{moves}",
        ".Please ONLY PLAY MOVES LISTED HERE. ANY move not in here is illegal. Best move: "
    ],
    "prompt_4": [
        "You are analyzing a competitive chess game. The current board position is represented in FEN notation: ",
        "{fen}",
        ". The legal moves available are: ",
        "{moves}",
        ". Based on the position, decide which move is the best. Best move: "
    ],
    "prompt_5": [
        "[FEN '{fen}'] Legal Moves: {moves}. Based on the current board, determine the best move from the provided options. Best Move: "
    ],
    "prompt_6": [
        "As a world-class chess engine, your task is to analyze the following board position and select the best move. Board in FEN: ",
        "{fen}",
        ". Legal moves available: ",
        "{moves}",
        ". Choose the strongest move from the list. Best move: "
    ],
    "prompt_7": [
        "You are a chess grandmaster. This is the board position in FEN notation: ",
        "{fen}",
        "The legal moves are: ",
        "{moves}",
        "Which of these is the best move? Best move: \\n1. "
    ],
    "prompt_8": [
        "You are a chess grandmaster. This is the board position in FEN notation: ",
        "{fen}",
        "The legal moves are: ",
        "{moves}",
        "Which of these is the best move? Best move: \n1. "
    ],
}

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
        }
        print(f"Prompt template is {prompt_template}")

        # Iterate over all batches we've decided to load
        for i in range(num_batches):
            print(f"starting batch {i}")
            seq, attn_mask, loss_mask, fen_batch, best_move_batch = next(data_iter)
            for fen, best_move in zip(fen_batch, best_move_batch):
                board.set_fen(fen)
                legal_moves = [str(m) for m in board.legal_moves]
                prompt_str = "".join(prompt_template).format(fen=fen, moves=", ".join(legal_moves))
                inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
                num_tokens = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    tokens = model.generate(**inputs, max_length=num_tokens + 7)
                output = tokenizer.decode(tokens[0, -7:], skip_special_tokens=True)

                # Generate candidate moves
                if strip_away_characters:
                    candidates = output.split(" ")
                else:
                    candidates = [output]

                # Update statistics based on conditions
                if any(candidate.strip() in valid_san_moves for candidate in candidates):
                    statistics["valid_san_moves"] += 1
                if any(candidate.strip() in legal_moves for candidate in candidates):
                    statistics["legal_moves"] += 1
                if any(candidate.strip() == best_move for candidate in candidates):
                    statistics["best_moves"] += 1
                if any(candidate.strip() == legal_moves[0] for candidate in candidates):
                    statistics["first_move_chosen"] += 1
                statistics["total_prompts"] += 1

                # print(
                #     f"Best move: {best_move}, prediction: {output}, legal move? {output in legal_moves}"
                # )

        # Write statistics to a YAML file
        with open(file_path, "a") as f:
            for key, value in statistics.items():
                f.write(f"{key}: {value}\n")
            f.write("-" * 40 + "\n")
            f.close()
        print(f"Statistics saved to {file_path}")

if __name__ == "__main__":
    config_file = "/workspace/searchless_chess/src/config_llama.yaml"
    model_load_path = "./Llama/llama-3.2-1B"
    model_name = "Llama3.2-1B"
    # config_file = "/workspace/searchless_chess/src/config_pythia.yaml"
    # model_load_path = "/workspace/searchless_chess/src/pythia/pythia-160m"
    # model_name = "pythia-160m"
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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