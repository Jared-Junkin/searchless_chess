from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
from language_data_loader import load_dataloader
from yaml import CLoader as Loader
import data_loader
import config as config_lib
import chess
import chess.engine
import numpy as np
import utils
prompt_str="You are a chess grandmaster. This is the board position in FEN notation: 5r2/p5kp/3p2p1/4p3/2Bb2PP/PPnP4/3B1P2/n2K1R2 w - - 3 29'. The legal moves are: 'd1e1', 'd1c1', 'd2c3'. Which of these is the best move? Best move: "
# Define prompts
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

def prompt_evaluation(model, tokenizer) -> None:
    # calculate exhaustive list of all valid san moves
    file_path = "prompt_statistics.txt"
    all_moves = utils._compute_all_possible_actions()
    valid_san_moves = set(all_moves[0].keys())
    
    # initialize stockfish helper to annotate how good our moves are.
    stockfish_path = "/usr/games/stockfish"
    helper = chess.engine.SimpleEngine.popen_uci(stockfish_path) # probably will need to make sure my helper has the same color as my agent
    
    num_batches = 2
    policy = "prompting"
    num_return_buckets = 128
    train_config = config_lib.TrainConfig(
        learning_rate=1e-4,
        data=config_lib.DataConfig(
            batch_size=256,
            shuffle=True,
            worker_count=0,  # 0 disables multiprocessing.
            num_return_buckets=num_return_buckets,
            policy=policy,
            split='train',
        ),
        log_frequency=1,
        num_steps=20,
        ckpt_frequency=5,
        save_frequency=10,
    )
    data_iter = data_loader.build_data_loader(config=train_config.data).__iter__()
    board = chess.Board()
    
    # iterate over all prompt templates
    for prompt_name, prompt_template in PROMPTS.items():
        print(f"prompt template is {prompt_template}")
        prompt_statistics = np.zeros(shape=(6,1))

        # iterate over all batches we've decided to load
        for _ in range(num_batches):
            fen_batch, best_move_batch = next(data_iter)
            
            # iterate over all game states in batch
            for fen, best_move in zip(fen_batch, best_move_batch):
                
                
                
                board.set_fen(fen)
                                
                # get legal moves for AI prompt
                legal_moves = [str(m) for m in board.legal_moves]
                
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

                # Convert best_moves_san to UCI
                best_moves_uci = [(board.parse_san(move).uci(), score) for move, score in best_moves_san] # -40 centipawn score = -0.4 in chess.com evaluation (which is in pawn score)
                best_moves_uci_only = [board.parse_san(move).uci() for move, score in best_moves_san] # same as line above except this just returns move and not evaluation

                
                # Combine template parts into a single prompt string
                prompt = "".join(prompt_template).format(
                    fen=fen,
                    moves=", ".join(legal_moves),
                    best_move=best_move
                )

                # Tokenize the prompt and pass it through the model
                inputs = tokenizer(prompt, return_tensors="pt")
                num_tokens = inputs["input_ids"].shape[1]
                outputs = model.generate(**inputs, max_length=num_tokens + 7)
                generated_tokens = outputs[0][num_tokens:]  # skip the prompt tokens and extract only the generated tokens.s
                decoded_output = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
                
                # Print prompt and output for debugging purposes
                print(f"Prompt ({prompt_name}): {prompt}")
                print(f"Model output: {decoded_output}")
                print("-" * 80)
                
                candidates = decoded_output.split(" ")
                if any(candidate.strip() in valid_san_moves for candidate in candidates): # outputted a valid SAN move
                    prompt_statistics[0] += 1 
                if any(candidate.strip() in legal_moves for candidate in candidates): # AI has outputted valid SAN that is also a legal move in the current board configuration
                    prompt_statistics[1] += 1
                    legal_move = next(candidate.strip() for candidate in candidates if candidate.strip() in legal_moves)
                    move_position = best_moves_uci_only.index(legal_move)
                    centipawn_delta = abs(best_moves_uci[move_position][1] - best_moves_uci[0][1]) # difference in centipawn evaluation betyween best move and move chosen. should go to 0 with perfect play. can use abs because impossible to do better than best move in behavioral cloning.
                    prompt_statistics[4] += centipawn_delta                           
                else:
                    prompt_statistics[4] += 1000
                if any(candidate.strip() == best_move for candidate in candidates): # AI has outputted the best move correctly
                    prompt_statistics[2] += 1  
                if any(candidate.strip() == legal_moves[0] for candidate in candidates):    # AI has chosen the first legal move that was listed
                    prompt_statistics[3] += 1
                prompt_statistics[5]+=1 # 
        # ai likes to put numbers like 1. nxe5 in front of the correct move (nxe5). strip away these numbers
        # just say the agents move is the longest string in " ".split(outputs[0])
        # calculate for each prompt % of responses meeting the three  criteria:
            # 1. it is valid san. do this exact thing. valid_san = move in set(utils)
            # 2. is it a legal move given the current board configuration
            # 3. is it just printing the first legal move every time? 
            # 4. is it printing the best move? 
            # 5. what is the average position in the best moves matrix? 
        # Normalize statistics by dividing entries 0:n-1 by the last element
        if prompt_statistics[5] != 0:  # Guard to ensure no division by zero
            prompt_statistics[:-1] /= prompt_statistics[5]

        # Write the statistics to a file
        with open(file_path, "a") as f:
            f.write(f"Prompt: {prompt_name}\n")
            f.write("Statistics:\n")
            for i, stat in enumerate(prompt_statistics.flatten()):
                f.write(f"Statistic {i}: {stat}\n")
            f.write("-" * 40 + "\n")
        f.close()
        return
if __name__ == "__main__":
    # config_file = "/workspace/searchless_chess/src/config_pythia.yaml"
    # with open(config_file, "r") as stream:
    #     config = yaml.load(stream=stream, Loader=Loader)
    # model = AutoModelForCausalLM.from_pretrained(config["model_load_dir"])
    # tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_load_dir"])
    # prompt_evaluation(model, tokenizer)


    ## hacky test code to make sure I can load the models I've saved in train_pythia
    config_path = "/workspace/searchless_chess/src/pythia/ckpts/ckpt2000"
    model = AutoModelForCausalLM.from_pretrained(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config_path)
    prompt_evaluation(model, tokenizer)