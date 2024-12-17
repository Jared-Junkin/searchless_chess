from transformers import GPTNeoXForCausalLM, AutoTokenizer

# Load model and tokenizer
local_dir = "./pythia-160m/"
# local_dir = "./ckpts/ckpt6000"
model = GPTNeoXForCausalLM.from_pretrained(local_dir)
tokenizer = AutoTokenizer.from_pretrained(local_dir)

# FEN and moves data
fen = '5r2/p5kp/3p2p1/4p3/2Bb2PP/PPnP4/3B1P2/n2K1R2 w - - 3 29'
moves = ['d1e1', 'd1c1', 'd2c3']

# Define prompts
prompts = {
    "prompt_1": [
        "You are a chess grandmaster. This is the board position in FEN notation: ",
        "{fen}",
        "The legal moves are: ",
        "{moves}",
        "Which of these is the best move? Best move: "
    ],
    # "prompt_2": [
    #     f"[White 'Magnus Carlsen'] [Black 'Stockfish'] Board position: {fen}, Legal Moves: {moves}, Best Move: "
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
    # ]
}

# Iterate over prompts and perform inference
for key, prompt_template in prompts.items():
    # Join the prompt and substitute variables
    prompt_str = "".join(prompt_template).format(fen=fen, moves=", ".join(moves))
    
    # Tokenize and prepare inputs
    inputs = tokenizer(prompt_str, return_tensors="pt")
    num_tokens = inputs["input_ids"].shape[1]
    
    # Perform generation
    tokens = model.generate(**inputs, max_length=num_tokens + 7)
    output = tokenizer.decode(tokens[0])
    
    # Print the result
    print(f"Evaluating {key}. Response is {output}\n\n\n")
