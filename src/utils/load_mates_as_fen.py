
import os
import chess
import chess.engine
from filelock import FileLock
STOCKFISH_PATH = "/usr/games/stockfish"
MATES_FILE = '../mates.txt'
LOCK_FILE = MATES_FILE + ".lock"
# note that there are some positions where 
# info[0]['score'].relative.is_mate() = True
# but stockfish depth is too low to see it. 

# Initialize the file lock
lock = FileLock(LOCK_FILE)
helper = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
helper.configure({"Skill Level": 20})
# Load the FEN string from the file

# # Acquire the lock before reading
# with lock, open(MATES_FILE, "r") as file:
#     fen_strings = [line.strip() for line in file.readlines()]
# # Initialize a new board with the loaded FEN string
# for pos in fen_strings:
#     board = chess.Board(pos)
#     info = helper.analyse(board, chess.engine.Limit(depth=10), multipv=5)
    
#     print(f"Stockfish says: {info[0]['score'].relative}. it is mate in {info[0]['score'].relative.mate()}")

# define dataset and dataloader object for torch

# def build_dataset_file

####################
#### this isn't going to be used now.
def build_dataset_onetime(filepath: str, newfilepath: str):
    """
    Reads FEN strings from the given input file, evaluates each position using Stockfish,
    and writes the FEN string along with the computed Q_val to a new file.

    Args:
        filepath (str): The path to the input file containing FEN strings (one per line).
        newfilepath (str): The path to the output file where the FEN and Q_val will be written.

    Q_val Calculation:
        - If the top Stockfish move is a mating sequence (Mate in X), Q_val = 1 / X.
        - Otherwise, Q_val = 0.
    """
    # Read all FEN strings from the input file
    with open(filepath, "r") as infile:
        fen_strings = [line.strip() for line in infile.readlines()]

    # Open the output file for writing
    with open(newfilepath, "w") as outfile:
        for pos in fen_strings:
            try:
                # Initialize the board from the FEN string
                board = chess.Board(pos)

                # Analyze the position with Stockfish (depth 10, multipv 5)
                info = helper.analyse(board, chess.engine.Limit(depth=10), multipv=5)

                # Get the top move evaluation
                top_score = info[0]['score'].relative

                # Determine Q_val based on the evaluation
                if top_score.is_mate():
                    mate_in = top_score.mate()
                    if mate_in > 0:  # Mate in X (for the current player)
                        Q_val = 1 / mate_in
                    else:  # Opponent has a forced mate (Mate in -X)
                        Q_val = 0
                else:
                    Q_val = 0

                # Write the FEN and Q_val to the output file
                outfile.write(f"{pos} : {Q_val}\n")

            except Exception as e:
                print(f"Error processing FEN: {pos}, Error: {str(e)}")

    # Close the Stockfish engine
    helper.quit()

    print(f"Dataset saved to {newfilepath}")



build_dataset_onetime(filepath=MATES_FILE, newfilepath="../mates_rlhf.txt")
