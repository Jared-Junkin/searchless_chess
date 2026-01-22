import chess
import chess.engine
import os
import csv
import random
import time
import platform
import random


# NOTE: LLAMA AND NANOGPT ARE EXPERIMENTAL PLAYERS that most people won't need to use
# They are commented by default to avoid unnecessary dependencies such as pytorch.
# from llama_module import BaseLlamaPlayer, LocalLlamaPlayer, LocalLoraLlamaPlayer
from jared_models.nanoGPT  import NanoGptPlayer
# import gpt_query

from typing import Optional, Tuple
from dataclasses import dataclass

from jared_data_loader import InferenceTimeBehavioralCloning
from filelock import FileLock

@dataclass
class LegalMoveResponse:
    move_san: Optional[str] = None
    move_uci: Optional[chess.Move] = None
    attempts: int = 0
    is_resignation: bool = False
    is_illegal_move: bool = False


# Define base Player class
class Player:
    def get_move(self, board: chess.Board, game_state: str, temperature: float) -> str:
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError



# inserted this. still not ready obviously.
class humanPlayer(Player):
    def __init__(self):
        super().__init__()
        self.model_name = "Homo Sapiens"
    def get_move(self, board, game_state, temperature):
        print("\n\n")
        print("______________________________________________________")
        print(board)
        print("______________________________________________________")
        print("\n\n")
        max_attempts = 10
        for attempt in range(max_attempts):
            move_san = input("Make a move: ")
            try: 
                return move_san
            except:
                print(f"Move {move_san} is not legal. Please try again ({max_attempts - attempt} remaining.)")
        print(f"You have failed to enter a legal move {max_attempts} times. Resigning.")
        return LegalMoveResponse(
                    move_san=None,
                    move_uci=None,
                    attempts=max_attempts,
                    is_resignation=True,
                )
    def get_config(self):
        return {"model": self.model_name, "skill_level": 1, "play_time":1000}

class StockfishPlayer(Player):
    @staticmethod
    def get_stockfish_path() -> str:
        """
        Determines the operating system and returns the appropriate path for Stockfish.

        Returns:
            str: Path to the Stockfish executable based on the operating system.
        """
        if platform.system() == "Linux":
            return "/home/ext-jjunkin/data_snatha11/searchless_chess/src/engines/stockfish"
        elif platform.system() == "Darwin":  # Darwin is the system name for macOS
            return "stockfish"
        elif platform.system() == "Windows":
            return (
                r"C:\Users\adamk\Documents\Stockfish\stockfish-windows-x86-64-avx2.exe"
            )
        else:
            raise OSError("Unsupported operating system")

    def __init__(self, skill_level: int, play_time: float):
        self._skill_level = skill_level
        self._play_time = play_time
        # If getting started, you need to run brew install stockfish
        stockfish_path = StockfishPlayer.get_stockfish_path()
        self._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def get_move(
        self, board: chess.Board, game_state: str, temperature: float
    ) -> Optional[str]:
        if self._skill_level == -2:
            legal_moves = list(board.legal_moves)
            random_move = random.choice(legal_moves)
            return board.san(random_move)
        elif self._skill_level < 0:
            self._engine.configure({"Skill Level": 0})
            result = self._engine.play(
                board, chess.engine.Limit(time=1e-8, depth=1, nodes=1)
            )

        else:
            self._engine.configure({"Skill Level": self._skill_level})
            result = self._engine.play(board, chess.engine.Limit(time=self._play_time))
        if result.move is None:
            return None
        return board.san(result.move)

    def get_config(self) -> dict:
        return {"skill_level": self._skill_level, "play_time": self._play_time}

    def close(self):
        self._engine.quit()




def get_move_from_gpt_response(response: Optional[str]) -> Optional[str]:
    if response is None:
        return None

    # Parse the response to get only the first move
    moves = response.split()
    first_move = moves[0] if moves else None
    print(f"all moves: {moves}")
    print(f"chosen move: {first_move}")
    return first_move


def record_results(
    board: chess.Board,
    player_one: Player,
    player_two: Player,
    game_state: str,
    player_one_illegal_moves: int,
    player_two_illegal_moves: int,
    player_one_legal_moves: int,
    player_two_legal_moves: int,
    total_time: float,
    player_one_resignation: bool,
    player_two_resignation: bool,
    player_one_failed_to_find_legal_move: bool,
    player_two_failed_to_find_legal_move: bool,
    total_moves: int,
    illegal_moves: int,
):
    unique_game_id = generate_unique_game_id()

    (
        player_one_title,
        player_two_title,
        player_one_time,
        player_two_time,
    ) = get_player_titles_and_time(player_one, player_two)

    if player_one_resignation or player_one_failed_to_find_legal_move:
        result = "0-1"
        player_one_score = 0
        player_two_score = 1
    elif player_two_resignation or player_two_failed_to_find_legal_move:
        result = "1-0"
        player_one_score = 1
        player_two_score = 0
    else:
        result = board.result()
        # Hmmm.... debating this one. Annoying if I leave it running and it fails here for some reason, probably involving some
        # resignation / failed move situation I didn't think of
        # -1e10 at least ensures it doesn't fail silently
        if "-" in result:
            player_one_score = result.split("-")[0]
            player_two_score = result.split("-")[1]
        elif result == "*":  # Draw due to hitting max moves
            player_one_score = 1 / 2
            player_two_score = 1 / 2
        else:
            player_one_score = -1e10
            player_two_score = -1e10

    info_dict = {
        "game_id": unique_game_id,
        "transcript": game_state,
        "result": result,
        "player_one": player_one_title,
        "player_two": player_two_title,
        "player_one_time": player_one_time,
        "player_two_time": player_two_time,
        "player_one_score": player_one_score,
        "player_two_score": player_two_score,
        "player_one_illegal_moves": player_one_illegal_moves,
        "player_two_illegal_moves": player_two_illegal_moves,
        "player_one_legal_moves": player_one_legal_moves,
        "player_two_legal_moves": player_two_legal_moves,
        "player_one_resignation": player_one_resignation,
        "player_two_resignation": player_two_resignation,
        "player_one_failed_to_find_legal_move": player_one_failed_to_find_legal_move,
        "player_two_failed_to_find_legal_move": player_two_failed_to_find_legal_move,
        "game_title": f"{player_one_title} vs. {player_two_title}",
        "number_of_moves": board.fullmove_number,
        "time_taken": total_time,
        "total_moves": total_moves,
        "illegal_moves": illegal_moves,
    }

    if RUN_FOR_ANALYSIS:
        csv_file_path = (
            f"logs/{player_one_recording_name}_vs_{player_two_recording_name}"
        )
        csv_file_path = csv_file_path.replace(
            ".", "_"
        )  # filenames can't have periods in them. Useful for e.g. gpt-3.5 models
        csv_file_path += ".csv"
    else:
        csv_file_path = recording_file

    # Determine if we need to write headers (in case the file doesn't exist yet)
    write_headers = not os.path.exists(csv_file_path)

    # Append the results to the CSV file
    with open(csv_file_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=info_dict.keys())
        if write_headers:
            writer.writeheader()
        writer.writerow(info_dict)

    with open("game.txt", "w") as f:
        f.write(game_state)


def generate_unique_game_id() -> str:
    timestamp = int(time.time())
    random_num = random.randint(1000, 9999)  # 4-digit random number
    return f"{timestamp}-{random_num}"


def get_player_titles_and_time(
    player_one: Player, player_two: Player
) -> Tuple[str, str, Optional[float], Optional[float]]:
    player_one_config = player_one.get_config()
    player_two_config = player_two.get_config()

    # For player one
    if "model" in player_one_config:
        player_one_title = player_one_config["model"]
        player_one_time = None
    else:
        player_one_title = f"Stockfish {player_one_config['skill_level']}"
        player_one_time = player_one_config["play_time"]

    # For player two
    if "model" in player_two_config:
        player_two_title = player_two_config["model"]
        player_two_time = None
    else:
        player_two_title = f"Stockfish {player_two_config['skill_level']}"
        player_two_time = player_two_config["play_time"]

    return (player_one_title, player_two_title, player_one_time, player_two_time)


def initialize_game_with_opening(
    game_state: str, board: chess.Board
) -> Tuple[str, chess.Board]:
    with open("openings.csv", "r") as file:
        lines = file.readlines()[1:]  # Skip header
    moves_string = random.choice(lines)
    game_state += moves_string
    # Splitting the moves string on spaces
    tokens = moves_string.split()

    for token in tokens:
        # If the token contains a period, it's a move number + move combination
        if "." in token:
            move = token.split(".")[-1]  # Take the move part after the period
        else:
            move = token

        board.push_san(move)
    return game_state, board


# Return is (move_san, move_uci, attempts, is_resignation, is_illegal_move)
def get_legal_move(
    player: Player,
    board: chess.Board,
    game_state: str,
    player_one: bool,
    max_attempts: int = 5,
) -> LegalMoveResponse:
    """Request a move from the player and ensure it's legal."""
    move_san = None
    move_uci = None

    for attempt in range(max_attempts):
        move_san = player.get_move(
            board, game_state, min(((attempt / max_attempts) * 1) + 0.001, 0.5)
        )

        # Sometimes when GPT thinks it's the end of the game, it will just output the result
        # Like "1-0". If so, this really isn't an illegal move, so we'll add a check for that.
        if move_san is not None:
            if move_san == "1-0" or move_san == "0-1" or move_san == "1/2-1/2":
                print(f"{move_san}, player has resigned")
                return LegalMoveResponse(
                    move_san=None,
                    move_uci=None,
                    attempts=attempt,
                    is_resignation=True,
                )

        try:
            move_uci = board.parse_san(move_san)
        except Exception as e:
            print(f"Error parsing move {move_san}: {e}")
            # check if player is gpt-3.5-turbo-instruct
            # only recording errors for gpt-3.5-turbo-instruct because it's errors are so rare
            if player.get_config()["model"] == "gpt-3.5-turbo-instruct":
                with open("gpt-3.5-turbo-instruct-illegal-moves.txt", "a") as f:
                    f.write(f"{game_state}\n{move_san}\n")
            continue

        if move_uci in board.legal_moves:
            if not move_san.startswith(" "):
                move_san = " " + move_san
            return LegalMoveResponse(move_san, move_uci, attempt)
        print(f"Illegal move: {move_san}")

    # If we reach here, the player has made illegal moves for all attempts.
    print(f"{player} provided illegal moves for {max_attempts} attempts.")
    return LegalMoveResponse(
        move_san=None, move_uci=None, attempts=max_attempts, is_illegal_move=True
    )


def play_turn(
    player: Player, board: chess.Board, game_state: str, player_one: bool
) -> Tuple[str, bool, bool, int]:
    result = get_legal_move(player, board, game_state, player_one, 5)
    illegal_moves = result.attempts
    move_san = result.move_san
    move_uci = result.move_uci
    resignation = result.is_resignation
    failed_to_find_legal_move = result.is_illegal_move

    if resignation:
        print(f"{player} resigned with result: {board.result()}")
    elif failed_to_find_legal_move:
        print(f"Game over: 5 consecutive illegal moves from {player}")
    elif move_san is None or move_uci is None:
        print(f"Game over: {player} failed to find a legal move")
    else:
        board.push(move_uci)
        game_state += move_san
        print(move_san, end=" ")

    return game_state, resignation, failed_to_find_legal_move, illegal_moves


def initialize_game_with_random_moves(
    board: chess.Board, initial_game_state: str, randomize_opening_moves: int
) -> tuple[str, chess.Board]:
    # We loop for multiple attempts because sometimes the random moves will result in a game over
    MAX_INIT_ATTEMPTS = 5
    for attempt in range(MAX_INIT_ATTEMPTS):
        board.reset()  # Reset the board for a new attempt
        game_state = initial_game_state  # Reset the game state for a new attempt
        moves = []
        for moveIdx in range(1, randomize_opening_moves + 1):
            for player in range(2):
                moves = list(board.legal_moves)
                if not moves:
                    break  # Break if no legal moves are available

                move = random.choice(moves)
                moveString = board.san(move)
                if moveIdx > 1 or player == 1:
                    game_state += " "
                game_state += (
                    str(moveIdx) + ". " + moveString if player == 0 else moveString
                )
                board.push(move)

            if not moves:
                break  # Break if no legal moves are available

        if moves:
            # Successful generation of moves, break out of the attempt loop
            break
    else:
        # If the loop completes without a break, raise an error
        raise Exception("Failed to initialize the game after maximum attempts.")

    print(game_state)
    return game_state, board

def play_game_human(
        player_one: Player,
        player_two: Player,
        max_games: int = 1,
        randomize_opening_moves: Optional[int] = None,
)-> None:
    with open("gpt_inputs/prompt.txt", "r") as f:
            game_state = f.read()
    board = chess.Board()

    player_one_illegal_moves = 0
    player_two_illegal_moves = 0
    player_one_legal_moves = 0
    player_two_legal_moves = 0
    player_one_resignation = False
    player_two_resignation = False
    player_one_failed_to_find_legal_move = False
    player_two_failed_to_find_legal_move = False
    start_time = time.time()

    total_moves = 0
    illegal_moves = 0

    while not board.is_game_over():
        with open("game.txt", "w") as f:
            f.write(game_state)
        current_move_num = str(board.fullmove_number) + "."
        total_moves += 1
        # I increment legal moves here so player_two isn't penalized for the game ending before its turn
        player_one_legal_moves += 1
        player_two_legal_moves += 1

        # this if statement may be overkill, just trying to get format to exactly match PGN notation
        if board.fullmove_number != 1:
            game_state += " "
        game_state += current_move_num
        print(f"{current_move_num}", end="")
        (
            game_state,
            player_one_resignation,
            player_one_failed_to_find_legal_move,
            illegal_moves_one,
        ) = play_turn(player_one, board, game_state, player_one=True)
        player_one_illegal_moves += illegal_moves_one
        print(f"successfully processed move {player_one_legal_moves} for player 1")
        if illegal_moves_one != 0:
            player_one_legal_moves -= 1
        if (
            board.is_game_over()
            or player_one_resignation
            or player_one_failed_to_find_legal_move
        ):
            break

        (
            game_state,
            player_two_resignation,
            player_two_failed_to_find_legal_move,
            illegal_moves_two,
        ) = play_turn(player_two, board, game_state, player_one=False)
        player_two_illegal_moves += illegal_moves_two
        print(f"successfully processed move {player_two_legal_moves} for player 2")
        if illegal_moves_two != 0:
            player_two_legal_moves -= 1
        if (
            board.is_game_over()
            or player_two_resignation
            or player_two_failed_to_find_legal_move
        ):
            break

        print("\n", end="")


    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nGame over. Total time: {total_time} seconds")
    print(f"Result: {board.result()}")
    print(board)
    print()
    record_results(
        board,
        player_one,
        player_two,
        game_state,
        player_one_illegal_moves,
        player_two_illegal_moves,
        player_one_legal_moves,
        player_two_legal_moves,
        total_time,
        player_one_resignation,
        player_two_resignation,
        player_one_failed_to_find_legal_move,
        player_two_failed_to_find_legal_move,
        total_moves,
        illegal_moves,
    )
    if isinstance(player_one, StockfishPlayer):
        player_one.close()
    if isinstance(player_two, StockfishPlayer):
        player_two.close()
    return



#################################################################################

N_MATE_MOVES = 6 # mate in 5 or less
MATES_FILE = "mates_rlhf.txt"
LOCK_FILE = MATES_FILE + ".lock"
lock = FileLock(LOCK_FILE)
SEEN_MATES = set()
def play_game(
    player_one: Player,
    player_two: Player,
    max_games: int = 10,
    randomize_opening_moves: Optional[int] = None,
):
    # NOTE: I'm being very particular with game_state formatting because I want to match the PGN notation exactly
    # It looks like this: 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 etc. HOWEVER, GPT prompts should not end with a trailing whitespace
    # due to tokenization issues. If you make changes, ensure it still matches the PGN notation exactly.
    # stockfish_path = "/usr/games/stockfish"
    stockfish_path = "/home/ext-jjunkin/data_snatha11/searchless_chess/src/engines/stockfish"
    helper = chess.engine.SimpleEngine.popen_uci(stockfish_path) # probably will need to make sure my helper has the same color as my agent
    for _ in range(max_games):  # Play 10 games
        ## commenting this out for now. but I likely will need some kind of prompt once I start training language-capable models
        with open("gpt_inputs/prompt.txt", "r") as f:
            game_state = f.read()
        board = chess.Board()
        # print(board)

        if randomize_opening_moves is not None:
            game_state, board = initialize_game_with_random_moves(
                board, game_state, randomize_opening_moves
            )

        player_one_illegal_moves = 0
        player_two_illegal_moves = 0
        player_one_legal_moves = 0
        player_two_legal_moves = 0
        player_one_resignation = False
        player_two_resignation = False
        player_one_failed_to_find_legal_move = False
        player_two_failed_to_find_legal_move = False
        start_time = time.time()

        total_moves = 0
        illegal_moves = 0
        last_add = 0
        while not board.is_game_over():
            with open("game.txt", "w") as f:
                f.write(game_state)
            current_move_num = str(board.fullmove_number) + "."
            total_moves += 1
            # I increment legal moves here so player_two isn't penalized for the game ending before its turn
            player_one_legal_moves += 1
            player_two_legal_moves += 1

            # this if statement may be overkill, just trying to get format to exactly match PGN notation
            if board.fullmove_number != 1:
                game_state += " "
            game_state += current_move_num
            ############ This code is for getting out mate <M5
            # not attempting to generalize this to black yet, even though it wouldn't be hard. probablyt should.
            # info = helper.analyse(board, chess.engine.Limit(depth=10), multipv=5)
            # top_score = info[0]['score'].relative
            # # if it's mate and we aren't the one getting mated
            # if top_score.is_mate() and top_score.mate() > 0: 
            #     fen_string = board.fen()
            #     sp = fen_string.split()
            #     position_only_fen = " ".join(sp[:-2])
            #     move_num = int(sp[-1])
            #     # make it so there are no repeat positions and 10 moves must pass
            #     # between mate positions added (this way there's more diversity and 
            #     # # less overfitting)
            #     if position_only_fen not in SEEN_MATES and (move_num - last_add >= 10 or last_add==0):
            #         last_add = move_num
            #         SEEN_MATES.add(position_only_fen)
            #         with lock, open(MATES_FILE, 'a') as f:
            #             f.write(f"{fen_string}\n") # move & half-move shouldn't be in train process, but it is, so I'll keep it in RL
            #             # not putting starting eval in there. we can calculate in the loop
                    
                    
            # if info[0]['score'].relative.is_mate():
            #     mx = info[0]['score'].relative.mate()
            #     if mx < N_MATE_MOVES:
            #         fen_string = board.fen()
            #         position_only_fen = " ".join(fen_string.split()[:-2])
            #         if position_only_fen not in SEEN_MATES:
            #             SEEN_MATES.add(position_only_fen)
            #             with lock, open(MATES_FILE, 'a') as f:
            #                 f.write(f"{fen_string}\n")
                            

            (
                game_state,
                player_one_resignation,
                player_one_failed_to_find_legal_move,
                illegal_moves_one,

            ) = play_turn(player_one, board, game_state, player_one=True)
            player_one_illegal_moves += illegal_moves_one
            if illegal_moves_one != 0:
                player_one_legal_moves -= 1
            if (
                board.is_game_over()
                or player_one_resignation
                or player_one_failed_to_find_legal_move
            ):
                break

            (
                game_state,
                player_two_resignation,
                player_two_failed_to_find_legal_move,
                illegal_moves_two,
            ) = play_turn(player_two, board, game_state, player_one=False)
            player_two_illegal_moves += illegal_moves_two
            if illegal_moves_two != 0:
                player_two_legal_moves -= 1
            if (
                board.is_game_over()
                or player_two_resignation
                or player_two_failed_to_find_legal_move
            ):
                break

            print("\n", end="")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nGame over. Total time: {total_time} seconds")
        print(f"Result: {board.result()}")
        print(board)
        print()
        record_results(
            board,
            player_one,
            player_two,
            game_state,
            player_one_illegal_moves,
            player_two_illegal_moves,
            player_one_legal_moves,
            player_two_legal_moves,
            total_time,
            player_one_resignation,
            player_two_resignation,
            player_one_failed_to_find_legal_move,
            player_two_failed_to_find_legal_move,
            total_moves,
            illegal_moves,
        )
    if isinstance(player_one, StockfishPlayer):
        player_one.close()
    if isinstance(player_two, StockfishPlayer):
        player_two.close()

        # print(game_state)


RUN_FOR_ANALYSIS = True
recording_file = "logs/determine.csv"  # default recording file. Because we are using list [player_ones], recording_file is overwritten
# player_ones = ['ckpt12000.pt']# player_ones = ["ckpt600000.pt"]
player_ones = ['ckpt' + str(200000) + "_causal.pt"]
player_two_recording_name = "stockfish"

CODEX = InferenceTimeBehavioralCloning()



if __name__ == "__main__":
    
    # ## playing a human game aginst the bot
    start = 0 # forcing human to play black
    if start == 0: # human is white
        color = "white"
        player_one = humanPlayer()
        player_one_recording_name = "human"
        player_two_recording_name = player_ones[0]
        player_two = NanoGptPlayer(model_name=player_one_recording_name, model_path="/home/ext-jjunkin/data_snatha11/searchless_chess/src/out/ckpt96000_causal.pt", tokenizer=CODEX.encode, decoder=CODEX.decode)
    # else: # human is black
    #     color = "black"
    #     player_one_recording_name = player_ones[0]
    #     player_two_recording_name = "human"
    #     player_one = NanoGptPlayer(model_name=player_one_recording_name, model_path="/workspace/searchless_chess/src/out", tokenizer=CODEX.encode, decoder=CODEX.decode)
    #     player_two = humanPlayer()
    # print(f"Starting game against model {player_ones[0]}. You are {color}. Good luck!")
    play_game_human(player_one=player_one, player_two=player_two)
    
    
    ## play 100 games against each stockfish agent to test how strong the model really is
    ## removing draws as deepmind did.
    # stockfish_play_time=0.1
    # num_games=100
    # player_one_recording_name=player_ones[0]
    # DRAW_FILE = "draws.txt"
    # for i in range(11):
    #     player_one = NanoGptPlayer(model_name=player_one_recording_name, model_path="/workspace/searchless_chess/src/out", tokenizer=CODEX.encode, decoder=CODEX.decode)
    #     player_two_recording_name = "stockfish" + str(i)
    #     with open(DRAW_FILE, 'a') as f:
    #         f.write(f"{player_two_recording_name}\n")
    #         f.close()
    #     player_two = StockfishPlayer(skill_level=i, play_time=stockfish_play_time)
    #     play_game(player_one, player_two, num_games)
    
    ## initial testing. playing a few games against a single stockfish agent of hardcoded level
    # for player in player_ones:
    #     player_one_recording_name = player
    #     num_games = 10
    #     player_one = NanoGptPlayer(model_name=player_one_recording_name, model_path="/workspace/searchless_chess/src/out", tokenizer=CODEX.encode, decoder=CODEX.decode)
    #     player_two = StockfishPlayer(skill_level=25, play_time=stockfish_play_time)
    #     play_game(player_one, player_two, num_games)
        
    ## play 100 games against each stockfish agent (50 with white, 50 with black) to test how strong the model really is
    # stockfish_play_time=0.1
    # num_games=50
    # # player_one_recording_name=player_ones[0]
    # player_one_recording_name = "ckpt96000_causal.pt"
    # DRAW_FILE = "draws.txt"
    # # white
    # for i in range(11):
    #     player_one = NanoGptPlayer(model_name=player_one_recording_name, model_path="/home/ext-jjunkin/data_snatha11/searchless_chess/src/out/", tokenizer=CODEX.encode, decoder=CODEX.decode)
    #     player_two_recording_name = "stockfish" + str(i)
    #     # with open(DRAW_FILE, 'a') as f:
    #     #     f.write(f"{player_two_recording_name}\n")
    #     #     f.close()
    #     player_two = StockfishPlayer(skill_level=i, play_time=stockfish_play_time)
    #     play_game(player_one, player_two, num_games)
        
    # stockfish_play_time=0.1
    # num_games=50
    # player_two_recording_name=player_ones[0]
    # DRAW_FILE = "draws.txt"
    # # black
    # for i in range(11):
    #     player_two = NanoGptPlayer(model_name=player_two_recording_name, model_path="/workspace/searchless_chess/src/out", tokenizer=CODEX.encode, decoder=CODEX.decode)
    #     player_one_recording_name = "stockfish" + str(i)
    #     # with open(DRAW_FILE, 'a') as f:
    #     #     f.write(f"{player_one_recording_name}\n")
    #     #     f.close()
    #     player_one = StockfishPlayer(skill_level=i, play_time=stockfish_play_time)
    #     play_game(player_one, player_two, num_games)