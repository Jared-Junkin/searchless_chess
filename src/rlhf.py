from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os
import logging
from dataclasses import dataclass
import chess
from itertools import cycle
from torch.optim import Adam
import chess.engine
from filelock import FileLock
from jared_data_loader import build_data_loader_rlhf, InferenceTimeBehavioralCloning
import time
import config as config_lib
import platform
from jared_models.nanoGPT  import NanoGptPlayer_rlhf
import random
from typing import Optional, Tuple
# from play_game import StockfishPlayer
# you are really tempting fate with all these global variables. be careful...
# Define base Player class
class Player:
    def get_move(self, board: chess.Board, game_state: str, temperature: float) -> str:
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError

class StockfishPlayer(Player):
    @staticmethod
    def get_stockfish_path() -> str:
        """
        Determines the operating system and returns the appropriate path for Stockfish.

        Returns:
            str: Path to the Stockfish executable based on the operating system.
        """
        if platform.system() == "Linux":
            return "/usr/games/stockfish"
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
        
class Critic(StockfishPlayer):
    def __init__(self, skill_level: int, play_time: float, depth: int=10, multipv: int=5):
        super().__init__(skill_level=skill_level, play_time=play_time)
        self.depth = depth
        self.multipv = multipv
        
    # has the added benefit that it is agnostic to which mate in 2 GPT chooses.
    # it only cares that it finds some mate in the minimum number of moves.
    def V(self, board: chess.Board, play_as: str)->int:
        original_turn = board.turn
        if play_as == "white":
            board.turn = chess.WHITE
        else:
            board.turn = chess.BLACK
        top_score = self.getTopScore(board)
        # if mate is no longer on the board, return 0
        if not top_score.is_mate():
            return 0
        # if our opponent somehow has mate, return 0
        if top_score.is_mate() and top_score.mate() < 0:
            return 0
        moves_to_mate = top_score.mate()
        # return our oracle-evaluated utility of being in state s
        board.turn = original_turn
        return 1 / (moves_to_mate + 1)
    def getTopScore(self, board: chess.Board):
        info = self._engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=self.multipv)
        top_score = info[0]['score'].relative
        return top_score
@dataclass
class LegalMoveResponse:
    move_san: Optional[str] = None
    move_uci: Optional[chess.Move] = None
    attempts: int = 0
    is_resignation: bool = False
    is_illegal_move: bool = False

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
            board, game_state, min(((attempt / max_attempts) * 1) + 0.15, 0.5)# incresasing the temperature to 2.0 (this is an important hyperparameter that should be backed out at some point)
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

def perform_rlhf(config: dict)->None:
    # set up logging.
    logfile_path = config["logfile"]
    log_dir = os.path.dirname(logfile_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logger = logging.getLogger(config["logname"])
    logger.setLevel("DEBUG")
    file_log =logging.FileHandler(config["logfile"])
    logger.addHandler(file_log)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_log.setFormatter(formatter)
    # create dataset, data sampler, data loader

    player_one_recording_name = 'ckpt' + str(config["ckpt_num"]) + ".pt"
    CODEX = InferenceTimeBehavioralCloning()
    # set up train.py as above before
    train_data = config_lib.DataConfig(
        batch_size=config["batch_size"],
        shuffle=True,
        worker_count=config["worker_count"],  # 0 disables multiprocessing.
        num_return_buckets=config["num_return_buckets"],
        policy=config["policy"],
        split=config["mates_file_path"]
    )
    start_time = time.time()
    # save model every 2000 checkpoints
    # set up tensorboard
    # set up logging
    
    # then you should be good to go!! Goal is to have these 4 things done by 9:00
    data_loader = build_data_loader_rlhf(config=train_data, rank=config["ddp_rank"], world_size=config["ddp_world_size"])
    infinite_loader = cycle(data_loader)
    num_batches_per_epoch = len(data_loader)

    player_one = NanoGptPlayer_rlhf(model_name=player_one_recording_name, model_path="/workspace/searchless_chess/src/out", tokenizer=CODEX.encode, decoder=CODEX.decode, draws_okay=True)
    player_two = StockfishPlayer(skill_level=config["stockfish_skill_level"], play_time=config["stockfish_play_time"])
    critic = Critic(skill_level=config["stockfish_skill_level"], play_time=config["stockfish_play_time"])
    optimizer = Adam(params=player_one.model.parameters(), lr=config["lr"])
    discount = config["discount"]
    # main rl game loop
    # while num_iters < iters:
    
    current_epoch = 0
    for i in range(int(config["num_games"])):
        if i % num_batches_per_epoch == 0:
            current_epoch +=1
            data_loader.sampler.set_epoch(current_epoch)
        game_start_pos = next(infinite_loader)[0] # get the 0th game (this stil returns as a tuple/list because of old function)
        board = chess.Board(game_start_pos)
        color = "white" if chess.WHITE else "black"

    
        top_score = critic.getTopScore(board)
        # if mate is no longer on the board, return 0
        if not top_score.is_mate():
            print(f"Original board position is note mate:\n {board}")
            continue
        with open("gpt_inputs/prompt.txt", "r") as f:
            game_state = f.read()
        moves_to_mate = top_score.mate()
        total_moves = 0
        
        
        player_one_illegal_moves = 0
        player_two_illegal_moves = 0
        player_one_legal_moves = 0
        player_two_legal_moves = 0
        player_one_resignation = False
        player_two_resignation = False
        player_one_failed_to_find_legal_move = False
        player_two_failed_to_find_legal_move = False
        start_time = time.time()
        illegal_moves = 0
        last_add = 0
        
        while not board.is_game_over():
            total_moves+=1
            optimizer.zero_grad()
            current_move_num = str(board.fullmove_number) + "."

            if board.fullmove_number != 1:
                game_state += " "
            game_state += current_move_num
            old_board_eval = critic.V(board=board,play_as=color)
            (
                    game_state,
                    player_one_resignation,
                    player_one_failed_to_find_legal_move,
                    illegal_moves_one,

            ) = play_turn(player_one, board, game_state, player_one=True)
            # make next move
            player_one_illegal_moves += illegal_moves_one
            if illegal_moves_one != 0:
                player_one_legal_moves -= 1
            if board.is_game_over() or player_one_resignation or player_one_failed_to_find_legal_move:
                # we'll have to backpropagate here when they do find mate
                if player_one_resignation or player_one_failed_to_find_legal_move:
                    continue
                else:
                    reward = 1
                    loss = player_one.logProbs * reward
                    print(f"player found mate!")
                    # backpropagate loss
                    loss.backward()
                    optimizer.step()
                    continue
            if color!="white":
                old_board_eval = critic.V(board=board, play_as=color)
            (
                game_state,
                player_two_resignation,
                player_two_failed_to_find_legal_move,
                illegal_moves_two,
            ) = play_turn(player_two, board, game_state, player_one=False)
            player_two_illegal_moves += illegal_moves_two
            
            # player blundered a draw or lost
            if board.is_game_over():
                reward = 0
            else:
                reward = (discount * critic.V(board=board, play_as=color)) - old_board_eval
            # calculate loss
            loss = player_one.logProbs * reward 
            # print(player_one.logProbs.requires_grad)  # Should print True
            # backpropagate loss
            loss.backward()
            optimizer.step()
            
        
        if isinstance(player_one, StockfishPlayer):
            player_one.close()
        if isinstance(player_two, StockfishPlayer):
            player_two.close()
        # I'm going to need to get the log probability of playing a|s
        # can I do that from nanoGPTPlayer?
    critic.close()
    
if __name__ == "__main__":
    
    config = {
        "discount": 0.99,
        "logfile": "/workspace/searchless_chess/src/logs/RLHF.log", # absolute path to place we're logging to.
        "logname": "rlhf_log",
        "batch_size": 1,
        "ddp_rank":0,
        "ddp_world_size": 1,
        "policy":"rlhf",
        "worker_count":0,
        "num_return_buckets":128,
        "mates_file_path": "/workspace/searchless_chess/src/mates_rlhf.txt", # absolute path to location of data source we're training on.
        "ckpt_num": 600000,
        "num_games": 1e7,
        "stockfish_skill_level": 20,
        "stockfish_play_time": 1e-2,
        "lr":7e-4,
    }
    
    perform_rlhf(config=config)