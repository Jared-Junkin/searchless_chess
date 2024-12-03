from typing import Any, List, Tuple, Callable
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import os
from typing import List
import bagz
import grain.python as pygrain
import jax
# deepmind packages
import tokenizer
import utils
import constants
import config as config_lib
from data_loader import ConvertToSequence
# for Llama:
import chess


def _process_fen(fen: str) -> np.ndarray:
  return tokenizer.tokenize(fen).astype(np.int32)


def _process_move(move: str) -> np.ndarray:
  return np.asarray([utils.MOVE_TO_ACTION[move]], dtype=np.int32)


class ConvertBehavioralCloningDataToSequence:
    """Converts the FEN and move into a sequence of integers."""

    def __init__(self, num_return_buckets: int) -> None:
        # Initialize any necessary variables
        self.num_return_buckets = num_return_buckets
        # Initialize _loss_mask if needed
        self._sequence_length = tokenizer.SEQUENCE_LENGTH + 1  # (s) + (a)
        self._loss_mask: np.ndarray = np.full(
            shape=(self._sequence_length,),
            fill_value=True,
            dtype=bool,
        )
        self._loss_mask[-1] = False

    def map(self, element: bytes) -> Tuple[np.ndarray, np.ndarray]:
        fen: str
        move: str
        fen, move = constants.CODERS['behavioral_cloning'].decode(element)
        state: np.ndarray = _process_fen(fen)
        action: np.ndarray = _process_move(move)
        sequence: np.ndarray = np.concatenate([state, action])

        return sequence, self._loss_mask  # Return sequence and loss_mask if needed

# # this is going to be used for Llama.
# import numpy as np

# class ConvertBehavioralCloningDataToLanguage(ConvertToSequence):
#     def __init__(self, num_return_buckets: int) -> None:
#         # Initialize any necessary variables
#         super().__init__(num_return_buckets)
#         self.num_return_buckets = num_return_buckets
#         self._loss_mask: np.ndarray = np.full(
#             shape=(self._sequence_length,),
#             fill_value=True,
#             dtype=bool,
#         )
#         self._board = chess.Board()

#     @property
#     def _sequence_length(self) -> int:
#         return tokenizer.SEQUENCE_LENGTH + 1  # (s) + (a)

#     def map(self, element: bytes) -> Tuple[str, str, np.ndarray, np.ndarray]:
#         # Decode FEN and move
#         fen, move = constants.CODERS['behavioral_cloning'].decode(element)
#         self._board.set_fen(fen=fen)

#         # Get legal moves and create a fixed-size array
#         legal_moves = [str(m) for m in self._board.legal_moves]
#         predefined_array = np.full(128, "<|pad|>", dtype=object)  # Fixed-size array with padding
#         predefined_array[:min(len(legal_moves), 128)] = legal_moves[:128]  # Populate with legal moves

#         prompt = (
#             f"You are a chess grandmaster playing a game of chess. You want to win and demonstrate your incredible skill."
#             f"This is the board. It is your move: {fen}."
#             f"Please select the best legal moves: {', '.join(predefined_array)}. Please ONLY PLAY MOVES LISTED HERE. ANY move not in here is illegal. You lose if you play an illegal move."
#             f"Best move is: {move}"
#         )


#         # Return all required elements
#         return fen, move, self._loss_mask, predefined_array, len(legal_moves)

    
class ConvertRLHFDataToSequence:
    def __init__(self, num_return_buckets: int = 0) -> None:
        self.num_return_buckets = num_return_buckets
    def map(self, element: str)->Tuple[np.ndarray, np.ndarray]:
        return element # currently I am storing my mating positions as raw fen strings in a text file, so I won't need to do any transformations. keeping this in if that changes

_TRANSFORMATION_BY_POLICY = {
    'behavioral_cloning': ConvertBehavioralCloningDataToSequence,
    'rlhf': ConvertRLHFDataToSequence
    # 'action_value': ConvertActionValueDataToSequence, # commenting out because we don't need these
    # 'state_value': ConvertStateValueDataToSequence,
}


class InferenceTimeBehavioralCloning:
    def __init__(self):
        # self.MOVE_TO_ACTION is first thing returned "_" but we don't need it anywhere.
        _, self.ACTION_TO_MOVE = utils._compute_all_possible_actions()
    def encode(self, fen: str)->np.ndarray:
        state = _process_fen(fen)
        pad = np.zeros(shape=(1,))
        sequence = np.concatenate([pad, state]) # have to zero pad the first column of sequence (entry really, because it's just a vector. batch size = 1 @ inference time) because that is how it recognizes the start of a board. (for inputs.)
        return sequence

    def decode(self, token: int)->str:
        return self.ACTION_TO_MOVE[token]
    
    

class BagDataset(Dataset):
    def __init__(self, data_source: Any, transformations: List[Any]) -> None:
        self.data_source = data_source
        self.transformations = transformations

    def __len__(self) -> int:
        return len(self.data_source)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Get the raw data from data_source
        element: bytes = self.data_source[idx]

        # Apply transformations
        for transform in self.transformations:
            element = transform.map(element)

        # Assuming the transformation returns (sequence)
        try:
            sequence, attn_mask, loss_mask = element 
        except:
            sequence = element
            

        # Convert to torch tensor if it's not already
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence, dtype=torch.long)

        return sequence, attn_mask, loss_mask
# import time
def build_data_loader_parallel(
    config: config_lib.DataConfig,
    rank: int,
    world_size: int
) -> DataLoader:
    """Returns a data loader for chess from the config."""
    # start_time = time.time()
    data_source = bagz.BagDataSource(
        os.path.join(
            os.getcwd(),
            f'../data/{config.split}/{config.policy}_data.bag',
        ),
    )
    # print(f"data_source initialized in {time.time()-start_time} seconds")
    
    if config.num_records is not None:
        num_records: int = config.num_records
        if len(data_source) < num_records:
            raise ValueError(
                f'[Process {rank}]: The number of records requested'
                f' ({num_records}) is larger than the dataset ({len(data_source)}).'
            )
        else:
            # Optionally, limit the data_source to num_records
            data_source = data_source[:num_records]

    transformations: List[Any] = [
        _TRANSFORMATION_BY_POLICY[config.policy](
            num_return_buckets=config.num_return_buckets
        ),
    ]

    # Create the dataset
    # start_time = time.time()
    dataset: Dataset = BagDataset(data_source, transformations)
    # print(f"Dataset initialized in {time.time()-start_time} seconds")
    # start_time = time.time()
    # Create the DistributedSampler
    sampler: DistributedSampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=config.shuffle,
    )
    # print(f"distributed sampler initialized in {time.time()-start_time} seconds")

    # Create the DataLoader
    # start_time = time.time()
    data_loader: DataLoader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.worker_count,  # Adjust based on your system
        pin_memory=True,
        drop_last=True,
    )
    # print(f"dataloader initialized in {time.time()-start_time} seconds")
    return data_loader


class RLHFDataset(Dataset):
    def __init__(self, data_source: str, transformations: Any) -> None:
        """
        Args:
            data_source (str): Path to the .txt file containing FEN strings (one per line).
            transformations (List[Callable]): List of transformations to apply to each FEN string.
        """
        self.data_source = data_source
        self.transformations = transformations
        self.line_offsets = self._get_line_offsets()

    def _get_line_offsets(self) -> List[int]:
        """
        Reads the file once to determine the byte offsets of each line (lazy loading).
        This allows random access to any line in the file using `seek`.
        """
        line_offsets = []
        offset = 0

        with open(self.data_source, "r") as f:
            for line in f:
                line_offsets.append(offset)
                offset += len(line)

        return line_offsets

    def __len__(self) -> int:
        """Returns the number of FEN strings in the dataset."""
        return len(self.line_offsets)

    def __getitem__(self, index: int) -> Any:
        """Returns the FEN string at the specified index, applying any transformations."""
        if index < 0 or index >= len(self):
            raise IndexError("Index out of bounds")

        # Open the file and seek to the correct line offset
        with open(self.data_source, "r") as f:
            f.seek(self.line_offsets[index])
            fen_string = f.readline().strip()

        # Apply transformations
        for transform in self.transformations:
            fen_string = transform.map(fen_string)

        return fen_string
    
# jared's dataloader intended for conducting rlhf to improve upon Deepmind's results.
def build_data_loader_rlhf(config: config_lib.DataConfig,
    rank: int,
    world_size: int) -> DataLoader:
    data_source = config.split
    transformations: List[Any] = [
        _TRANSFORMATION_BY_POLICY[config.policy](
            num_return_buckets=config.num_return_buckets
        ),
    ]
    dataset = RLHFDataset(data_source=data_source, transformations=transformations)
    sampler: DistributedSampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=config.shuffle,
    )
    data_loader: DataLoader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.worker_count,  # Adjust based on your system
        pin_memory=True,
        drop_last=False,
    )
    return data_loader



# # Follows the base_constants.DataLoaderBuilder protocol.
# def build_data_loader_language(config: config_lib.DataConfig) -> pygrain.DataLoader:
#   """Returns a data loader for chess from the config."""
#   data_source = bagz.BagDataSource(
#       os.path.join(
#           os.getcwd(),
#           f'../data/{config.split}/{config.policy}_data.bag',
#       ),
#   )

#   if config.num_records is not None:
#     num_records = config.num_records
#     if len(data_source) < num_records:
#       raise ValueError(
#           f'[Process {jax.process_index()}]: The number of records requested'
#           f' ({num_records}) is larger than the dataset ({len(data_source)}).'
#       )
#   else:
#     num_records = len(data_source)

#   sampler = pygrain.IndexSampler(
#       num_records=num_records,
#       shard_options=pygrain.NoSharding(),
#       shuffle=config.shuffle,
#       num_epochs=None,
#       seed=config.seed,
#   )
#   transformations = (
#       _TRANSFORMATION_BY_POLICY["language"](
#           num_return_buckets=config.num_return_buckets
#       ),
#       pygrain.Batch(config.batch_size, drop_remainder=True),
#   )
#   return pygrain.DataLoader(
#       data_source=data_source,
#       sampler=sampler,
#       operations=transformations,
#       worker_count=config.worker_count,
#       read_options=None,
#   )

# def build_data_loader_language(config: config_lib.DataConfig,
#                                rank:int,
#                                world_size: int,
# )->DataLoader:
#     """Returns a data loader for chess from the config."""
#     # start_time = time.time()
#     data_source = bagz.BagDataSource(
#         os.path.join(
#             os.getcwd(),
#             f'../data/{config.split}/{config.policy}_data.bag',
#         ),
#     )
#     # print(f"data_source initialized in {time.time()-start_time} seconds")
    
#     if config.num_records is not None:
#         num_records: int = config.num_records
#         if len(data_source) < num_records:
#             raise ValueError(
#                 f'[Process {rank}]: The number of records requested'
#                 f' ({num_records}) is larger than the dataset ({len(data_source)}).'
#             )
#         else:
#             # Optionally, limit the data_source to num_records
#             data_source = data_source[:num_records]

#     transformations: List[Any] = [
#         _TRANSFORMATION_BY_POLICY["language"](
#             num_return_buckets=config.num_return_buckets
#         ),
#     ]

#     # Create the dataset
#     # start_time = time.time()
#     dataset: Dataset = BagDataset(data_source, transformations)
#     # print(f"Dataset initialized in {time.time()-start_time} seconds")
#     # start_time = time.time()
#     # Create the DistributedSampler
#     sampler: DistributedSampler = DistributedSampler(
#         dataset,
#         num_replicas=world_size,
#         rank=rank,
#         shuffle=config.shuffle,
#     )
#     # print(f"distributed sampler initialized in {time.time()-start_time} seconds")

#     # Create the DataLoader
#     # start_time = time.time()
#     data_loader: DataLoader = DataLoader(
#         dataset,
#         batch_size=config.batch_size,
#         sampler=sampler,
#         num_workers=config.worker_count,  # Adjust based on your system
#         pin_memory=True,
#         drop_last=True,
#     )
#     # print(f"dataloader initialized in {time.time()-start_time} seconds")
#     return data_loader