from typing import Any, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import os
from typing import List
import bagz

# deepmind packages
import tokenizer
import utils
import constants
import config as config_lib


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
    
    

_TRANSFORMATION_BY_POLICY = {
    'behavioral_cloning': ConvertBehavioralCloningDataToSequence,
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

        # Assuming the transformation returns (sequence, loss_mask)
        sequence, _ = element

        # Convert to torch tensor if it's not already
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence, dtype=torch.long)

        return sequence
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