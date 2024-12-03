import dataclasses
from typing import Literal
from transformers import PreTrainedTokenizer
# import config as config_lib

PolicyType = Literal['action_value', 'state_value', 'behavioral_cloning']
@dataclasses.dataclass(kw_only=True)
class LanguageDataConfig:
  """Config for the data generation."""

  # The batch size for the sequences.
  batch_size: int
  # Pretrained Tokenizer to use for embedding
  tokenizer: PreTrainedTokenizer
  tokenizer_save_path: str
  # Whether to shuffle the dataset (shuffling is applied per epoch).
  shuffle: bool = False
  # The seed used for shuffling and transformations of the data.
  seed: int | None = 0
  # Whether to drop partial batches.
  drop_remainder: bool = False
  # The number of child processes launched to parallelize the transformations.
  worker_count: int | None = 0
  # The number of return buckets.
  num_return_buckets: int
  # The dataset split.
  split: Literal['train', 'test']
  # The policy used to create the dataset.
  policy: PolicyType
  # The number of records to read from the dataset (can be useful when, e.g.,
  # the dataset does not fit into memory).
  num_records: int | None = None