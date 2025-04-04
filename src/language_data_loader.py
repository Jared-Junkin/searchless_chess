import numpy as np
import chess
import grain.python as pygrain
from data_loader import ConvertToSequence
import tokenizer as deepmind_tokenizer
import constants
from typing import Tuple
from config_language import LanguageDataConfig
import os
import utils
import torch
import bagz
import jax
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
import jax.numpy as jnp
from jared_data_loader import BagDataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler, RandomSampler
# goal: integrate the tokenizer with this.


class ConvertBehavioralCloningDataToLanguage:
    def __init__(self, num_return_buckets: int, tokenizer: PreTrainedTokenizer, tokenizer_save_path: str) -> None:
        self.num_return_buckets = num_return_buckets
        self._tokenizer = None
        self._move_encodings = {}
        self._pretokenized_prompt = []
        self._predefined_array = []
        self._loss_mask = []
        self._attn_mask = []
        self._save_file_name = tokenizer_save_path
        self._initTokenizer(tokenizer=tokenizer)
        self._board = chess.Board()

    def _initTokenizer(self, tokenizer: PreTrainedTokenizer)->None:
        
        # pretokenize chess moves
        all_moves = utils._compute_all_possible_actions()
        all_moves = list(all_moves[0].keys())
        original_mappings = {move:tokenizer(move, add_special_tokens=False)["input_ids"] for move in all_moves}     # embedding of 'g1h2' before special tokens added. 'g1h2': [70, 16, 71, 17]
        tokenizer.add_special_tokens({"additional_special_tokens": all_moves})                                      # adding tokens
        self._tokenizer = tokenizer                                                                                 # storing tokenizer
        final_mappings = {move: tokenizer.convert_tokens_to_ids(move) for move in all_moves}                        # embedding of 'g1h2 after special tokens added: 'g1h2': [128413]
        
        
        self._move_encodings = {move: original_mappings[move] + 
                               self._tokenizer(": ", add_special_tokens=False)["input_ids"] + 
                               [final_mappings[move]] + 
                               self._tokenizer(", ", add_special_tokens=False)["input_ids"] for move in all_moves}  # 'g1h2': [70, 16, 71, 17] + tokenizer(": ") + [128413] + ", "
        

        
        # # Pre-tokenize the static part of the prompt
        
        # prompt_components [fen, then best move]

        prompt_components = [
            "You are a chess grandmaster. This is the board position in FEN notation: ",   # fen comes after this
           "The legal moves are: ",                                                                                                            # legal moves comes after this
            "Which of these is the best move? Best move: "                                       # best move comes after this.     
            
        ]

        # define pretokenized prompt
        for component in prompt_components:
            token_ids = tokenizer(component, add_special_tokens=False)["input_ids"]
            self._pretokenized_prompt.append(np.array(token_ids, dtype=np.int32))

        # preallocated array
        static_prompt_length = sum(len(comp) for comp in self._pretokenized_prompt)             # number of tokens in static prompt
        max_encoding_length = int(np.mean([len(self._move_encodings[move]) for move in all_moves]))     # max length in tokens of a single legal move in prompt
        dynammic_prompt_length = max_encoding_length * 80 # (empircally, there aren't any board states in dataset with more legal moves than this)                                 # max length needed for dynammic prompt (there will never by more than 128 legal moves)
        total_prompt_length = static_prompt_length + dynammic_prompt_length + 1                 # total prompt length (+1 for legal move token at end)
        
        pad_token_id = self._tokenizer.convert_tokens_to_ids("<|pad|>")                         # get token id of pad character
        self._predefined_array = np.full(                                                    # This array is going to be our prompty. pre-definining it so we don't have ot initialize each time
            (total_prompt_length,), pad_token_id, dtype=np.int32
        )
        


        # Create the directory if it doesn't exist
        os.makedirs(self._save_file_name, exist_ok=True)

        # Save the tokenizer
        tokenizer.save_pretrained(self._save_file_name)

        # print(f"Tokenizer saved at: {self._save_file_name}")
        # create and save the loss mask
        self._loss_mask: torch.Tensor = torch.full(
            size=(total_prompt_length,),
            fill_value=True,
            dtype=bool,
        )
        
        self._attn_mask: torch.Tensor = torch.full(
            size=(total_prompt_length,),
            fill_value=True,
            dtype=bool,
        )

    def _tokenize(self, fen: str, move: str)->np.ndarray:
        # should replace all this with torch.cat and make commands torch.
        
        
        # get legal moves
        self._board.set_fen(fen=fen)
        # tokenize fen
        # fen_tokens = torch.tensor(self._tokenizer(fen, add_special_tokens=False)['input_ids'], dtype=torch.int32)
        fen_tokens = np.array(self._tokenizer(fen, add_special_tokens=False)['input_ids'], dtype=np.int32)
        # tokenize legal moves
        # legal_moves = torch.cat([torch.tensor(self._move_encodings[str(m)], dtype=torch.int32) for m in self._board.legal_moves])
        legal_moves = [
                       np.array(self._move_encodings[str(m)], dtype=np.int32) for m in self._board.legal_moves
                       ]
        legal_moves = np.concatenate(legal_moves, axis=0)
        # tokenize best next move
        # move_tokens = torch.tensor(self._tokenizer(move, add_special_tokens=False)['input_ids'], dtype=torch.int32)
        move_tokens = np.array(self._tokenizer(move, add_special_tokens=False)['input_ids'], dtype=np.int32)

        
        # assemble prompt tokens
        # Ensure all components are tensors and safely clone/detach them
        prompt_tokens = np.concatenate(
            [
                self._pretokenized_prompt[0],
                fen_tokens,
                self._pretokenized_prompt[1],
                legal_moves,
                self._pretokenized_prompt[2],
                move_tokens
            ]
        )
        # prompt_tokens = torch.cat([
        #     self._pretokenized_prompt[0],  # Clone and detach the pretokenized prompt part 0
        #     torch.tensor(fen_tokens, dtype=torch.int32).clone().detach() if not isinstance(fen_tokens, torch.Tensor) else fen_tokens.clone().detach(),
        #     self._pretokenized_prompt[1],  # Clone and detach the pretokenized prompt part 1
        #     torch.tensor(legal_moves, dtype=torch.int32).clone().detach() if not isinstance(legal_moves, torch.Tensor) else legal_moves.clone().detach(),
        #     self._pretokenized_prompt[2]   # Clone and detach the pretokenized prompt part 2
        # ])
        
        predefined_array = np.copy(self._predefined_array)  # Create a copy to avoid modifying the original
        predefined_attn_mask = np.copy(self._attn_mask)
        
        # Copy tokens into the preallocated array (up to its size limit)
        tokens_to_copy = min(len(prompt_tokens), len(predefined_array))
        predefined_array[:tokens_to_copy] = prompt_tokens[:tokens_to_copy]
        predefined_attn_mask[:tokens_to_copy-1] = False # attend to all non-padding tokens except the target token that we're trying to predict
        
        ## this code set the final entry in the array (after padding) to be the target value we wanted to predict. 
        ## I've decided it's better practice to make the final value the model wants to predict the token immediately following the last token in the prompt
        # # Set the final token to be best move token
        # predefined_array[-1] = move_tokens
        # self._attn_mask[-1] = False # attend to final token (best move)
        
        # set loss mask to be false for just the targe ttoken @ predefined_array[tokens_to_copy-1] (the last non padding entry in the array)
        predefined_loss_mask = np.copy(self._loss_mask)
        predefined_loss_mask[tokens_to_copy-1] = False # calculate loss on only location of token we want to predict
        
        # return predefined array.
        return predefined_array, predefined_attn_mask, predefined_loss_mask
        

    def map(self, element: bytes) -> Tuple[np.ndarray, np.ndarray]:
        # Decode FEN and move
        fen, move = constants.CODERS['behavioral_cloning'].decode(element)
        sequence, attn, loss = self._tokenize(fen, move)

        return sequence, attn, loss
    
_TRANSFORMATION_BY_POLICY = {
    "language": ConvertBehavioralCloningDataToLanguage
    # 'action_value': ConvertActionValueDataToSequence, # commenting out because we don't need these
    # 'state_value': ConvertStateValueDataToSequence,
}


# Follows the base_constants.DataLoaderBuilder protocol.
def build_data_loader_language(config: LanguageDataConfig, world_size: int= 0, rank: int = 0) -> pygrain.DataLoader:
    """Returns a data loader for chess from the config."""
    data_source = bagz.BagDataSource(
        os.path.join(
            "/workspace/searchless_chess/data", config.split, "behavioral_cloning_data.bag"
        )
    )

    if config.num_records is not None:
        num_records = config.num_records
        if len(data_source) < num_records:
            raise ValueError(
                f'[Process {jax.process_index()}]: The number of records requested'
                f' ({num_records}) is larger than the dataset ({len(data_source)}).'
            )
    else:
        num_records = len(data_source)

    transformations = (
        [_TRANSFORMATION_BY_POLICY[config.policy](
            num_return_buckets=config.num_return_buckets,
            tokenizer = config.tokenizer,
            tokenizer_save_path=config.tokenizer_save_path
        )]
    )
    dataset: Dataset = BagDataset(data_source, transformations)
    if world_size > 1:
        sampler: DistributedSampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=config.shuffle,
        )
    else:
        sampler = RandomSampler(dataset) if config.shuffle else Sampler(dataset)
        
    #   sampler = pygrain.IndexSampler(
    #       num_records=num_records,
    #       shard_options=pygrain.NoSharding(),
    #       shuffle=config.shuffle,
    #       num_epochs=None,
    #       seed=config.seed,
    #   )

    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.worker_count,
        pin_memory=True,
        drop_last=True
    )
    
    
# simple wrapper class for the torch dataloader object. Wrote it because I need a way to iterate repeatedly over the dataloader
# this just re-initializes when we reach the end.
class LlamaLoader:
    def __init__(self, loader: DataLoader):
        self._loader = loader
        self._loader_iter = iter(self._loader)
    def __len__(self)->int:
        return len(self._loader)
    def __next__(self)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            seq, attn_mask, loss_mask = next(self._loader_iter)
        except StopIteration:
            self._loader_iter = iter(self._loader)
            seq, attn_mask, loss_mask = next(self._loader_iter)

        return seq, attn_mask, loss_mask
        
def load_dataloader(tokenizer: PreTrainedTokenizer, split: str, batch_size: int, out_dir: str, shuffle: bool, policy: str, worker_count: int = 0, world_size: int = 0, local_rank: int = 0, num_return_buckets: int = 128)->DataLoader:
    train_data = LanguageDataConfig(
        batch_size= batch_size,
        tokenizer=tokenizer,
        tokenizer_save_path=out_dir,
        shuffle=shuffle,
        worker_count=worker_count,  # 0 disables multiprocessing.
        num_return_buckets=num_return_buckets,
        policy=policy,
        split=split,
    )
    print(f"building data loader with world size = {world_size} and local rank = {local_rank}")
    data_iter = build_data_loader_language(config=train_data, world_size=world_size, rank=local_rank)
    return data_iter