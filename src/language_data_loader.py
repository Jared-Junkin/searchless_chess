import numpy as np
import chess
import grain.python as pygrain
import constants
from typing import Tuple
from config_language import LanguageDataConfig
import os
import utils
import torch
from typing import List, Any, Optional
import bagz
import jax
from transformers import PreTrainedTokenizer
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler, RandomSampler
# goal: integrate the tokenizer with this.


class BagDataset(Dataset):
    def __init__(self, data_source: Any, tokenizer: PreTrainedTokenizer, tokenizer_save_path: str, prompt_components: Optional[List[str]]=None) -> None:
        if not prompt_components:
            # default prompt
            prompt_components = [
                "You are a chess grandmaster. This is the board position in FEN notation: ",   # fen comes after this
                "The legal moves are: ",                                                                                                            # legal moves comes after this
                "Which of these is the best move? Best move: "                                       # best move comes after this.     
                
            ]
        if len(prompt_components)!=3:
            raise ValueError(f"Expected prompt_components to have a length of 3, but got len {len(prompt_components)}.")
        
        self.data_source = data_source # required data source object we'll sample from
        self._tokenizer = None
        self._move_encodings = {}
        self._pretokenized_prompt = []
        self._predefined_array = []
        self._loss_mask = []
        self._attn_mask = []
        self._last_fen = None
        self._last_best_move = None
        self._prompt = prompt_components
        self._save_file_name = tokenizer_save_path
        self._init_tokenizer(tokenizer=tokenizer)
        self._board = chess.Board()
        
    def _init_tokenizer(self, tokenizer: PreTrainedTokenizer)-> None:
        # pretokenize chess moves
        all_moves = utils._compute_all_possible_actions()
        all_moves = list(all_moves[0].keys())
        original_mappings = {move:tokenizer(move, add_special_tokens=False)["input_ids"] for move in all_moves}     # embedding of 'g1h2' before special tokens added. 'g1h2': [70, 16, 71, 17]
        # tokenizer.add_special_tokens({"additional_special_tokens": all_moves})                                      # adding tokens
        self._tokenizer = tokenizer                                                                                 # storing tokenizer
        # final_mappings = {move: tokenizer.convert_tokens_to_ids(move) for move in all_moves}                        # embedding of 'g1h2 after special tokens added: 'g1h2': [128413]
        
        ## in this branch, we don't use new tokens
        # self._move_encodings = {move: original_mappings[move] +                                                     # 'g1h2': [70, 16, 71, 17]
        #                        self._tokenizer(": ", add_special_tokens=False)["input_ids"] +                       # tokenizer(": ")
        #                        [final_mappings[move]] + 
        #                        self._tokenizer(", ", add_special_tokens=False)["input_ids"] for move in all_moves}  # 'g1h2': [70, 16, 71, 17] + tokenizer(": ") + [128413] + ", "
        
        self._move_encodings = {move: original_mappings[move] + self._tokenizer(", ", add_special_tokens=False)["input_ids"] for move in all_moves}

        # define pretokenized prompt
        for component in self._prompt:
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
        # Ensure all components are ndarrays and safely clone them
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
        
        predefined_array = np.copy(self._predefined_array)  # Create a copy to avoid modifying the original
        predefined_attn_mask = np.copy(self._attn_mask)
        
        # Copy tokens into the preallocated array (up to its size limit)
        tokens_to_copy = min(len(prompt_tokens), len(predefined_array))
        predefined_array[:tokens_to_copy] = prompt_tokens[:tokens_to_copy]
        predefined_attn_mask[:tokens_to_copy-len(move_tokens)] = False # attend to all non-padding tokens except the target tokens we're trying to predict (the last four on this git branch)
        
        ## this code set the final entry in the array (after padding) to be the target value we wanted to predict. 
        ## I've decided it's better practice to make the final value the model wants to predict the token immediately following the last token in the prompt
        # # Set the final token to be best move token
        # predefined_array[-1] = move_tokens
        # self._attn_mask[-1] = False # attend to final token (best move)
        
        # set loss mask to be false for just the targe ttoken @ predefined_array[tokens_to_copy-1] (the last non padding entry in the array)
        predefined_loss_mask = np.copy(self._loss_mask)
        predefined_loss_mask[tokens_to_copy-len(move_tokens):tokens_to_copy] = False # calculate loss on only location of token we want to predict
        
        # return predefined array.
        return predefined_array, predefined_attn_mask, predefined_loss_mask
    
    # return the fen and move associated with the most recent element processed (hacky right now, so it won't account for batching properly)
    def getFen(self)->Tuple[str, str]:
        return self._last_fen, self._last_best_move
    
    def __len__(self) -> int:
        return len(self.data_source)

    # retrieve single sample from dataset
    def __getitem__(self, idx: int) -> torch.Tensor:
        element: bytes = self.data_source[idx]
        fen, move = constants.CODERS['behavioral_cloning'].decode(element)
        # setting these in here for debug purposes (temporary)
        self._last_fen = fen
        self._last_best_move = move
        sequence, attn, loss = self._tokenize(fen, move)
        
        # Convert to PyTorch tensors with correct dtypes
        sequence = torch.tensor(sequence, dtype=torch.long)
        attn = torch.tensor(attn, dtype=torch.bool)
        loss = torch.tensor(loss, dtype=torch.bool)
        return sequence, attn, loss, fen, move
    
class LlamaLoader:
    def __init__(self, 
                 config: dict,
                 tokenizer: PreTrainedTokenizer, 
                 split: str,
                 data_dir: str = "/workspace/searchless_chess/data",
                 data_source_name: str="behavioral_cloning_data.bag",
                 )->None:
        
        world_size = config["ddp_world_size"]
        rank = config["ddp_local_rank"]
        config = LanguageDataConfig(
            batch_size= config["batch_size"],
            tokenizer=tokenizer,
            tokenizer_save_path=config["out_dir"],
            shuffle=config["shuffle"],
            worker_count=config["worker_count"],  # 0 disables multiprocessing.
            num_return_buckets=config["num_return_buckets"],
            policy=config["policy"],
            split=split,
        )
        """Returns a data loader for chess from the config."""
        data_source = bagz.BagDataSource(
            os.path.join(
                data_dir, config.split, data_source_name
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

        dataset: Dataset = BagDataset(tokenizer=config.tokenizer, 
                                    tokenizer_save_path=config.tokenizer_save_path,
                                    data_source=data_source,
                                    prompt_components=None
                                    )
        if world_size > 1:
            sampler: DistributedSampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=config.shuffle,
            )
        else:
            sampler = RandomSampler(dataset) if config.shuffle else Sampler(dataset)
            
        self._loader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.worker_count,
            pin_memory=True,
            drop_last=True
        )
            
        self._dataset = dataset
        self._tokenizer = dataset._tokenizer
        self._loader_iter = iter(self._loader)

    # TODO: make it tokenize directly.
    def getTokenizer(self)->PreTrainedTokenizer:
        return self._tokenizer
    
    # # get a batch of fen samples. You should 
    # def getFen(self)->Tuple[str, str]:
    #     pass
    
    def __len__(self)->int:
        return len(self._loader)
    def __next__(self)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            seq, attn_mask, loss_mask, fen, move = next(self._loader_iter)
        except StopIteration:
            # TODO: log that you're starting over for loader_name 
            self._loader_iter = iter(self._loader)
            seq, attn_mask, loss_mask, fen, move = next(self._loader_iter)

        return seq, attn_mask, loss_mask, fen, move

            

