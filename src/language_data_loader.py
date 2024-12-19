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
    def __init__(self, 
                 data_source: Any, 
                 tokenizer: PreTrainedTokenizer, 
                 tokenizer_save_path: str, 
                 prompt_components: Optional[List[str]]=None, 
                 pad_token: str = "<|padding|>",
                 eot_token: str = "<|endoftext|>") -> None:
        if not prompt_components:
            # default prompt
            # prompt_components = [
            #     "You are a chess grandmaster. This is the board position in FEN notation: ",   # fen comes after this                                                                                                        # legal moves comes after this
            #     "What is the best move? Best move: "                                       # best move comes after this.     
                
            # ]
            prompt_components = [
                "You are a chess grandmaster. This is the board position in FEN notation: ",   # fen comes after this
                "The legal moves are: ",                                                                                                            # legal moves comes after this
                "Which of these is the best move? Best move: "                                       # best move comes after this.     
                
            ]
            
        self._CHARACTERS = [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
            'a',
            'b',
            'c',
            'd',
            'e',
            'f',
            'g',
            'h',
            'p',
            'n',
            'r',
            'k',
            'q',
            'P',
            'B',
            'N',
            'R',
            'Q',
            'K',
            'w',
            '.',
            ' '
        ]
        self.encodings = {key: tokenizer.convert_tokens_to_ids(key) for key in self._CHARACTERS}
        self.encodings[pad_token] = tokenizer.convert_tokens_to_ids(pad_token)
        self.comma_space = tokenizer.encode(", ")
        self._board = chess.Board()
        self._tokenizer = tokenizer
        self._pretokenized_prompt = [tokenizer.encode(comp) for comp in prompt_components]
        
        
        # calculate buffer size
        self._SEQUENCE_LENGTH=77 + (6*60) + sum([len(prompt) for prompt in self._pretokenized_prompt]) # assuming we'll never have more than 50 legal moves, and each move takes up 5 pieces.
        
        
        # self._SEQUENCE_LENGTH=77+5+sum([len(prompt) for prompt in self._pretokenized_prompt])
        
        
        self.data_source = data_source # required data source object we'll sample from
        self._move_encodings = {}
        self.eot_id = [tokenizer.convert_tokens_to_ids(eot_token)] # end of text token to put at end.

        pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)                         # get token id of pad character
        self._predefined_array = np.full(                                                    # This array is going to be our prompty. pre-definining it so we don't have ot initialize each time
            (self._SEQUENCE_LENGTH,), pad_token_id, dtype=np.int32
        )
        self._loss_mask: np.ndarray = np.full(
            shape=(self._SEQUENCE_LENGTH,),
            fill_value=False,
            dtype=bool
        )
        
        self._attn_mask: np.ndarray = np.full(
            shape=(self._SEQUENCE_LENGTH,),
            fill_value=False,
            dtype=bool
        )
        
        self._last_fen = None
        self._last_best_move = None
        all_moves = utils._compute_all_possible_actions()
        all_moves = list(all_moves[0].keys())
        self.all_move_encodings = {}
        for move in all_moves:
            encoding = []
            for char in move: 
                encoding.append(self.encodings[char])
            self.all_move_encodings[move] = encoding
        


        
    def _tokenize(self, fen: str, move: SyntaxError)->Tuple[np.ndarray, np.ndarray]:

        
        spaces_characters = frozenset({'1', '2', '3', '4', '5', '6', '7', '8'})

        # extract relevant board informatino from fen
        board, side, castling, en_passant, halfmoves_last, fullmoves = fen.split(' ')
        board = board.replace('/', '')
        board = side + board

        indices = list()



        # replace integer representations of empty space with dots (5 -> .....)
        for char in board:
            if char in spaces_characters:
                indices.extend(int(char) * [self.encodings['.']])
            else:
                indices.append(self.encodings[char])
        # if no one can castle, make castling ....
        if castling == '-':
            indices.extend(4 * [self.encodings['.']])
        # otherwise, pad castling to be four characters exactly.
        else:
            for char in castling:
                indices.append(self.encodings[char])
            # Padding castling to have exactly 4 characters.
            if len(castling) < 4:
                indices.extend((4 - len(castling)) * [self.encodings['.']])

        # if en passant isn't possible, make it .. otherwise, it will be e3 (for example). the square where en passant is possible.
        if en_passant == '-':
            indices.extend(2 * [self.encodings['.']])
        else:
            # En passant is a square like 'e3'.
            for char in en_passant:
                indices.append(self.encodings[char])

        # Three digits for halfmoves (since last capture) is enough since the game
        # ends at 50. AI doesn't care about halfmoves. just noise
        halfmoves_last += '.' * (3 - len(halfmoves_last))
        indices.extend([self.encodings[x] for x in halfmoves_last])

        # AI also doesn't care about fullmoves
        # Three digits for full moves is enough (no game lasts longer than 999
        # moves).
        fullmoves += '.' * (3 - len(fullmoves))
        indices.extend([self.encodings[x] for x in fullmoves])
        
        # encode best move. Encoding explicitly on a letter-by-letter basis.
        best_move=list()
        for char in move:
            best_move.append(self.encodings[char])
            # set up loss mask 

        
        # add in legal moves to prompt.
        self._board.set_fen(fen=fen)
        legal_tokens = []
        legal_moves = [str(m) for m in self._board.legal_moves]
        for move in legal_moves:
            legal_tokens.extend(self.all_move_encodings[move])
            legal_tokens.extend(self.comma_space) # add ", " to help the LLM with readability.
            
        # assemble our final prompt
        prompt_tokens = np.concatenate([
            self._pretokenized_prompt[0],
            indices,
            self._pretokenized_prompt[1],
            legal_tokens,
            self._pretokenized_prompt[2],
            best_move,
            self.eot_id
        ])
        
        # define objects we're going to return 
        predefined_array = np.copy(self._predefined_array)
        attn_mask = np.copy(self._attn_mask)
        loss_mask = np.copy(self._loss_mask)
        
        # copy prompt into predefined array
        tokens_to_copy = min(len(prompt_tokens), len(predefined_array))
        predefined_array[:tokens_to_copy] = prompt_tokens[:tokens_to_copy]
        
        # set the model only to attend to non-padding tokens.
        attn_mask[:tokens_to_copy] = True

        
        # make sure loss mask aligns with tokens we want to predict
        loss_mask[tokens_to_copy-len(best_move)-1:tokens_to_copy] = True # calculate loss on only location of token we want to predict (-1 in there because we want it to predict <|endoftext|>)
        

        assert len(predefined_array) == self._SEQUENCE_LENGTH


        return predefined_array, attn_mask, loss_mask
    
    
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
        return sequence, attn
        # return sequence, attn, loss

def build_data_loader(training_config: dict,
                      tokenizer: PreTrainedTokenizer,
                      split: str,
                      data_dir: str = "/workspace/searchless_chess/data",
                 data_source_name: str="behavioral_cloning_data.bag",
                 )->DataLoader:
    world_size = training_config["ddp_world_size"]
    rank = training_config["ddp_local_rank"]
    config = LanguageDataConfig(
        batch_size= training_config["batch_size"],
        tokenizer=tokenizer,
        tokenizer_save_path=training_config["out_dir"],
        shuffle=training_config["shuffle"],
        worker_count=training_config["worker_count"],  # 0 disables multiprocessing.
        num_return_buckets=training_config["num_return_buckets"],
        policy=training_config["policy"],
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
                                pad_token=training_config["pad_token"],
                                eot_token=training_config["eot_token"],
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
        
    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.worker_count,
        pin_memory=True,
        drop_last=True
    )




class LlamaLoader:
    
    def __init__(self, 
                 training_config: dict,
                 tokenizer: PreTrainedTokenizer, 
                 split: str,
                 data_dir: str = "/workspace/searchless_chess/data",
                 data_source_name: str="behavioral_cloning_data.bag",
                 )->None:
        
        world_size = training_config["ddp_world_size"]
        rank = training_config["ddp_local_rank"]
        config = LanguageDataConfig(
            batch_size= training_config["batch_size"],
            tokenizer=tokenizer,
            tokenizer_save_path=training_config["out_dir"],
            shuffle=training_config["shuffle"],
            worker_count=training_config["worker_count"],  # 0 disables multiprocessing.
            num_return_buckets=training_config["num_return_buckets"],
            policy=training_config["policy"],
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
                                    pad_token=training_config["pad_token"],
                                    eot_token=training_config["eot_token"],
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
            seq, attn_mask, loss_mask = next(self._loader_iter)
        except StopIteration:
            # TODO: log that you're starting over for loader_name 
            self._loader_iter = iter(self._loader)
            seq, attn_mask, loss_mask= next(self._loader_iter)

        return seq, attn_mask, loss_mask

            

