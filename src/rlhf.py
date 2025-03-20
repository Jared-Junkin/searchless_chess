from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, BitsAndBytesConfig
from torch.distributed import barrier, init_process_group, destroy_process_group
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import torch
from torch.utils.data import DataLoader
import math
from torch.nn import functional as F
import pandas as pd
import chess.engine
STOCKFISH_PATH = "/usr/games/stockfish"
import wandb
from language_data_loader import LlamaLoader
from config_language import LanguageDataConfig
from torch.optim import Adam
from contextlib import nullcontext
import yaml
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler
import time
import logging
from typing import Tuple, Callable, Any
import os
from hooks import *
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disabling Autotokenizer parallelism so we can do distributed training.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from yaml import CLoader as Loader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import traceback

from torch.utils.data import Dataset, RandomSampler
import chess
import utils
import csv
class RLHFDataSet(Dataset):
    def __init__(self,
                 filepath: str,
                 tokenizer: PreTrainedTokenizer,
                 prompt_components: Optional[List[str]]=None, 
                 bos_token: str = None,
                 pad_token: str = "<|padding|>",
                 eot_token: str = "<|endoftext|>") -> None:
        if not prompt_components:
            # default prompt
            # prompt_components = [
            #     "You are a chess grandmaster. This is the board position in FEN notation: ",   # fen comes after this                                                                                                        # legal moves comes after this
            #     "What is the best move? Best move: "                                       # best move comes after this.     
                
            # ]
            prompt_components = [
                "FEN: ",
                "Best Move: "                                                                                                   
            ]
            ###################
            # now we just need a piece of code to read in from datasource
            # this doens't use lazy loading, which might cause problems later
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                self.data_source = [(row[0], row[1])for row in reader] # get board position, top stockfish move
            f.close()
            ###################
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
        self.comma_space = tokenizer.encode(", ", add_special_tokens=False)
        self._board = chess.Board()
        self._tokenizer = tokenizer
        self._pretokenized_prompt = [tokenizer.encode(comp, add_special_tokens=False) for comp in prompt_components]
        
        
        # calculate buffer size
        self._SEQUENCE_LENGTH=77 + 6 + sum([len(prompt) for prompt in self._pretokenized_prompt]) # assuming we'll never have more than 50 legal moves, and each move takes up 5 pieces.
        
        
        # self._SEQUENCE_LENGTH=77+5+sum([len(prompt) for prompt in self._pretokenized_prompt])
        
        
        self._move_encodings = {}
        if bos_token:
            self.bos_token_id = [tokenizer.convert_tokens_to_ids(bos_token)]
            self._SEQUENCE_LENGTH += 1 # add 1 extra spot for bos token if it exists.
        else:
            self.bos_token_id=None
            
        self.eot_id = [tokenizer.convert_tokens_to_ids(eot_token)] # end of text token to put at end.

        pad_token_id = [tokenizer.convert_tokens_to_ids(pad_token)]                         # get token id of pad character
        self._predefined_array = np.full(                                                    # This array is going to be our prompty. pre-definining it so we don't have ot initialize each time
            (self._SEQUENCE_LENGTH,), pad_token_id, dtype=np.int32
        )
        self._loss_mask: np.ndarray = np.full(
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

    def __len__(self)->int:
        return len(self.data_source)
    def __getitem__(self, index):
        fen, move = self.data_source[index]
        # need to handle removing <endoftext from move> (oops)
        sequence = self._tokenize(fen, move)
        
        # Convert to PyTorch tensors with correct dtypes
        sequence = torch.tensor(sequence, dtype=torch.long)
        return sequence, fen
    def _tokenize(self, fen: str, move: str)->Tuple[np.ndarray, np.ndarray]:
        
        
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
        ## don't want to annotate with best move because we're manually doing autoregressive generation here in RLHF
        ## also don't need loss mask because loss mask only determines which tokens should contribute to average categorical crossentropy
        # if we're using accuracy rewards like in RLHF, we don't calculate reward at the token level at all. 
        
        # assemble our final prompt
        if self.bos_token_id:
            prompt_tokens = np.concatenate([
                self.bos_token_id,
                self._pretokenized_prompt[0],
                indices,
                self._pretokenized_prompt[1]
            ])
        else:
            raise NotImplementedError(f"SimpleBagDataset only implemented for Llama. self.bos_token_id required.")
        # define objects we're going to return 
        return prompt_tokens

# Ensure this runs within each DDP process
def create_optimizer(model: AutoModelForCausalLM, config: dict) -> Adam:
    # Extract optimizer configuration
    weight_decay = config['weight_decay']
    learning_rate = config['learning_rate']
    betas = (config['beta1'], config['beta2'])
    device_type = config['device_type']  # GPU/CPU setup
   
    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay
    )
    return optimizer

# this function will produce a batch of a single prompt and loss mask stacked batch_size
# times on top of each other. this is what you'd want to do for unsupervised RL with GRPO

# don't trust torch.expand here because expand does not allocate new memory,
# creates a new view on the existing tensor where a dimension of size one is expanded to a 
# larger size. I'm sure that would cause subtle and hard to debug gradient / tensor errors later

# write a custom collate function to stack a single-item sample batch_size times for GRPO
def custom_collate_fn(batch: List, batch_size: int)->tuple[torch.Tensor, torch.Tensor]:
    sequence, fen = batch[0] # dataloader will always just see a single batch size
    sequences = sequence.unsqueeze(0).repeat(batch_size, 1).contiguous()
    return sequences, fen # return shapes [(batch_size x seq_length), (batch_size x seq_length), (fen_string_length x 1)]
    
# def autoregressive_forward_pass(model:)

def RLHF(config: dict)-> None:
    # print("Hello, world")
    logger = setupLogger(config=config)
    for key, value in config.items():
        logger.info(f"{key}: {value}\n")
    max_iters = config["max_iters"]
    out_dir = config["out_dir"]
    decay_lr = config["decay_lr"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    log_interval = config["log_interval"]
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}['bfloat16']
    ctx = nullcontext() if config['device_type'] == 'cpu' else torch.amp.autocast(device_type=config['device_type'], dtype=ptdtype)
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    # load model, tokenizer, dataloader
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_load_dir"])
    # set up ddp params
    config, device, ddp, ddp_local_rank, master_process = set_ddp_params(config=config)
    
    train_dataset = RLHFDataSet(filepath = config['prompts_file'], 
                                tokenizer=tokenizer,
                                pad_token= "<|pad|>",
                                bos_token="<|begin_of_text|>",
                                eot_token= "<|end_of_text|>")
    print("done creating dataset object")
    print(len(train_dataset))
    
    model = AutoModelForCausalLM.from_pretrained(config["model_load_dir"])
    # model.gradient_checkpointing_enable()
    model.train()
    model.to(device)
    
    # not parallelizing this just yet

    # how do I set up dataloader and sampler to correctly sample from dataset?
    q_batch_size = config['batch_size']
    sampler = RandomSampler(train_dataset)
    
    # this is how you integrate a custom collate function with a dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=sampler,
        batch_size=1, # this is q size for RLHF (# of samples)
        collate_fn=lambda x: custom_collate_fn(batch=x, batch_size=q_batch_size), 
        pin_memory=True, # what does 
        drop_last=True
    )
    train_loader=iter(train_loader)
    # make dataloader and sampler (this part shouldn't be very hard)
    
    sequence, fen_state = next(train_loader)
    sequence = sequence.to(device)
    # loss_mask = loss_mask.to(device)
    org_seq_length = sequence.shape[1]

    # now set up main train loop
    # stockfish will be our reward model
    helper = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    eot_token_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    for i in range(100):
        # for micro_step in range(gradient_accumulation_steps):
            # ddp. but not going to use yet
            # if ddp:
            #     model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            trajectory_probs = torch.ones(size=(sequence.shape[0], 1)).to(device)
            eot_array = torch.zeros_like(trajectory_probs, dtype=torch.bool).to(device)
            for _ in range(6):
                # generate output from current model
                
                # generate output
                outputs = model(input_ids=sequence, output_attentions=False)
                
                # get logits from output
                logits = outputs.logits[:, -1, :]
                
                
                # get the probs from the logits. implicitly sampling with temperature = 1.0 bc/ no temp above this
                probs = F.softmax(logits, dim=-1)
                # sample distribution
                top_tokens = torch.tensor(torch.multinomial(probs, num_samples=1, replacement=True))

                # insert new token at the end of hte sequence (insert a column into the EOS_index-th column in our existing tensor)
                sequence = torch.cat((sequence, top_tokens), dim=-1)
                
                # update trajectory only if we haven't hit EOT yet in this arc
                mask = ~eot_array.squeeze(1)
                trajectory_probs[mask] *= probs.gather(1, top_tokens)[mask]
                
                # update mask to reflect where we've hit eot
                eot_array[top_tokens == eot_token_id] = True
                
                # while eot == flase
                    # update model's trajectory 
                

                # generate output from reference model pi_ref
        
        print(f"sequence: {sequence}")
        # this will filter all answers dynamically 
        target_tokens = sequence[:, org_seq_length:torch.where(sequence == 128001)[1].max().item()] # this will return max EOS index across all rows. 
        
        # stockfish as reward function. calculate reward = 1 if model found mate, 0 if it didn't.
        board = chess.Board(fen=fen_state)
        info = helper.analyse(board, chess.engine.Limit(depth=10), multipv=5)
        rewards=torch.zeros(size=(1, q_batch_size))
        # rewards = []
        for i in range(target_tokens.shape[0]):
            move_str = tokenizer.decode(target_tokens[i][target_tokens[i] != 128001]).strip()
            
            try:
                move = chess.Move.from_uci(move_str)
            except ValueError:
                continue
            temp_board = board.copy()
            temp_board.push(move)
            if temp_board.is_checkmate():
                rewards[0][1]=1
        print(rewards)
        
        # okay, so now you have your reward function. time to backpropagate. 
        mu = torch.mean(rewards)
        var = torch.var(rewards)
        
        sequence, fen_state = next(train_loader)
        # calculate group relative policy optimization backprop equation.
            # store reference policy
            # you'll have to calculate p(pi_ref for each response i)
        # backpropogate
        
        
    return

if __name__ == "__main__":
    config_file = "/workspace/searchless_chess/src/rlhf.yaml" # removing legal moves. hoping this allows me to 10x batch size, leading to more accurate gradient signal and better model.
    

    with open(config_file, "r") as stream:
        config = yaml.load(stream=stream, Loader=Loader)
    # mates_df = pd.read_csv(config["prompts_file"])
    # print(mates_df)
    run_with_error_handling(RLHF, config=config, log_path=config["log_path"])