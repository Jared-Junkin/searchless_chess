
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

## nanoGPTPlayer imports
from typing import Optional, Callable
from contextlib import nullcontext
import os
import tiktoken
import re
import numpy as np
import pickle
from typing import List
import chess
import chess.engine

STOCKFISH_PATH = "/usr/games/stockfish"
DRAW_FILE = "draws.txt"

from transformers import AutoModelForCausalLM, AutoTokenizer
class PythiaPlayer:
    def __init__(self, 
                 tokenizer_config_path: str, 
                 model_config_path: str, 
                 draws_okay: bool = False, 
                 prompt_components: Optional[List[str]]=None
    ) -> None:
        # this MUST be the same prompt the model was fine-tuned on (I assume. I actually haven't tested how resilient it is to different prompts)
        if not prompt_components:
            # default prompt
            prompt_components = [
                "You are a chess grandmaster. This is the board position in FEN notation: ",   # fen comes after this
                "The legal moves are: ",                                                                                                            # legal moves comes after this
                "Which of these is the best move? Best move: "                                       # best move comes after this.     
                
            ]
        # 
        if len(prompt_components)!=3:
            raise ValueError(f"Expected prompt_components to have a length of 3, but got len {len(prompt_components)}.")
        
        self.model = AutoModelForCausalLM.from_pretrained(model_config_path)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_config_path)
        self.helper = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        self.draws_okay = draws_okay
        self._pretokenized_prompt = []
        for component in prompt_components:
            token_ids = self._tokenizer(component, add_special_tokens=False)["input_ids"]
            self._pretokenized_prompt.append(np.array(token_ids, dtype=np.int32))
            
        static_prompt_length = sum(len(comp) for comp in self._pretokenized_prompt) 
        max_encoding_length = 6     # hard coding for now. what it was for training
        dynammic_prompt_length = max_encoding_length * 80 # (empircally, there aren't any board states in dataset with more legal moves than this)                                 # max length needed for dynammic prompt (there will never by more than 128 legal moves)
        total_prompt_length = static_prompt_length + dynammic_prompt_length + 1  
        print(f"total prompt length is {total_prompt_length}")
        pad_token_id = self._tokenizer.convert_tokens_to_ids("<|pad|>")                         # get token id of pad character
        self._predefined_array = np.full(                                                    # This array is going to be our prompty. pre-definining it so we don't have ot initialize each time
            (total_prompt_length,), pad_token_id, dtype=np.int32
        )
        self._attn_mask: torch.Tensor = torch.full(
            size=(total_prompt_length,),
            fill_value=True,
            dtype=bool,
        )
    # get a move from the agent and return it in the proper format
    def get_move(self, board: chess.Board, game_state: str, temperature: float) -> str:
        if self.draws_okay:
            completion = self.get_response(fen=board.fen(), temperature=temperature)
            return completion
        else:
            # if victory is certain and the bot is going to draw because of FEN, play top stockfish move
            # note that if min_wins (out of 1000 estimated according to stockfish) is less than 990, bot is on its own
            # so this won't catch forced draws. Just draws where it's almost guaranteed to win
            if board.is_repetition(2):
                info = self.helper.analyse(board, chess.engine.Limit(depth=10), multipv=5)
                best_move_san = info[0]['pv'][0].uci()
                min_wins = float('Inf')
                for variant in range(len(info)):
                    winsDrawsLosses = info[variant]['score'].wdl()
                    min_wins = min(min_wins, winsDrawsLosses.relative.wins)

                print(f"We're about to draw. Win prob for worst move is {min_wins/1000}")
                # if stockfish says we have >99% chance of winning on all top 5 moves, then return top move
                # put the opponent out of their misery
                if min_wins > 990:
                    return best_move_san
                
            completion = self.get_response(board=board, temperature=temperature)
            return completion # just removing this because rightnow my decoder can only handle the most likely move.
        
    def get_response(self, board: chess.Board, temperature: float) -> str:
        legal_moves = [str(m) for m in board.legal_moves]
        info = self.helper.analyse(board, chess.engine.Limit(depth=10), multipv=len(legal_moves))
        best_move = info[0]['pv'][0].uci()
        move_tokens = np.array(self._tokenizer(best_move, add_special_tokens=False)['input_ids'], dtype=np.int32)
        
        fen = board.fen()
        fen_tokens = np.array(self._tokenizer(fen, add_special_tokens=False)['input_ids'], dtype=np.int32)
        legal_move_tokens = np.array([self._tokenizer(move + ", ", add_special_tokens=False)["input_ids"] for move in legal_moves])
        # one way to know for sure that it works. the model cannot cheat now.
        legal_move_tokens = np.concatenate(legal_move_tokens, axis=0)
        prompt_tokens = np.concatenate(
            [
                self._pretokenized_prompt[0],
                fen_tokens,
                self._pretokenized_prompt[1],
                legal_move_tokens,
                self._pretokenized_prompt[2],
                np.zeros_like(move_tokens)
            ]
        )
        predefined_array = np.copy(self._predefined_array)
        tokens_to_copy = min(len(prompt_tokens), len(predefined_array))
        predefined_array[:tokens_to_copy] = prompt_tokens[:tokens_to_copy]
        predefined_attn_mask = np.copy(self._attn_mask)
        predefined_attn_mask[:tokens_to_copy] = False
        
        # generate best move
        predefined_array = torch.tensor(predefined_array).unsqueeze(0)
        predefined_attn_mask = torch.tensor(predefined_attn_mask).unsqueeze(0)
        generated_outputs = self.model.generate(
            input_ids=predefined_array,
            attention_mask=predefined_attn_mask,
            max_length=predefined_array.size(1)+5,  # Ensure the generated sequence matches the input length
        )
        print(self._tokenizer.batch_decode(generated_outputs))
        response = generated_outputs[tokens_to_copy: tokens_to_copy+4]
        
        top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
        start_ids = self.encode(fen)

        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
        with torch.no_grad():
            y = self.model.generate(x, 1, temperature=temperature, top_k=top_k)
            model_response = self.decode(y[0][-1].tolist()) # this would give you 'e2e4' for example. 
            # but I'm not sure the generate function is really working correctly, because it seems like it can only really generate 1 sample. so I think I'm not sampling from the probability distribution correclt.y
        return model_response
    
    
class NanoGptPlayer:
    def __init__(
        self,
        model_name: str,
        tokenizer: Callable[[str], np.ndarray],
        decoder: Callable[[str],np.ndarray],
        activation_name: Optional[str] = None,
        activation_coefficient: Optional[float] = None,
        base_dir: str = "jared_models/nanogpt/",
        model_path: str = None,
        draws_okay: bool = False
    ):
        self.model_name = model_name
        # in here to prevent draws when M1 is on the board as per deepmind paper
        self.helper = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        # -----------------------------------------------------------------------------
        init_from = "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
        out_dir = "out"  # ignored if init_from is not 'resume'
        input_dir = "addition"
        test_name = "test.txt"
        start = "12+44="  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
        num_samples = 1  # number of samples to draw
        max_new_tokens = 6  # number of tokens generated in each sample
        self.draws_okay = draws_okay
        temperature = (
            0.01  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        )
        top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
        seed = 1337
        device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        # device = "cpu"
        dtype = "float16"  # 'float32' or 'bfloat16' or 'float16'
        compile = False  # use PyTorch 2.0 to compile the model to be faster
        self.model_path = model_path
        exec(
            open(f"{base_dir}configurator.py").read()
        )  # overrides from command line or config file
        # -----------------------------------------------------------------------------

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        # model
        if init_from == "resume":
            # init from a model saved in a specific directory
            # ckpt_path = os.path.join(BASE_DIR, out_dir, self.model_name)
            if self.model_path: 
                ckpt_path = os.path.join(self.model_path, self.model_name)
            else:
                ckpt_path = f"nanogpt/out/{self.model_name}" # old
            # ckpt_path = f"out/{self.model_name}" # new
            checkpoint = torch.load(ckpt_path, map_location=device)
            gptconf = GPTConfig(**checkpoint["model_args"])

            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

            if activation_name is not None:
                state_dict, gptconf = add_activation_bias_to_state_dict(
                    state_dict, device, activation_name, gptconf, activation_coefficient
                )
            model = GPT(gptconf)
            model.load_state_dict(state_dict)
            # model = torch.compile(model)
        elif init_from.startswith("gpt2"):
            # init from a given GPT-2 model
            model = GPT.from_pretrained(init_from, dict(dropout=0.0))

        model.eval()
        model.to(device)
        if compile:
            model = torch.compile(model)  # requires PyTorch 2.0 (optional)

        
        # get these from deepmind.
        self.encode = tokenizer
        self.decode = decoder
        self.model = model
        self.ctx = ctx
        self.device = device

    def get_nanogpt_response(self, fen: str, temperature: float) -> str:
        
        top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
        start_ids = self.encode(fen)

        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
        with torch.no_grad():
            with self.ctx:
                y = self.model.generate(x, 1, temperature=temperature, top_k=top_k)
                model_response = self.decode(y[0][-1].tolist()) # this would give you 'e2e4' for example. 
                # but I'm not sure the generate function is really working correctly, because it seems like it can only really generate 1 sample. so I think I'm not sampling from the probability distribution correclt.y
        return model_response

    # def get_move_from_response(self, response: str) -> str:
    #     # Parse the response to get only the first move
    #     moves = response.split()
    #     if not moves:
    #         return ""
    #     first_move = moves[0]

    #     return first_move

    def get_move(self, board: chess.Board, game_state: str, temperature: float) -> str:
        # for RLHF, we'll want the agent to be able to play draws
        if self.draws_okay:
            completion = self.get_nanogpt_response(fen=board.fen(), temperature=temperature)
            return completion
        else:
            # if victory is certain and the bot is going to draw because of FEN, play top stockfish move
            # note that if min_wins (out of 1000 estimated according to stockfish) is less than 990, bot is on its own
            # so this won't catch forced draws. Just draws where it's almost guaranteed to win
            if board.is_repetition(2):
                info = self.helper.analyse(board, chess.engine.Limit(depth=10), multipv=5)
                best_move_san = info[0]['pv'][0].uci()
                min_wins = float('Inf')
                with open(DRAW_FILE, 'a') as f:
                    for variant in range(len(info)):
                        winsDrawsLosses = info[variant]['score'].wdl()
                        min_wins = min(min_wins, winsDrawsLosses.relative.wins)
                        # f.write(f"{info[variant]['score'].relative}\n")
                        if winsDrawsLosses.relative.wins > 0:
                            f.write(f"Eval: {info[variant]['score'].relative}, Win Prob: {winsDrawsLosses.relative.wins/1000}\n")
                        else: 
                            f.write(f"Eval: {info[variant]['score'].relative}, Win Prob: {winsDrawsLosses.relative.wins}\n")
                    f.close()
                print(f"We're about to draw. Win prob for worst move is {min_wins/1000}")
                # if stockfish says we have >99% chance of winning on all top 5 moves, then return top move
                # put the opponent out of their misery
                if min_wins > 990:
                    return best_move_san
                
            completion = self.get_nanogpt_response(fen=board.fen(), temperature=temperature)
            return completion # just removing this because rightnow my decoder can only handle the most likely move.

    def get_config(self) -> dict:
        return {"model": self.model_name}