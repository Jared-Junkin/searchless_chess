from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, BitsAndBytesConfig
from torch.distributed import barrier, init_process_group, destroy_process_group
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import torch
from torch.utils.data import DataLoader
import math
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

def plot_discrete_pdf(values, savefile: str = "EDDPDF.png")->None:
    """
    Plots a discrete probability density function (PDF) from a 6-length float array.
    
    Parameters:
        values (list or np.array): A list or numpy array of length 6 containing float values.
    """
    if len(values) != 6:
        raise ValueError("Input array must have exactly 6 elements.")
    
    indices = np.arange(len(values))
    
    plt.bar(indices, values, tick_label=[f'X{i+1}' for i in range(len(values))], alpha=0.7, color='skyblue', edgecolor='black')
    
    # Print value on top of each bar
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Discrete Variable')
    plt.ylabel('Probability')
    plt.title('Discrete Probability Density Function')
    plt.ylim(0, max(values) + 0.1 * max(values))  # Add some padding on top
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(savefile)
# edit distance is the minimum number of operations needed to transpose one iterable into another. 
# this calculates the word-level edit distsance between two strings (it first splits them along whitespace, \t, and \n).
def editDistance(word1: List[int], word2: List[int])->int:
    
    ## operaing directly on token space instead.
    
    # split strings apart (should modify this to also split along \t and \n)
    # word1=word1.split(" ")
    # word2=word2.split(" ")
    
    #  dp table is of length len(word2)+1 x len(word1)+1
        # the +1 is added because we want the base case sub problem to be an empty string, so dp[0][:] and dp[:][0] should represent "" not word1[0] or word2[0]
    dp_table = [[0]*(len(word1)+1) for _ in range(len(word2)+1)]

    # edit distance between "" and word1[:i] is i. same with word2. these are our base cases
    for i in range(len(word1)+1):
        dp_table[0][i]=i
    for i in range(len(word2)+1):
        dp_table[i][0]=i

    # "recursve" / inductive case. 
    # recurrence relation:
        # edit distance[word2[:i], word1[:j]] is
            # edit_distance(word2[:i-1], word1[j-1]) + one insertion, deletion, or substitution operation made to one word. 
    for i in range(1, len(word2)+1):
        for j in range(1, len(word1)+1):
            if word2[i-1]==word1[j-1]:
                dp_table[i][j]=dp_table[i-1][j-1]
            else:
                dp_table[i][j]=min(dp_table[i][j-1]+1,      # delete element from word1
                                   dp_table[i-1][j]+1,      # delete element from word2 (call this insertion)
                                   dp_table[i-1][j-1]+1)    # delete element from both (substitution)
    return dp_table[len(word2)][len(word1)]
    
## load in model and tokenizer
## load in datasets
## generate moves from datasets and calculate the edit distance distribution
## make sure device == cpu
def sampleEditBatches(model: AutoModelForCausalLM, 
                    t_iter: LlamaLoader,
                    pad_token_id: int,
                    n_batches: int = 10,
                    n_stochastic_samples: int = 20,
                    len_response: int = 5,
                    device: str = 'cpu')->List[int]:

    edit_dists = [0]*len_response
    for _ in range(n_batches):
        seq, loss = next(t_iter)
        seq = seq.to(device)
        loss=loss.to(device)
        print(seq.shape[0])
        for i in range(16): # 0th dimension is batch dimension
            cutoff_train = int(torch.where(loss[i])[0][0])
            outputs_train = model.generate(input_ids=seq[i][:cutoff_train].unsqueeze(0), max_length=cutoff_train+7, pad_token_id=pad_token_id, do_sample=False) # do_sample=False for deterministic decoding.
            outputs_train = outputs_train[0][torch.where(loss[i])[0][:]]
            print("##################################")
            for _ in range(n_stochastic_samples):
                output_edit = model.generate(input_ids=seq[i][:cutoff_train].unsqueeze(0), max_length=cutoff_train+7, pad_token_id=pad_token_id)
                indices = torch.where(loss[i])[0]
                indices = indices[indices<output_edit.shape[1]]
                output_edit=output_edit[0][indices]
                
                # output_edit = output_edit[0][torch.where(loss[i])[0][:]]
                EDD = editDistance(outputs_train, output_edit)
                edit_dists[EDD]+= 1
                print(f"0 temp is {outputs_train}, sample is {output_edit}, EDD is {EDD}")
            print("##################################")
    total_samples_processed = n_stochastic_samples*n_batches*seq.shape[0]
    edit_dists = [edit_dists[i]/total_samples_processed for i in range(len(edit_dists))]
    return edit_dists
        
def edit_distance_main(config: dict)->None:

    # you'd have to sample it, right? I think do_sample = False makes it incorrect according to EDD


    device = config['device']
    ## load in model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(config["model_load_dir"])
    model.eval() # not training anything
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_load_dir"])
    pad_token_id = tokenizer.convert_tokens_to_ids(config['pad_token'])
    
    ## load in datasets
    config, _, _, _, _ = set_ddp_params(config=config)
    train_iter = LlamaLoader(training_config=config, tokenizer=tokenizer, split="train")
    test_iter = LlamaLoader(training_config=config, tokenizer=tokenizer, split="test")
    
    dist_array = sampleEditBatches(model=model, 
                                   pad_token_id=pad_token_id, 
                                   t_iter=train_iter, 
                                   n_batches=50, 
                                   n_stochastic_samples=20)
    with open("EDDPDF.txt", 'a') as f:
        f.write(" ".join(map(str, dist_array)))
        f.write("\n#########################\n")
    f.close()
    print(f"edit distsance distribution is {dist_array}")
    dist_array_test = sampleEditBatches(model=model, 
                                   pad_token_id=pad_token_id, 
                                   t_iter=test_iter, 
                                   n_batches=50, 
                                   n_stochastic_samples=20)
    with open("EDDPDF.txt", 'a') as f:
        f.write(" ".join(map(str, dist_array_test)))
    f.close()
    return 
    
    

# if __name__ == "__main__":
#     s1 = "this is a test"
#     s2 = "this is test"
#     print(editDistance(s1, s2))

if __name__ == "__main__":
    
    config_file = "/workspace/searchless_chess/src/confi_llama_smallPrompt.yaml" # removing legal moves. hoping this allows me to 10x batch size, leading to more accurate gradient signal and better model.
    

    with open(config_file, "r") as stream:
        config = yaml.load(stream=stream, Loader=Loader)
    config['device']='cpu'# hardcoding for now
    config['model_load_dir'] = '/workspace/searchless_chess/src/Llama/ckpts_smallPrompt/ckpt95000'
    config['tokenizer_load_dir'] = '/workspace/searchless_chess/src/Llama/ckpts_smallPrompt/ckpt95000'
    run_with_error_handling(edit_distance_main, config=config)