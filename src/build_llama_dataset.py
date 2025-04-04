# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""An example training script."""

from collections.abc import Sequence

from absl import app
from absl import flags
import time
import config as config_lib
import jared_data_loader
import metrics_evaluator
import tokenizer
import training
import transformer
import utils

import chess
from jared_data_loader import ConvertBehavioralCloningDataToLanguage

_POLICY = flags.DEFINE_enum(
    'policy',
    'behavioral_cloning',
    config_lib.POLICY_TYPES,
    'The policy used to play moves with the model.',
)
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_legal_moves_histogram(
    data_iter, 
    n_batches: int, 
    bin_width: int = 10, 
    save_path: str = "./num_legal_moves_hist.png"
):
    # Initialize bin counters using defaultdict for automatic zero initialization
    bin_counters = defaultdict(int)
    
    for i in range(n_batches):
      if i % 1000 == 0:
        print(f"starting batch {i}")
        try:
            _, _, _, _, num_legal_moves = next(data_iter)
            
            # Increment the appropriate bin counter for each value
            for value in num_legal_moves:
                bin_index = (value // bin_width) * bin_width  # Determine bin range start
                bin_counters[bin_index] += 1
        except StopIteration:
            print("Data iterator exhausted before reaching the desired number of batches.")
            break

    # Prepare data for plotting
    bins = sorted(bin_counters.keys())
    counts = [bin_counters[b] for b in bins]

    # Plot histogram
    plt.bar(bins, counts, width=bin_width, align='edge', edgecolor='black')
    plt.xlabel('Number of Legal Moves (binned)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Number of Legal Moves')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()

    print(f"Histogram saved to {save_path}.")

# Example usage:
# plot_legal_moves_histogram(data_iter, n_batches=10, bin_width=10, save_path="./num_legal_moves_hist.png")

def test_throughput(train_config: config_lib.DataConfig)->None:
  data_iter = jared_data_loader.build_data_loader_language(config=train_config.data).__iter__()
  data_iter_normal = jared_data_loader.build_data_loader_parallel(config=train_config.data, rank=0, world_size=1).__iter__()
  fen, move, loss_mask, legal_moves,_  = next(data_iter)
  print(f"FEN is {len(fen)}\nBest move is: {len(move)}\nLegal Moves are: {len(legal_moves)}") 
  
  # decode and shift fen. (although what's even the point of doing this. I should be able to just get the plain fen out further upstream.)
  # fen, move, loss_mask, legal_moves 
  for i in range(50):
    tot_new = 0
    tot_old = 0
    tot_iters = 0
    if i > 20:
      start = time.time()
      fen, move, loss_mask, legal_moves,_  = next(data_iter)
      end = time.time()
      delta_new = end - start
      start = time.time()
      sequence= next(data_iter_normal)
      end=time.time()
      delta_2 = end - start
      tot_new += delta_new
      tot_old += delta_2
      tot_iters+=1
    else:
        fen, move, loss_mask, legal_moves,_  = next(data_iter)
        sequence= next(data_iter_normal)
        
  print(f"avg load time for language (with legalmoves): {tot_new/max(tot_iters, 1)}, avg load time for decoder only architecture (just fen): {tot_old/max(tot_iters,1)}")
  # this code is to see if my function has significantly decreased throughput. it is now 15% slower with the chess board
  # output: avg load time for language (with legalmoves): 0.08965802192687988, avg load time for decoder only architecture (just fen): 0.07647871971130371 (size of predefined moves is 512)
  # output: avg load time for language (with legalmoves): 0.053000688552856445, avg load time for decoder only architecture (just fen): 0.07760214805603027 (sizeof predefined moves is 128)
  
def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  policy: config_lib.PolicyType = _POLICY.value  # pytype: disable=annotation-type-mismatch
  num_return_buckets = 128

  match policy:
    case 'action_value':
      output_size = num_return_buckets
    case 'behavioral_cloning':
      output_size = utils.NUM_ACTIONS
    case 'state_value':
      output_size = num_return_buckets

  predictor_config = transformer.TransformerConfig(
      vocab_size=utils.NUM_ACTIONS,
      output_size=output_size,
      pos_encodings=transformer.PositionalEncodings.LEARNED,
      max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
      num_heads=4,
      num_layers=4,
      embedding_dim=64,
      apply_post_ln=True,
      apply_qk_layernorm=False,
      use_causal_mask=False,
  )
  train_config = config_lib.TrainConfig(
      learning_rate=1e-4,
      data=config_lib.DataConfig(
          batch_size=256,
          shuffle=True,
          worker_count=0,  # 0 disables multiprocessing.
          num_return_buckets=num_return_buckets,
          policy=policy,
          split='train',
      ),
      log_frequency=1,
      num_steps=20,
      ckpt_frequency=5,
      save_frequency=10,
  )
  data_iter = jared_data_loader.build_data_loader_language(config=train_config.data).__iter__()
  fen, move, loss_mask, legal_moves, num_legal_moves  = next(data_iter)
  print(num_legal_moves)
  test_throughput(train_config=train_config)
  # plot_legal_moves_histogram(
  #   data_iter=data_iter, 
  #   n_batches=100000, 
  #   bin_width=5, 
  #   save_path="./Llama/num_legal_moves_hist.png"
  # )
  
  
if __name__ == '__main__':
  app.run(main)
