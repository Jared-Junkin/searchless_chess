from language_data_loader import LlamaLoader
from typing import Tuple
from transformers import AutoTokenizer
import chess
import matplotlib.pyplot as plt
import yaml
from torch.distributed import destroy_process_group
from hooks import set_ddp_params
import pandas as pd
from yaml import CLoader as Loader
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disabling Autotokenizer parallelism so we can do distributed training.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Assuming LlamaLoader and set_ddp_params are implemented elsewhere in the codebase

def plot_num_moves(config: dict, 
                   saveFilePath: str, 
                   num_batches: int = 10000, 
                   binwidth: int = 1, 
                   cutoff: int = 60) -> None:
    """
    Generate a histogram of the number of legal chess moves from a dataset and save it.

    Args:
        config (dict): Configuration containing tokenizer and dataset information.
        saveFilePath (str): Path to save the histogram image.
        num_batches (int): Number of batches to iterate through.
        binwidth (int): Width of each bin in the histogram.
        cutoff (int): Cutoff value for the number of legal moves.
    """
    print(f"loading in tokenizer and dataloader")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_load_dir"])
    config, device, ddp, ddp_local_rank, master_process = set_ddp_params(config=config)
    train_iter = LlamaLoader(training_config=config, tokenizer=tokenizer, split="train")
    chess_board = chess.Board()
    num_moves = []
    print(f"starting iteration")

    for i in range(num_batches):
        print(f"starting batch {i}")
        _, _, _, fen_batch, _ = next(train_iter)
        for fen in fen_batch:
            chess_board.set_fen(fen=fen)
            num_moves.append(len([str(m) for m in chess_board.legal_moves]))

    # Calculate bins dynamically based on binwidth
    max_moves = max(num_moves)
    bins = range(0, max_moves + binwidth, binwidth)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(num_moves, bins=bins, color='blue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Legal Moves", fontsize=16)
    plt.xlabel("Number of Legal Moves in Board Position", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Draw a vertical line at the cutoff
    plt.axvline(x=cutoff, color='red', linestyle='--', label=f'Cutoff: {cutoff}')
    plt.legend(fontsize=12)

    # Calculate and print the percentage of positions above the cutoff
    above_cutoff = sum(1 for moves in num_moves if moves > cutoff)
    percentage_above_cutoff = (above_cutoff / len(num_moves)) * 100
    print(f"Percentage of positions with legal moves > {cutoff}: {percentage_above_cutoff:.2f}%")

    # Save the plot to the specified file path
    plt.savefig(saveFilePath, format='png')
    plt.close()

    if ddp:
        destroy_process_group()


def plotSweep(results: dict, save_file_path: str, agent_name: str = "Llama")->None:
    """
    Plots a bar chart of the number of wins against each Stockfish agent.

    Args:
        results (dict): Dictionary where keys are Stockfish levels and values are tuples (wins, draws, losses).
    """
    # Extract stockfish levels and wins from the dictionary
    levels = list(results.keys())
    wins = [result[0] for result in results.values()]
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(levels, wins, color='blue', alpha=0.7)
    
    # Add titles and labels
    plt.title(f"{agent_name} Wins Against Stockfish Agents", fontsize=14)
    plt.xlabel("Stockfish Level", fontsize=12)
    plt.ylabel("Wins out of 100", fontsize=12)
    
    plt.ylim(0,100)
    # Show the plot
    plt.tight_layout()
    plt.savefig(save_file_path)
    

# accepts a dictionary of tuples where each entry represents wins/draws/losses against stockfish level i. estimates elo rating based on this performance.
def calcELO(results: dict)-> float:
    # source: https://github.com/official-stockfish/Stockfish/commit/a08b8d4
    stockfish_ratings = {
        "stockfish0": 1320,
        "stockfish1": 1467,
        "stockfish2": 1608,
        "stockfish3": 1742,
        "stockfish4": 1922,
        "stockfish5": 2203,
        "stockfish6": 2363,
        "stockfish7": 2499,
        "stockfish8": 2596,
        "stockfish9": 2702,
        "stockfish10": 2788
    }
    start_estimated_rating = 1300 # according to Karvonen's estimate
    K_factor = 16 # scaling factor. 16 is standard
    # todo: why are there < 100 matches played in most of these? illegal moves made by gpt? those should count as losses (shouldn't matter for the purposes of ELO calculation)
    for opponent in stockfish_ratings.keys():
        if opponent in results:
            N = sum(results[opponent]) # total matches played
            wins = results[opponent][0]
            expected_wins = 1/(1 + 10**((stockfish_ratings[opponent]-start_estimated_rating)/400))
            start_estimated_rating += K_factor * (wins - (N * expected_wins))
            print(f"estimated rating after {N} matches against {opponent}: {start_estimated_rating}")
    return start_estimated_rating
# function to read in n csv files detailing one agent's performance across n stockfish agents. outputs df containing
def calcOneWinRate(df: pd.DataFrame)->Tuple:
    wins = sum(df["result"].str.count("1-0"))
    draws = sum(df["result"].str.count("1/2-1/2"))
    losses = sum(df["result"].str.count("0-1"))
    print(f"Results against Stockfish: {wins} wins, {draws} draws, {losses} losses")
    results = (wins, draws, losses)
    return results

def plot_prompt_statistics(path_to_prompt_statistics: str, path_to_save_plots: str) -> None:
    """
    Generates a single figure with subplots for prompt statistics, displays percentages above bars, and saves the plot.
    
    Args:
        path_to_prompt_statistics (str): Path to the prompt statistics text file.
        path_to_save_plots (str): Path to save the generated plot.
    """
    # Read the data from the file
    with open(path_to_prompt_statistics, 'r') as file:
        data = file.read()
    
    # Parse the data into prompts
    prompts = [prompt.strip() for prompt in data.split("----------------------------------------") if prompt.strip()]
    
    # Initialize lists for plotting
    prompt_names = []
    valid_san_ratios = []
    legal_ratios = []
    best_ratios = []
    
    for prompt in prompts:
        try:
            lines = prompt.split("\n")
            # Extract the necessary fields
            prompt_name = next(line.split(": ")[1] for line in lines if line.startswith("prompt_name"))
            total_prompts = int(next(line.split(": ")[1] for line in lines if line.startswith("total_prompts")))
            valid_san_moves = int(next(line.split(": ")[1] for line in lines if line.startswith("valid_san_moves")))
            legal_moves = int(next(line.split(": ")[1] for line in lines if line.startswith("legal_moves")))
            best_moves = int(next(line.split(": ")[1] for line in lines if line.startswith("best_moves")))
            
            # Compute proportions
            prompt_names.append(prompt_name)
            valid_san_ratios.append(valid_san_moves / total_prompts)
            legal_ratios.append(legal_moves / total_prompts)
            best_ratios.append(best_moves / total_prompts)
        except (IndexError, ValueError, StopIteration):
            print(f"Error parsing prompt: {prompt}")
            continue

    # Create the figure with subplots
    n = len(prompt_names)
    cols = 3  # Number of columns in the subplot grid
    rows = (n + cols - 1) // cols  # Calculate number of rows needed
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten the axes array for easier indexing
    
    for i in range(n):
        proportions = [valid_san_ratios[i], legal_ratios[i], best_ratios[i]]
        bars = axes[i].bar(["Valid SAN", "Legal", "Best"], proportions, color=["blue", "orange", "green"])
        axes[i].set_title(f"{prompt_names[i]}", pad=20)
        axes[i].set_ylabel("Proportion")
        axes[i].set_ylim(0, 1)
        
        # Add percentages above each bar
        for bar, proportion in zip(bars, proportions):
            percentage = f"{proportion * 100:.3f}%"  # Format as percentage
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, 
                         percentage, ha='center', va='bottom', fontsize=10)
    
    # Hide unused subplots if prompts are fewer than the grid size
    for j in range(n, len(axes)):
        axes[j].axis("off")
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(path_to_save_plots)
    plt.close()
if __name__ == "__main__":
    
    ################### plotting histogram of number of legal moves in each board position across a sample of dataset
    # config_file =  "/workspace/searchless_chess/src/config_llama.yaml"   
    # with open(config_file, 'r') as stream:
    #     config = yaml.load(stream=stream, Loader=Loader)
    # plot_num_moves(config=config, 
    #                saveFilePath="/workspace/searchless_chess/src/Llama/num_moves_hist_cutoff.png")

    ################### pythia exposure bias
    plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/pythia/pythia_exposure_bias_stats.txt",
                           path_to_save_plots="/workspace/searchless_chess/src/pythia/exposure_bias_bar_charts.png")
    
    ################### llama exposure bias
    # plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/Llama/llama_exposure_bias_stats.txt",
    #                        path_to_save_plots="/workspace/searchless_chess/src/Llama/exposure_bias_bar_charts.png")
    ################### baseline pythia performance
    # plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/pythia/pythia_prompt_stats.txt",
    #                        path_to_save_plots="/workspace/searchless_chess/src/pythia/prompt_bar_charts.png")
    
    ################### final vs. baseline pythia performance
    # plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/pythia/final_vs_baseline.txt",
    #                        path_to_save_plots="/workspace/searchless_chess/src/pythia/final_vs_baseline.png")
    
    ################### baseline llama performance
    # plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/Llama/llama_prompt_stats.txt",
    #                        path_to_save_plots="/workspace/searchless_chess/src/Llama/prompt_bar_charts.png")
    
    ################### final vs. baseline llama performance
    # plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/Llama/final_vs_baseline.txt",
    #                        path_to_save_plots="/workspace/searchless_chess/src/Llama/final_vs_baseline.png")
    
    ################### plotting number of wins, draws, losses against stockfish levels and calculating elo for Llama
    # fileDir = "/workspace/searchless_chess/src/pythia/logs"
    # # level=0
    # agent_name = "llama3_1_ckpt1000000"
    # performance = {}
    # for level in range(11):
    #     opponent = "stockfish" + str(level)
    #     filename = agent_name + "_vs_" + opponent + ".csv"
    #     fullfile = os.path.join(fileDir, filename)
    #     df = pd.read_csv(fullfile)
    #     result = calcOneWinRate(df=df)
    #     print(f"w/d/l against stockfish level {level}: {result}")
    #     performance[opponent]=result
    # calcELO(results = performance)
    # plotSweep(results=performance, save_file_path="/workspace/searchless_chess/src/Llama/stockfish_results.png")
    
    ################### plotting number of wins, draws, losses against stockfish levels and calculating elo for pythia
    # fileDir = "/workspace/searchless_chess/src/pythia/logs"
    # # level=0
    # agent_name = "pythia160m_ckpt208000"
    # performance = {}
    # for level in range(11):
    #     opponent = "stockfish" + str(level)
    #     filename = agent_name + "_vs_" + opponent + ".csv"
    #     fullfile = os.path.join(fileDir, filename)
    #     if os.path.exists(fullfile):
    #         df = pd.read_csv(fullfile)
    #         result = calcOneWinRate(df=df)
    #         print(f"w/d/l against stockfish level {level}: {result}")
    #         performance[opponent]=result
    # calcELO(results = performance)
    # plotSweep(results=performance, save_file_path="/workspace/searchless_chess/src/pythia/stockfish_results.png", agent_name="Pythia-160M")