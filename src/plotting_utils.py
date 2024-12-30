from language_data_loader import LlamaLoader
from transformers import AutoTokenizer
import chess
import matplotlib.pyplot as plt
import yaml
from torch.distributed import destroy_process_group
from hooks import set_ddp_params
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
        axes[i].set_title(f"{prompt_names[i]}")
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
    
    ###################
    # config_file =  "/workspace/searchless_chess/src/config_llama.yaml"   
    # with open(config_file, 'r') as stream:
    #     config = yaml.load(stream=stream, Loader=Loader)
    # plot_num_moves(config=config, 
    #                saveFilePath="/workspace/searchless_chess/src/Llama/num_moves_hist_cutoff.png")
    ###################
    
    ################### baseline pythia performance
    # plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/pythia/pythia_prompt_stats.txt",
    #                        path_to_save_plots="/workspace/searchless_chess/src/pythia/prompt_bar_charts.png")
    ###################
    
    ################### baseline llama performance
    plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/Llama/llama_prompt_stats.txt",
                           path_to_save_plots="/workspace/searchless_chess/src/Llama/prompt_bar_charts.png")