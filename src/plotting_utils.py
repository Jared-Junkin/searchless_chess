from language_data_loader import LlamaLoader
from typing import Tuple, Optional
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
import re
# this function will exactly plot loss as a function of iter given the most recent format I've used for my logfiles
def plot_eval_loss_from_log_pythia(filepath: str, savepath: str = "/workspace/searchless_chess/src/pythia/eval_loss_curve.png", ignore_first: int = 500, model_name: str = "Pythia") -> None:
    train_pattern = re.compile(r'iter (\d+): loss ([\d\.]+), time [\d\.]+ms')
    eval_pattern = re.compile(r'eval iter \d+, loss: ([\d\.]+),')
    
    eval_losses = []
    avg_eval_losses = []
    eval_iters = []
    current_train_iter = None
    
    with open(filepath, 'r') as file:
        for line in file:
            train_match = train_pattern.search(line)
            eval_match = eval_pattern.search(line)
            
            if train_match:
                current_train_iter = int(train_match.group(1))
            
            elif eval_match and current_train_iter is not None:
                loss_value = float(eval_match.group(1))
                eval_losses.append(loss_value)
                
                if len(eval_losses) == 100:
                    if current_train_iter >= ignore_first:
                        avg_loss = sum(eval_losses) / len(eval_losses)
                        eval_iters.append(current_train_iter + 10)
                        avg_eval_losses.append(avg_loss)
                        eval_losses.clear()
    
    if not eval_iters or not avg_eval_losses:
        print("No valid evaluation log entries found.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(eval_iters, avg_eval_losses, marker='o', linestyle='-', markersize=2)
    plt.xlabel('Iteration')
    plt.ylabel('Eval Loss')
    plt.title(f'Eval Loss for {model_name}')
    plt.grid(True)
    plt.savefig(savepath)
    print(f"Plot saved to {savepath}")

def plot_loss_from_log(filepath: str, savepath: str  = "/workspace/searchless_chess/src/Llama/loss_curve.png", ignore_first: int=500, model_name: str = "Llama3.1")->None:
    pattern = re.compile(r'iter (\d+): loss ([\d\.]+), time [\d\.]+ms')
    
    iterations = []
    losses = []
    
    with open(filepath, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                iter_num = int(match.group(1))
                if iter_num >= ignore_first: # high loss beginning will obscure trend
                    loss_value = float(match.group(2))
                    iterations.append(iter_num)
                    losses.append(loss_value)
    
    if not iterations:
        print("No valid log entries found.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, marker='o', linestyle='-',markersize=1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Train Loss for {model_name}')
    plt.grid(True)
    # plt.show()
    plt.savefig(savepath)

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
    
    
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import colorsys
def desaturate_color(color, factor=0.5):
    """
    Convert an RGB color to HLS, reduce its saturation, 
    and convert back to RGB.
    """
    h, l, s = colorsys.rgb_to_hls(*color)
    s *= factor  # Reduce saturation
    return colorsys.hls_to_rgb(h, l, s)

def plotSweep_all(
    results: List[dict], 
    save_file_path: str, 
    agent_names: List[str] = ["Llama"],
    gray_bars: Optional[List[str]] = None
) -> None:
    """
    Plots a bar chart of the % of wins for each agent against each Stockfish level.

    Args:
        results (List[dict]): List of dictionaries where each dictionary 
                              corresponds to an agent. Keys are Stockfish 
                              levels and values are tuples (wins, draws, losses).
        save_file_path (str): Path to save the resulting plot.
        agent_names (List[str]): List of agent names corresponding to the results.
        gray_bars (Optional[List[str]]): List of agent names to be grayed out 
                                         (desaturated) in the plot.
    """
    # Ensure the number of agents matches the results
    if len(results) != len(agent_names):
        raise ValueError("The number of results dictionaries must match the number of agent names.")

    # Extract Stockfish levels (assuming all agents have the same levels)
    stockfish_levels = list(results[0].keys())
    n_levels = len(stockfish_levels)
    n_agents = len(agent_names)

    # Prepare data for the plot
    bar_width = 0.8 / n_agents  # Adjust bar width to fit all agents in each level group
    x_positions = np.arange(n_levels)  # Base positions for Stockfish levels

    # We'll use the tab10 colormap for colors
    original_colors = list(plt.cm.tab10.colors)

    # For convenience, prepare a desaturated version of each color in tab10
    desaturated_colors = [desaturate_color(c, factor=0.2) for c in original_colors]

    plt.figure(figsize=(12, 8))

    # Plot each agent's results
    for i, (agent_results, agent_name) in enumerate(zip(results, agent_names)):
        wins = [agent_results[level][0] for level in stockfish_levels]
        total = [sum(agent_results[level]) for level in stockfish_levels]
        # Compute win percentage
        wins_percent = [(w / t) * 100 if t != 0 else 0 for w, t in zip(wins, total)]

        # Determine if this agent should be grayed out
        is_gray = gray_bars and (agent_name in gray_bars)
        # Choose color and alpha based on whether we're graying out
        color = desaturated_colors[i % len(desaturated_colors)] if is_gray else original_colors[i % len(original_colors)]
        alpha_val = 0.5 if is_gray else 1.0

        # Offset bars for each agent
        plt.bar(
            x_positions + i * bar_width, 
            wins_percent, 
            width=bar_width, 
            label=agent_name if not is_gray else f"{agent_name}", 
            color=color,
            alpha=alpha_val
        )

    # Customize the plot
    plt.title("Agent Wins Against Stockfish Levels", fontsize=16)
    plt.xlabel("Stockfish Level", fontsize=14)
    plt.ylabel("Percentage of Games (%)", fontsize=14)
    plt.xticks(x_positions + bar_width * (n_agents - 1) / 2, stockfish_levels)
    plt.ylim(0, 100)
    plt.legend(title="Agents", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # If there are any gray bars, add a footnote
    if gray_bars:
        footnote_text = "* Gray bars indicate Non-LLMs."
        plt.figtext(0.95, 0.02, footnote_text, ha="right", fontsize=10, color="black")

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(save_file_path)
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt


def winLossDrawGraph(df: pd.DataFrame, savePath: str, agent_name: str="Chess-GPT")->None:
    # Calculate percentages
    df["total"] = df["wins"] + df["draws"] + df["losses"]
    df["wins_pct"] = (df["wins"] / df["total"]) * 100
    df["draws_pct"] = (df["draws"] / df["total"]) * 100
    df["losses_pct"] = (df["losses"] / df["total"]) * 100

    # Plot
    plt.figure(figsize=(10, 6))
    levels = df["level"]
    plt.bar(levels, df["wins_pct"], label="Win", color="blue")
    plt.bar(levels, df["draws_pct"], bottom=df["wins_pct"], label="Draw", color="orange")
    plt.bar(
        levels,
        df["losses_pct"],
        bottom=df["wins_pct"] + df["draws_pct"],
        label="Loss",
        color="green",
    )

    # Formatting
    plt.title(f"{agent_name} vs Stockfish - Percentage Results", fontsize=14)
    plt.xlabel("Stockfish Level", fontsize=12)
    plt.ylabel("Percentage of Games (%)", fontsize=12)
    plt.xticks(levels, [f"Stockfish {i}" for i in levels], rotation=45)
    plt.legend(title="Game Outcome", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Show plot
    plt.savefig(savePath)


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
            # print(f"estimated rating after {N} matches against {opponent}: {start_estimated_rating}")
    return start_estimated_rating


def calcOneWinRate(df: pd.DataFrame, color: str = "white")->dict:
    # does 1-0 mean we win or does it mean white wins. I think it probably means white wins. (We'll know if black inexplicably beats stockfihs 10 90% of the time)
    if color == "white":
        wins = sum(df["result"].str.count("1-0"))
        draws = sum(df["result"].str.count("1/2-1/2"))
        losses = sum(df["result"].str.count("0-1"))
    elif color == "black":
        losses = sum(df["result"].str.count("1-0"))
        draws = sum(df["result"].str.count("1/2-1/2"))
        wins = sum(df["result"].str.count("0-1"))
    # print(f"Results against Stockfish Level {stockfish_level}: {wins} wins, {draws} draws, {losses} losses")
    return (wins, draws, losses)


# plots information theoretic performance (categorical crossentropy loss and accuracy of identical models trained on PGN and FEN
def plotITPerf(readfile: str = "filepathtodata.txt", writefile: str = "filepathtosavefigure.png") -> None:
    # Read the data from the text file into a DataFrame
    df = pd.read_csv(readfile)

    # Extract data
    models = df['model']
    loss = df['loss']
    accuracy = df['acc']

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Crossentropy Loss
    ax1.bar(models, loss, color=['blue', 'orange'])
    ax1.set_title('Crossentropy Loss (test data)')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Model')
    # ax1.legend(models, title="Model")

    # Plot Accuracy
    ax2.bar(models, accuracy, color=['blue', 'orange'])
    ax2.set_title('Accuracy (% Best Moves Played)')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Model')
    # ax2.legend(models, title="Model")

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(writefile)
    plt.close()

def plotratings(ELOs: List[float], models: List[str], savePath: str) -> None:
    """
    Plots a bar chart of estimated ELO ratings for different models.

    Args:
        ELOs (List[float]): List of ELO ratings for the models.
        models (List[str]): List of model names corresponding to the ELO ratings.
        savePath (str): Path to save the generated plot.
    """
    plt.figure(figsize=(10, 6))
    
    # Create the bar chart
    plt.bar(models, ELOs, color='skyblue', alpha=0.8)
    
    # Add titles and labels
    plt.title("Estimated ELO Ratings of Models", fontsize=14)
    plt.xlabel("Models", fontsize=12)
    plt.ylabel("ELO Rating", fontsize=12)
    
    # Annotate bars with ELO values
    for i, elo in enumerate(ELOs):
        plt.text(i, elo + 5, f'{elo:.0f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(savePath)
    print(f"Figure saved to {savePath}")

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



######################
import matplotlib.pyplot as plt
from typing import List, Optional
def plot_prompt_statistics_allmodels(
    path_to_prompt_statistics: str,
    path_to_save_plots: str,
    gray_bars: Optional[List[str]] = None
) -> None:
    """
    Generates a grouped bar chart showing % accuracy (best moves / total moves),
    % legal moves (legal moves / total moves), and % valid SAN (valid SAN moves / total moves)
    for different models. Optionally desaturates bars for models in `gray_bars` and includes
    a footnote indicating "Not LLMs".

    Args:
        path_to_prompt_statistics (str): Path to the prompt statistics text file.
        path_to_save_plots (str): Path to save the generated plot.
        gray_bars (Optional[List[str]]): List of model names to be highlighted with desaturated colors.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import colorsys

    # Read the data from the file
    with open(path_to_prompt_statistics, 'r') as file:
        data = file.read()

    # Parse the data into prompts
    prompts = [prompt.strip() for prompt in data.split("----------------------------------------") if prompt.strip()]

    # Initialize lists for storing extracted data
    models = []
    total_prompts = []
    valid_san_moves = []
    legal_moves = []
    best_moves = []

    for prompt in prompts:
        try:
            lines = prompt.split("\n")
            # Extract necessary fields
            model_name = next(line.split(": ")[1] for line in lines if line.startswith("prompt_name"))
            total = int(next(line.split(": ")[1] for line in lines if line.startswith("total_prompts")))
            valid_san = int(next(line.split(": ")[1] for line in lines if line.startswith("valid_san_moves")))
            legal = int(next(line.split(": ")[1] for line in lines if line.startswith("legal_moves")))
            best = float(next(line.split(": ")[1] for line in lines if line.startswith("best_moves")))

            # Append extracted values to lists
            models.append(model_name)
            total_prompts.append(total)
            valid_san_moves.append(valid_san)
            legal_moves.append(legal)
            best_moves.append(best)
        except (IndexError, ValueError, StopIteration):
            print(f"Error parsing prompt: {prompt}")
            continue

    # Convert lists to NumPy arrays
    total_prompts = np.array(total_prompts)
    valid_san_moves = np.array(valid_san_moves)
    legal_moves = np.array(legal_moves)
    best_moves = np.array(best_moves)

    # Compute percentages
    valid_san_percent = valid_san_moves / total_prompts * 100
    legal_moves_percent = legal_moves / total_prompts * 100
    best_moves_percent = best_moves / total_prompts * 100

    # Sort indices based on best moves percentage
    sorted_indices = np.argsort(best_moves_percent)
    models = [models[i] for i in sorted_indices]
    valid_san_percent = valid_san_percent[sorted_indices]
    legal_moves_percent = legal_moves_percent[sorted_indices]
    best_moves_percent = best_moves_percent[sorted_indices]

    # Define original colors
    original_colors = {
        "valid_san": (0, 0, 1),  # Blue
        "legal_moves": (1, 0.6, 0),  # Orange
        "best_moves": (0, 0.5, 0)  # Green
    }

    def desaturate_color(color, factor=0.5):
        """ Convert an RGB color to HLS, reduce its saturation, and convert back to RGB """
        h, l, s = colorsys.rgb_to_hls(*color)
        s *= factor  # Reduce saturation
        return colorsys.hls_to_rgb(h, l, s)

    # Adjust colors for gray bars
    desaturated_colors = {
        key: desaturate_color(value, factor=0.15) for key, value in original_colors.items()
    }

    # Plot grouped bar chart
    x = np.arange(len(models))  # X positions for the bars
    width = 0.25  # Width of each bar

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = []
    bars2 = []
    bars3 = []

    for i, model in enumerate(models):
        is_gray = gray_bars and model in gray_bars
        alpha = 0.5 if is_gray else 1.0
        color_valid = desaturated_colors["valid_san"] if is_gray else original_colors["valid_san"]
        color_legal = desaturated_colors["legal_moves"] if is_gray else original_colors["legal_moves"]
        color_best = desaturated_colors["best_moves"] if is_gray else original_colors["best_moves"]

        bars1.append(ax.bar(x[i] - width, valid_san_percent[i], width, color=color_valid, alpha = alpha, label="Valid SAN (%)" if i == 0 else ""))
        bars2.append(ax.bar(x[i], legal_moves_percent[i], width, color=color_legal, alpha=alpha, label="Legal Moves (%)" if i == 0 else ""))
        bars3.append(ax.bar(x[i] + width, best_moves_percent[i], width, color=color_best, alpha=alpha, label="Best Moves (%)" if i == 0 else ""))

    # Add labels and title
    ax.set_xlabel("Models")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Comparison of Chess Move Accuracy by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")

    # Main legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left")

    # Add percentage labels above bars
    for bars in [bars1, bars2, bars3]:
        for group in bars:
            for bar in group:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}%", 
                        ha='center', va='bottom', fontsize=10)

    # Add footnote if gray bars are present
    if gray_bars:
        footnote_text = "*Gray bars indicate Non-LLMs"
        plt.figtext(0.95, 0.02, footnote_text, ha="right", fontsize=10, color="black")

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
    # plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/pythia/pythia_exposure_bias_stats.txt",
    #                        path_to_save_plots="/workspace/searchless_chess/src/pythia/exposure_bias_bar_charts.png")
    
    ################### llama exposure bias
    # plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/Llama/llama_exposure_bias_stats.txt",
    #                        path_to_save_plots="/workspace/searchless_chess/src/Llama/exposure_bias_bar_charts.png")
    ################### baseline pythia performance
    # plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/pythia/pythia_prompt_stats.txt",
    #                        path_to_save_plots="/workspace/searchless_chess/src/pythia/prompt_bar_charts.png")
    
    ################### final vs. baseline pythia performance
    # plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/pythia/final_vs_baseline.txt",
    #                        path_to_save_plots="/workspace/searchless_chess/src/pythia/final_vs_baseline.png")
    
    ################### accuracy, valid san, legal moves for all models:
    # plot_prompt_statistics_allmodels(path_to_prompt_statistics="/workspace/searchless_chess/src/pythia/model_comparison.txt", 
    #                                  path_to_save_plots="/workspace/searchless_chess/src/all_plots_performance.png",
    #                                  gray_bars=["Karvonen*", "Ruoss et. al*"])
    
    ################## baseline llama performance
    # plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/Llama/llama_prompt_stats.txt",
    #                        path_to_save_plots="/workspace/searchless_chess/src/Llama/prompt_bar_charts.png")
    
    ################### final vs. baseline llama performance
    # plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/Llama/final_vs_baseline.txt",
    #                        path_to_save_plots="/workspace/searchless_chess/src/Llama/final_vs_baseline.png")
    
    ################### plot side-by-side baseline performance for llama and pythia
    # plot_prompt_statistics(path_to_prompt_statistics="/workspace/searchless_chess/src/baseline_both.txt",
    #                        path_to_save_plots="/workspace/searchless_chess/src/baseline.png")
    
    
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
    # eloLlama = calcELO(results = performance)
    # # plotSweep(results=performance, save_file_path="/workspace/searchless_chess/src/Llama/stockfish_results.png")
    
    
    # fileDir = "/workspace/searchless_chess/src/utils/stockfish_sweep.txt"
    # df = pd.read_csv(fileDir)
    # performance_llama ={f"stockfish{int(df['level'][i])}": (df["wins"][i], df["draws"][i], df["losses"][i]) for i in range(len(df))}
    
    # # ################### plotting number of wins, draws, losses against stockfish levels and calculating elo for pythia
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
    #         # print(f"w/d/l against stockfish level {level}: {result}")
        
    #     filename = opponent + "_vs_" + agent_name + ".csv"
    #     fullfile = os.path.join(fileDir, filename)
    #     if os.path.exists(fullfile):
    #         df2 = pd.read_csv(fullfile)
    #         result2 = calcOneWinRate(df=df2, color="black")
    #         # print(f"w/d/l against stockfish level with black {level}: {result2}")
    #     performance[opponent] = (result[0]+result2[0], result[1]+result2[1], result[2]+result2[2])
    # eloPythia = calcELO(results = performance)
    # print(f"elo of pythis is {eloPythia}")
    # df = pd.DataFrame.from_dict(performance, orient='index', columns=['wins', 'draws', 'losses']).reset_index().rename(columns={'index': 'level'})
    # df['level'] = df['level'].str.extract('(\d+)').astype(int)
    # df = df.sort_values('level').reset_index(drop=True)
    # # winLossDrawGraph(df=df, savePath="./pythia/wld.png", agent_name="Pythia")
    # # # # plotSweep(results=performance, save_file_path="/workspace/searchless_chess/src/pythia/stockfish_results.png", agent_name="Pythia-160M")
    
    
    
    # # # # ################### loading in data from karvonen's nanoGPT runs (my training runs on PGN data with his model architecture)
    # fileDir = "/workspace/searchless_chess/src/karvonen_nanoGPT/stockfish_results.txt"
    # df = pd.read_csv(fileDir)
    # d = {f"stockfish{int(df['level'][i])}": (df["wins"][i], df["draws"][i], df["losses"][i]) for i in range(len(df))}
    # eloKarvonen = calcELO(results=d)
    # print(eloKarvonen)
    
    # # # # ################## loading in data from deepmind runs (my training runs iwth deepmind's nanogpt architecture)
    # fileDir="/workspace/searchless_chess/src/deepmind_results/stockfish_final.txt"
    # df = pd.read_csv(fileDir)
    # winLossDrawGraph(df=df, savePath="./deepmind_results/wld.png", agent_name="Deepmind")
    # deepmind = {f"stockfish{int(df['level'][i])}": (df["wins"][i], df["draws"][i], df["losses"][i]) for i in range(len(df))}
    # eloDeepmind = calcELO(results=deepmind)
    # print(f"deepmind elo is {eloDeepmind}")
    
    # # # # ################### plotting number of wins, draws, losses against stockfish levels and calculating elo for all agents in one bar graph
    # plotSweep_all(results=[performance_llama, deepmind, d, performance], save_file_path="/workspace/searchless_chess/src/stockfish_results_all.png", agent_names=["Llama (Ours)", "Ruoss et. al", "Karvonen", "Pythia (Ours)"], gray_bars=["Ruoss et. al", "Karvonen"])
    
    # ################### making a bar chart of all the models' elo ratings.
    # plotratings(ELOs=[2910, 2579, 2118, 1306], models=["Deepmind", "Llama (Ours)", "Karvonen (White Pieces Only)", "Pythia (Ours)"], savePath="/workspace/searchless_chess/src/all_elo_ratings.png")
    
    
    
    # ################### Making w/l/d bar graph across stockfish levels for deepmind
    # df = pd.read_csv("./deepmind_results/stockfish_final.txt")
    # winLossDrawGraph(df=df, savePath="./deepmind_results/wld.png", agent_name = "Ruoss et. al nanoGPT")
    
    ################### Making w/l/d bar graph across stockfish levels for karvonen
    # fileDir = "/workspace/searchless_chess/src/karvonen_nanoGPT/stockfish_results.txt"
    # df = pd.read_csv(fileDir)
    # winLossDrawGraph(df=df, savePath="./karvonen_nanoGPT/wld.png", agent_name="Karvonen nanoGPT")
    ################## Making w/l/d bar graph across stockfish levels for Llama
    # fileDir = "/workspace/searchless_chess/src/utils/stockfish_sweep.txt"
    # df = pd.read_csv(fileDir)
    # #### {"stockfish"+str(df.iloc[i]['level']): (df.iloc[i]['wins'], df.iloc[i]['draws'], df.iloc[i]['losses']) for i in range(len(df))}
    # winLossDrawGraph(df=df, savePath="./Llama/wld.png", agent_name="Llama3.1")
    
    # #################### calculate elo from results dict pasted in
    # results = {'Stockfish 0': (30, 2, 68), 'Stockfish 1': (36, 2, 61), 'Stockfish 2': (25, 4, 69), 'Stockfish 3': (22, 5, 71), 'Stockfish 4': (20, 5, 74), 'Stockfish 5': (10, 4, 82), 'Stockfish 6': (12, 2, 82), 'Stockfish 7': (6, 2, 89), 'Stockfish 8': (1, 1, 97), 'Stockfish 9': (4, 1, 93), 'Stockfish 10': (1, 5, 93)}
    # print(calcELO(results=results))
    
    #################### plot loss and accuracy of identical models trained on FEN and PGN data
    # readfile = "/workspace/searchless_chess/src/karvonen_nanoGPT/ITLoss.txt"
    # plotITPerf(readfile=readfile, writefile="/workspace/searchless_chess/src/karvonen_nanoGPT/ITLoss.png")
    
    
    
    
    
    
    
    # #################### Plot train loss as a function of iter for Llama
    logfile = "/workspace/searchless_chess/src/Llama/logs/final_run.log"
    plot_loss_from_log(filepath=logfile)
    
    # #################### Plot eval loss as a function of iter for pythia
    # logfile = "/workspace/searchless_chess/src/pythia/logs/train_nof2.log"
    # plot_eval_loss_from_log_pythia(filepath=logfile)