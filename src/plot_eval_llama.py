import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

def process_log_file(filepath, block_size=10):
    """
    Processes a logfile to extract contiguous blocks of iteration lines and calculates
    mean loss for each block.

    Args:
        filepath (str): Path to the logfile.
        block_size (int): Number of lines to average together in each block.

    Returns:
        pd.DataFrame: A DataFrame containing the mean statistics for each block.
    """
    pattern = (
        r"iter (\d+): loss ([\d\.]+), time ([\d\.]+)ms"
    )
    data = []
    current_block = []

    # Read the logfile line by line
    with open(filepath, 'r') as f:
        for line in f:
            if "iter" in line and "loss" in line:
                match = re.search(pattern, line)
                if match:
                    current_block.append({
                        "iter": int(match.group(1)),
                        "loss": float(match.group(2)),
                        "time": float(match.group(3)),
                    })

                    # Process block when reaching block_size
                    if len(current_block) == block_size:
                        block_df = pd.DataFrame(current_block)
                        data.append({
                            "mean_iter": block_df["iter"].mean(),
                            "mean_loss": block_df["loss"].mean(),
                            "mean_time": block_df["time"].mean(),
                        })
                        current_block = []

    # Process any remaining lines in the last block
    if current_block:
        block_df = pd.DataFrame(current_block)
        data.append({
            "mean_iter": block_df["iter"].mean(),
            "mean_loss": block_df["loss"].mean(),
            "mean_time": block_df["time"].mean(),
        })

    # Convert to DataFrame
    return pd.DataFrame(data)

def plot_metrics(result, filename="performance_plot.png", ignore_index: int = 0):
    """
    Plots the metrics from the result DataFrame, starting after the specified number of rows.

    Args:
        result (pd.DataFrame): DataFrame containing the metrics grouped by contiguous blocks.
        filename (str): The name of the output file to save the plot.
        ignore_index (int): Number of rows to ignore from the start of the DataFrame.
    """
    if result.empty:
        print("Result DataFrame is empty. Nothing to plot.")
        return

    # Skip the first ignore_index rows
    if ignore_index > 0:
        result = result.iloc[ignore_index:]

    if result.empty:
        print(f"Result DataFrame is empty after ignoring the first {ignore_index} rows. Nothing to plot.")
        return

    # Adjust the layout
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Plot loss
    axs[0].plot(result.index, result['mean_loss'], marker='o')
    axs[0].set_title("Mean Loss per Block")
    axs[0].set_xlabel("Block Index")
    axs[0].set_ylabel("Mean Loss")

    # Plot time
    axs[1].plot(result.index, result['mean_time'], marker='o', color='orange')
    axs[1].set_title("Mean Time per Block")
    axs[1].set_xlabel("Block Index")
    axs[1].set_ylabel("Mean Time (ms)")

    plt.tight_layout()
    plt.savefig(filename)

def plot_metrics_from_n_dfs(
    results_list: List[pd.DataFrame],
    filename: str = "performance_plot.png",
    ignore_index: int = 0,
    names: Optional[List[str]] = None
):
    """
    Plots the 'mean_loss' and 'mean_time' columns from each DataFrame in results_list.
    Each DataFrame will be plotted as a separate line in the corresponding subplot.

    Args:
        results_list (List[pd.DataFrame]): List of DataFrames containing the metrics.
        filename (str): Name of the output file to save the plot.
        ignore_index (int): Number of rows to ignore from the start of each DataFrame.
        names (List[str], optional): Legend labels for each DataFrame. 
                                     If provided, names[i] is the legend entry for results_list[i].
                                     If not provided or shorter than results_list, lines are unlabeled / partially labeled.
    """

    # If the list is empty, nothing to plot
    if not results_list:
        print("No DataFrames provided. Nothing to plot.")
        return

    # Create a figure with 2 subplots for 'mean_loss' and 'mean_time'
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Choose a set of distinct colors for each DataFrame.
    # (You can also use e.g. plt.cm.tab10 or any other colormap.)
    colors = plt.cm.tab10(range(len(results_list)))

    for idx, df in enumerate(results_list):
        # Skip empty DataFrames
        if df.empty:
            continue

        # Ignore specified rows
        if ignore_index > 0:
            df = df.iloc[ignore_index:]

        # If it became empty after ignoring rows, skip
        if df.empty:
            continue

        # Fetch the label from names if available, otherwise None
        label = names[idx] if (names is not None and idx < len(names)) else None

        # Plot 'mean_loss'
        if 'mean_loss' in df.columns:
            axs[0].plot(
                df.index, df['mean_loss'],
                marker='o',
                color=colors[idx],
                label=label
            )

        # Plot 'mean_time'
        if 'mean_time' in df.columns:
            axs[1].plot(
                df.index, df['mean_time'],
                marker='o',
                color=colors[idx],
                label=label
            )

    # Titles and axis labels
    axs[0].set_title("Mean Loss per Block")
    axs[0].set_xlabel("Block Index")
    axs[0].set_ylabel("Mean Loss")

    axs[1].set_title("Mean Time per Block")
    axs[1].set_xlabel("Block Index")
    axs[1].set_ylabel("Mean Time (ms)")

    # If we provided any labels, show the legend
    # (We check each axis separately, in case some data frames
    #  didn't contain 'mean_loss' or 'mean_time'.)
    if names:
        axs[0].legend(loc='best')
        axs[1].legend(loc='best')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

# result = process_log_file(filepath="/workspace/searchless_chess/src/Llama/logs/train_4.log",block_size=100)# block size = 25 means 25 iters will ultimately be averaged together.
# # result = process_log_file(filepath="/workspace/searchless_chess/src/Llama/logs/improve_accuracy.log",block_size=100)# block size = 25 means 25 iters will ultimately be averaged together.
# print(result)
# # plot_metrics(result=result, filename="./Llama/improve_acc_plot.png", ignore_index=2)
# plot_metrics(result=result,filename="./Llama/metrics_plot_accuracy.png")

############ plot same loss curve but for QLoRA
# result = process_log_file(filepath="/workspace/searchless_chess/src/Llama/logs/train_qlora.log", block_size=100)
# print(result)
# plot_metrics(result=result.iloc[1:], filename="./Llama/qlora_plot_accuracy.png")


############ plot loss curves of QLoRA and base training
base_training_start = process_log_file(filepath="/workspace/searchless_chess/src/Llama/logs/train_3.log", block_size=100)
base_training_rest = process_log_file(filepath="/workspace/searchless_chess/src/Llama/logs/train_4.log", block_size=100)
base_training = pd.concat([base_training_start, base_training_rest], ignore_index=True)
qlora = process_log_file(filepath="/workspace/searchless_chess/src/Llama/logs/train_qlora.log", block_size=100)
print(qlora)
plot_metrics_from_n_dfs(results_list=[base_training.iloc[:len(qlora)], qlora], filename="./Llama/qloraAndBase_plot_accuracy.png", names=["Full Fine Tuning", "QLoRA"], ignore_index=1)