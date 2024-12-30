import re
import pandas as pd
import matplotlib.pyplot as plt

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


result = process_log_file(filepath="/workspace/searchless_chess/src/Llama/logs/train_4.log",block_size=100)# block size = 25 means 25 iters will ultimately be averaged together.
print(result)
plot_metrics(result=result, filename="./Llama/performance_plot.png", ignore_index=2)
