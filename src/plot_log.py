import re
import matplotlib.pyplot as plt
import pandas as pd

def process_log_file(filepath):
    """
    Processes a logfile to extract contiguous blocks of eval iteration lines and calculates
    mean loss, seq_acc, gt.conf, ans.conf, and diff for each block.

    Args:
        filepath (str): Path to the logfile.

    Returns:
        pd.DataFrame: A DataFrame containing the mean statistics for each contiguous block of eval iterations.
    """
    pattern = (
        r"iter (\d+),\s*"
        r"loss: ([\d\.]+),\s*"
        r"time:([\d\.]+)%,\s*"
        r"gt\. conf: ([\d\.]+),\s*"
        r"ans\. conf: ([\d\.]+)\s*"
        r"diff: ([\d\.]+)"
    )
    data = []
    current_block = []

    # Read the logfile line by line
    with open(filepath, 'r') as f:
        for line in f:
            if "iter" in line:
                match = re.search(pattern, line)
                if match:

                    # Append parsed data to the current block
                    current_block.append({
                    "eval_iter": int(match.group(1)),
                        "loss": float(match.group(2)),
                        "seq_acc": float(match.group(3)),
                        "gt_conf": float(match.group(4)),
                        "ans_conf": float(match.group(5)),
                        "diff": float(match.group(6)),
                    })
                
                
def plot_metrics(result, filename: str = "performance.png"):
    """
    Plots the metrics from the result DataFrame.

    Args:
        result (pd.DataFrame): DataFrame containing the metrics grouped by iteration.
    """
    if result.empty:
        print("Result DataFrame is empty. Nothing to plot.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot loss
    axs[0, 0].plot(result.index, result['loss'], marker='o')
    axs[0, 0].set_title("Loss vs Iteration")
    axs[0, 0].set_xlabel("Iteration Group")
    axs[0, 0].set_ylabel("Loss")

    # Plot sequence accuracy
    axs[0, 1].plot(result.index, result['seq_acc'], marker='o', color='orange')
    axs[0, 1].set_title("Sequence Accuracy vs Iteration")
    axs[0, 1].set_xlabel("Iteration Group")
    axs[0, 1].set_ylabel("Sequence Accuracy (%)")

    # Plot gt_conf and ans_conf on the same plot
    axs[1, 0].plot(result.index, result['gt_conf'], label='GT Confidence', marker='o', color='blue')
    axs[1, 0].plot(result.index, result['ans_conf'], label='Answer Confidence', marker='o', color='green')
    axs[1, 0].set_title("GT Conf and Ans Conf vs Iteration")
    axs[1, 0].set_xlabel("Iteration Group")
    axs[1, 0].set_ylabel("Confidence")
    axs[1, 0].legend()

    # Plot diff
    axs[1, 1].plot(result.index, result['diff'], marker='o', color='red')
    axs[1, 1].set_title("Diff vs Iteration")
    axs[1, 1].set_xlabel("Iteration Group")
    axs[1, 1].set_ylabel("Diff")

    plt.tight_layout()
    plt.savefig(filename)
# Example usage:
filepath = "/workspace/searchless_chess/src/Llama/logs/train_3.log"
# filepath = 'path_to_your_logfile.log'
result = process_log_file(filepath)
# Display all rows
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)  # Automatically adjust width for terminal
pd.set_option('display.colheader_justify', 'center')  # Center-align headers
print(result)
plot_metrics(result, filename="performance.png")
