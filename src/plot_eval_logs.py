import matplotlib.pyplot as plt
import pandas as pd
import re

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
        r"eval iter (\d+),\s*"
        r"loss: ([\d\.]+),\s*"
        r"seq_acc:([\d\.]+)%,\s*"
        r"gt\. conf: ([\d\.]+),\s*"
        r"ans\. conf: ([\d\.]+)\s*"
        r"diff: ([\d\.]+)"
    )
    data = []
    current_block = []

    # Read the logfile line by line
    with open(filepath, 'r') as f:
        for line in f:
            if "eval iter" in line:
                match = re.search(pattern, line)

                # Append parsed data to the current block
                current_block.append({
                   "eval_iter": int(match.group(1)),
                    "loss": float(match.group(2)),
                    "seq_acc": float(match.group(3)),
                    "gt_conf": float(match.group(4)),
                    "ans_conf": float(match.group(5)),
                    "diff": float(match.group(6)),
                })
            elif current_block:
                # If a non-eval line is encountered, process the current block
                data.append({
                    "loss": pd.DataFrame(current_block)["loss"].mean(),
                    "seq_acc": pd.DataFrame(current_block)["seq_acc"].mean(),
                    "gt_conf": pd.DataFrame(current_block)["gt_conf"].mean(),
                    "ans_conf": pd.DataFrame(current_block)["ans_conf"].mean(),
                    "diff": pd.DataFrame(current_block)["diff"].mean()
                })
                current_block = []

    # Process any remaining block at the end of the file
    if current_block:
        data.append({
            "loss": pd.DataFrame(current_block)["loss"].mean(),
            "seq_acc": pd.DataFrame(current_block)["seq_acc"].mean(),
            "gt_conf": pd.DataFrame(current_block)["gt_conf"].mean(),
            "ans_conf": pd.DataFrame(current_block)["ans_conf"].mean(),
            "diff": pd.DataFrame(current_block)["diff"].mean()
        })

    # Convert to DataFrame
    return pd.DataFrame(data)
def plot_metrics(result, filename="performance_nof2.png", ignore_index: int = 0):
    """
    Plots the metrics from the result DataFrame, starting after the specified number of rows.

    Args:
        result (pd.DataFrame): DataFrame containing the metrics grouped by contiguous eval blocks.
        filename (str): The name of the output file to save the plot.
        ignore_index (int): Number of rows to ignore from the start of the DataFrame.
    """
    if result.empty:
        print("Result DataFrame is empty. Nothing to plot.")
        return

    # Skip the first `ignore_index` rows
    if ignore_index > 0:
        result = result.iloc[ignore_index:]

    if result.empty:
        print(f"Result DataFrame is empty after ignoring the first {ignore_index} rows. Nothing to plot.")
        return

    # Adjust the layout to 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot loss
    axs[0, 0].plot(result.index, result['loss'], marker='o')
    axs[0, 0].set_title("Loss per Block")
    axs[0, 0].set_xlabel("Block Index")
    axs[0, 0].set_ylabel("Loss")

    # Plot sequence accuracy
    axs[0, 1].plot(result.index, result['seq_acc'], marker='o', color='orange')
    axs[0, 1].set_title("Sequence Accuracy per Block")
    axs[0, 1].set_xlabel("Block Index")
    axs[0, 1].set_ylabel("Sequence Accuracy (%)")

    # Plot gt_conf and ans_conf in the same subplot
    axs[1, 0].plot(result.index, result['gt_conf'], marker='o', color='blue', label='GT Confidence')
    axs[1, 0].plot(result.index, result['ans_conf'], marker='o', color='green', label='Answer Confidence')
    axs[1, 0].set_title("Confidence per Block")
    axs[1, 0].set_xlabel("Block Index")
    axs[1, 0].set_ylabel("Confidence")
    axs[1, 0].legend()

    # Plot diff
    axs[1, 1].plot(result.index, result['diff'], marker='o', color='red')
    axs[1, 1].set_title("Diff per Block")
    axs[1, 1].set_xlabel("Block Index")
    axs[1, 1].set_ylabel("Diff")

    plt.tight_layout()
    plt.savefig(filename)

import matplotlib.pyplot as plt


def compare(df1, df2, name1: str, name2: str, filename="comparison.png", ignore_index: int = 0):
    """
    Compares metrics from two DataFrames and produces 4 comparison plots.

    Args:
        df1 (pd.DataFrame): First DataFrame containing metrics.
        df2 (pd.DataFrame): Second DataFrame containing metrics.
        name1 (str): Name for the first dataset (used in legends).
        name2 (str): Name for the second dataset (used in legends).
        filename (str): The name of the output file to save the plot.
    """
    if df1.empty or df2.empty:
        print("One or both DataFrames are empty. Nothing to plot.")
        return
        # Skip the first `ignore_index` rows
    if ignore_index > 0:
        df1 = df1.iloc[ignore_index:]
        df2 = df2.iloc[ignore_index:]
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # Colors for each DataFrame
    color1 = 'blue'
    color2 = 'orange'

    # Plot loss
    axs[0, 0].plot(df1.index, df1['loss'], marker='o', color=color1, label=f"{name1} Loss")
    axs[0, 0].plot(df2.index, df2['loss'], marker='o', color=color2, label=f"{name2} Loss")
    axs[0, 0].set_title("Test Loss")
    axs[0, 1].set_xlabel("Iterations (x1000)")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    # Plot sequence accuracy
    axs[0, 1].plot(df1.index, df1['seq_acc'], marker='o', color=color1, label=f"{name1} Seq. Accuracy")
    axs[0, 1].plot(df2.index, df2['seq_acc'], marker='o', color=color2, label=f"{name2} Seq. Accuracy")
    axs[0, 1].set_title("Sequence Accuracy (% Correct Answers)")
    axs[0, 1].set_xlabel("Iterations (x1000)")
    axs[0, 1].set_ylabel("Sequence Accuracy (%)")
    axs[0, 1].legend()

    # Plot gt_conf and ans_conf in the same subplot
    axs[1, 0].plot(df1.index, df1['gt_conf'], marker='^', color=color1, label=f"{name1} GT Confidence")
    axs[1, 0].plot(df2.index, df2['gt_conf'], marker='^', color=color2, label=f"{name2} GT Confidence")
    axs[1, 0].plot(df1.index, df1['ans_conf'], marker='o', color=color1, label=f"{name1} Answer Confidence")
    axs[1, 0].plot(df2.index, df2['ans_conf'], marker='o', color=color2, label=f"{name2} Answer Confidence")
    axs[1, 0].set_title("Confidence per Block (Mean Prob Over Generated Tokens)")
    axs[0, 1].set_xlabel("Iterations (x1000)")
    axs[1, 0].set_ylabel("Confidence")
    axs[1, 0].legend()

    # Plot diff
    axs[1, 1].plot(df1.index, df1['diff'], marker='o', color=color1, label=f"{name1} Diff")
    axs[1, 1].plot(df2.index, df2['diff'], marker='o', color=color2, label=f"{name2} Diff")
    axs[1, 1].set_title("Confidence Difference Between Generated Response & Ground Truth")
    axs[0, 1].set_xlabel("Iterations (x1000)")
    axs[1, 1].set_ylabel("Diff")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(filename)




# Example usage
filepath = "/workspace/searchless_chess/src/pythia/logs/train_yesf2.log"
filepath2 = "/workspace/searchless_chess/src/pythia/logs/train_nof2.log"
result = process_log_file(filepath)
result2 = process_log_file(filepath=filepath2)
print(result2)
plot_metrics(result, filename="performance_yesf2.png", ignore_index=2)
compare(df1 = result, df2 = result2, name1="Decaying F2 Penalty", name2="No F2 Penalty", filename="comparison.png", ignore_index=10)