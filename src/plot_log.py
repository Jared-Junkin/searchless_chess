import re
import matplotlib.pyplot as plt
import pandas as pd

def process_log_file(filepath):
    """
    Processes a logfile to extract specific lines with iteration information and calculates
    mean loss, seq_acc, gt.conf, ans.conf, and diff for groups of 1000 iterations.
    
    Args:
        filepath (str): Path to the logfile.
        
    Returns:
        pd.DataFrame: A DataFrame containing the mean statistics for each group of 1000 iterations.
    """
    # Regex pattern to match valid lines (iter, loss, etc.)
    log_pattern = re.compile(
        r"iter (?P<iter>\d+): loss (?P<loss>\d+\.\d+), time \d+\.\d+ms, "
        r"seq_acc: (?P<seq_acc>\d+\.\d+)%, gt\. conf: (?P<gt_conf>\d+\.\d+), "
        r"ans\. conf: (?P<ans_conf>\d+\.\d+).*?diff: (?P<diff>\d+\.\d+)"
    )
    
    # Initialize a list to store parsed data
    data = []
    
    # Read and parse the file
    with open(filepath, 'r') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                # Extract numeric values and convert them to float
                iter_num = int(match.group("iter"))
                loss = float(match.group("loss"))
                seq_acc = float(match.group("seq_acc"))
                gt_conf = float(match.group("gt_conf"))
                ans_conf = float(match.group("ans_conf"))
                diff = float(match.group("diff"))
                
                # Append parsed data
                data.append({
                    "iter": iter_num,
                    "loss": loss,
                    "seq_acc": seq_acc,
                    "gt_conf": gt_conf,
                    "ans_conf": ans_conf,
                    "diff": diff
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No matching log lines were found.")
        return pd.DataFrame()
    
    # Group by 1000-iteration buckets
    df['group'] = df['iter'] // 1000
    grouped = df.groupby('group').agg({
        'loss': 'mean',
        'seq_acc': 'mean',
        'gt_conf': 'mean',
        'ans_conf': 'mean',
        'diff': 'mean'
    }).reset_index()
    
    # Rename group to iteration range for clarity
    grouped['iteration_range'] = grouped['group'].apply(lambda x: f"{x*1000}-{(x+1)*1000 - 1}")
    grouped = grouped.drop(columns=['group'])
    
    return grouped

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
filepath = "/workspace/searchless_chess/src/pythia/logs/train_jared.log"
# filepath = 'path_to_your_logfile.log'
result = process_log_file(filepath)
# Display all rows
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)  # Automatically adjust width for terminal
pd.set_option('display.colheader_justify', 'center')  # Center-align headers
print(result)
plot_metrics(result, filename="performance.png")
