import re
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

# Example usage:
filepath = "/workspace/searchless_chess/src/pythia/logs/train_21.log"
# filepath = 'path_to_your_logfile.log'
result = process_log_file(filepath)
# Display all rows
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)  # Automatically adjust width for terminal
pd.set_option('display.colheader_justify', 'center')  # Center-align headers
print(result)
