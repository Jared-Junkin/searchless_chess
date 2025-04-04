import matplotlib.pyplot as plt
import re

"""
chess_eval_histogram.py

This script processes a file named `draws.txt` containing Stockfish evaluations of chess board states
and generates a histogram of these evaluations. The script categorizes the evaluations into specific bins
and displays the distribution of these categories. The resulting histogram includes text annotations above
each bin showing the percentage of total evaluations that fall into that bin.

### Inputs:
- **draws.txt**: A text file containing lines with chess evaluations. Each line follows a specific format:
    - Mate evaluations: "Eval: #+N, Win Prob: X" (e.g., "Eval: #+1, Win Prob: 1.0")
    - Opponent mate evaluations: "Eval: #-N, Win Prob: X" (e.g., "Eval: #-4, Win Prob: 0")
    - Positive evaluations: "Eval: +N, Win Prob: X" (e.g., "Eval: +2641, Win Prob: 1.0")
    - Draw evaluations: "Eval: 0, Win Prob: X" (e.g., "Eval: 0, Win Prob: 0.018")

### Outputs:
- **draws_dist.png**: A histogram image file saved to the current directory. The histogram shows the distribution
  of evaluations across the following bins:
    - "M1": Mate in 1 move.
    - "M2": Mate in 2 moves.
    - "M3": Mate in 3 moves.
    - "M4": Mate in 4 moves.
    - "M5": Mate in 5 moves.
    - "M6+": Mate in 6 or more moves.
    - "M-": Opponent has a forced mate.
    - "100+": Evaluation score is greater than or equal to 100.
    - "50+": Evaluation score is between 50 and 99.
    - "10+": Evaluation score is between 10 and 49.
    - "0": Evaluation score is exactly 0 (forced draw).
    - "<0": Evaluation score is less than 0 (losing position).

### Example Usage:
1. Ensure the `draws.txt` file is in the same directory as this script.
2. Run the script using:
    ```bash
    python chess_eval_histogram.py
    ```
3. The script will save the histogram plot to `draws_dist.png`.

### Dependencies:
- `matplotlib`: Used for plotting the histogram.
- `re`: Used for parsing the evaluation lines using regular expressions.

### Notes:
- The script assumes that each line in `draws.txt` conforms to one of the specified formats.
- The script handles mate evaluations, positive evaluations, forced draws, and opponent mate sequences.
- If any lines do not match the expected format, they will be ignored without affecting the output.

"""

# Define the input file path and output image path
input_file = "../draws.txt"
output_image = "./draws_dist.png"

# Define histogram bins and their labels
bins = ["M1", "M2", "M3", "M4", "M5", "M6+", "100+", "50+", "10+", "0", "<0", "M-"]
bin_counts = {bin_label: 0 for bin_label in bins}
total_count = 0

# Regular expressions for mate, opponent mate, evaluations, and draws
mate_pattern = re.compile(r"Eval: #\+(\d+), Win Prob:")
opponent_mate_pattern = re.compile(r"Eval: #-(\d+), Win Prob:")
eval_pattern = re.compile(r"Eval: \+(\d+), Win Prob:")
draw_pattern = re.compile(r"Eval: 0, Win Prob:")

# Read the file and categorize each evaluation
with open(input_file, "r") as file:
    for line in file:
        total_count += 1
        mate_match = mate_pattern.search(line)
        opponent_mate_match = opponent_mate_pattern.search(line)
        eval_match = eval_pattern.search(line)
        draw_match = draw_pattern.search(line)

        if mate_match:
            mate_moves = int(mate_match.group(1))
            # Categorize based on the mate number
            if mate_moves == 1:
                bin_counts["M1"] += 1
            elif mate_moves == 2:
                bin_counts["M2"] += 1
            elif mate_moves == 3:
                bin_counts["M3"] += 1
            elif mate_moves == 4:
                bin_counts["M4"] += 1
            elif mate_moves == 5:
                bin_counts["M5"] += 1
            else:
                bin_counts["M6+"] += 1
        elif opponent_mate_match:
            # Categorize any mate sequence for the opponent as "M-"
            bin_counts["M-"] += 1
        elif draw_match:
            # Categorize forced draw evaluations
            bin_counts["0"] += 1
        elif eval_match:
            evaluation = int(eval_match.group(1))
            # Categorize based on evaluation score
            if evaluation >= 100:
                bin_counts["100+"] += 1
            elif evaluation >= 50:
                bin_counts["50+"] += 1
            elif evaluation >= 10:
                bin_counts["10+"] += 1
            else:  # For evaluations less than 10 but not zero
                bin_counts["<0"] += 1

# Calculate percentages for each bin
percentages = {bin_label: (count / total_count) * 100 for bin_label, count in bin_counts.items()}

# Prepare data for plotting
labels = list(bin_counts.keys())
counts = list(bin_counts.values())
percentage_texts = [f"{percent:.1f}%" for percent in percentages.values()]

# Create the histogram
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(labels, counts, color='skyblue')

# Annotate the percentages above each bar
for bar, text in zip(bars, percentage_texts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 1, text, ha='center', va='bottom')

# Set plot labels and title
ax.set_xlabel("Evaluation Category")
ax.set_ylabel("Count")
ax.set_title("Stockfish Evaluations When Agent Draws")

# Save the plot
plt.savefig(output_image)
print(f"Histogram saved to {output_image}.")
