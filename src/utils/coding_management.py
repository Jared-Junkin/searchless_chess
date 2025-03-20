import pandas as pd
import json
from datetime import datetime, timedelta
import os
from typing import List, Optional

###############################################################################
# Helpers for date parsing / formatting
###############################################################################

DATE_FORMAT = "%Y/%m/%d/%H/%M/%S"  # e.g. "2025/01/17/12/13/00"

def parse_date(date_str: str) -> datetime:
    """Parse a date string in the format yyyy/mm/dd/hh/mm/ss into a datetime."""
    return datetime.strptime(date_str, DATE_FORMAT)

def format_date(dt: datetime) -> str:
    """Format a datetime object into a string yyyy/mm/dd/hh/mm/ss."""
    return dt.strftime(DATE_FORMAT)

###############################################################################
# CSV Reading/Writing
###############################################################################

def read_spaced_df(csv_path: str) -> pd.DataFrame:
    """
    Reads the CSV file into a pandas DataFrame.
    We store lists in JSON format, so we must parse those columns after reading.
    If the file does not exist OR is empty, return an empty DataFrame with expected columns.
    """
    columns = [
        "Topic",
        "Last Touch",
        "Next Touch",
        "Problems Solved",
        "Difficulties",
        "Dates",
        "Problem Listed Difficulties",
    ]

    # If file doesn't exist or is empty, return a fresh empty DataFrame
    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        df = pd.DataFrame(columns=columns)
        return df

    df = pd.read_csv(csv_path)

    # Ensure all columns exist (in case the file is missing any)
    for col in columns:
        if col not in df.columns:
            df[col] = None  # or empty strings, or whatever you prefer

    # Convert JSON-strings to Python lists where appropriate
    list_cols = ["Problems Solved", "Difficulties", "Dates", "Problem Listed Difficulties"]
    for col in list_cols:
        df[col] = df[col].fillna("[]")  # Replace NaN with empty list
        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    return df


def write_spaced_df(df: pd.DataFrame, csv_path: str):
    """
    Writes the DataFrame to CSV, converting list columns back into JSON strings.
    """
    # Convert list columns to JSON strings
    list_cols = ["Problems Solved", "Difficulties", "Dates", "Problem Listed Difficulties"]
    for col in list_cols:
        df[col] = df[col].apply(json.dumps)

    df.to_csv(csv_path, index=False)

###############################################################################
# Spaced Repetition Scheduling
###############################################################################
def compute_next_touch(difficulties: List[int]) -> timedelta:
    """
    A more sophisticated spaced repetition heuristic, inspired by (but simplified from) Anki.
    We grow intervals faster if you've solved the topic multiple times and rated it easy,
    and we shorten intervals if you found it hard.

    Explanation:
      1) Let n = the total number of times you've solved the topic (len(difficulties)).
      2) If there are no difficulties recorded yet, schedule for 1 day from now.
      3) Otherwise:
         - Compute a 'base' interval that grows exponentially with n.
           We'll use:  base_interval = 2^(n+1) - 3
             (For n=1 -> 2^(2) - 3 = 1 day, n=2 -> 5 days, n=3 -> 13 days, n=4 -> 29 days, etc.)
           If that formula ever dips below 1 (e.g. for n=1), we set it back to 1 to avoid 0-day intervals.
         - Then scale that base interval by a factor depending on the *last* difficulty rating:
               last_diff <= 1 (easy or very easy)        => multiply by ~1.2
               last_diff in [2, 3] (medium / somewhat)   => multiply by ~1.0
               last_diff >= 4 (hard / missed)            => multiply by ~0.5
         - Finally, we round and clamp at least 1 day.
    """

    # If never solved, default to 1 day
    if not difficulties:
        return timedelta(days=1)

    last_diff = difficulties[-1]   # Difficulty of the most recent solve
    n = len(difficulties)          # How many times solved so far

    # Base interval that grows with each solve
    base_interval = (2 ** (n + 1)) - 3
    if base_interval < 1:
        base_interval = 1

    # Scale factor depends on how easy/hard you found it last time
    if last_diff <= 1:
        # Very easy or easy
        scale = 1.2
    elif last_diff <= 3:
        # Medium difficulty
        scale = 1.0
    else:
        # Hard or missed
        scale = 0.5

    next_interval_days = base_interval * scale

    # Ensure at least 1 day, then return as a timedelta
    next_interval_days = max(1, round(next_interval_days))
    return timedelta(days=next_interval_days)

###############################################################################
# Main SRS Functions
###############################################################################

def add(
    csv_path: str,
    category: str,
    problem_name: str,
    difficulty: int,
    listed_difficulty: str = "",
    last_touch: Optional[str] = None
):
    """
    1) Read the DataFrame from CSV.
    2) Find the row corresponding to `category`. If none, create a new row.
    3) Append the new problem to that category's "Problems Solved" list.
    4) Append difficulty to "Difficulties", and the current (or given) 'last_touch' to "Dates".
    5) Update 'Last Touch' with the given or current time.
    6) Compute 'Next Touch' using our spaced repetition function.
    7) Write back to the CSV.
    """
    df = read_spaced_df(csv_path)

    # Check if this topic exists
    row_index = df.index[df["Topic"] == category].tolist()
    if len(row_index) == 0:
        # Create a new row
        new_row = {
            "Topic": category,
            "Last Touch": "",
            "Next Touch": "",
            "Problems Solved": [],
            "Difficulties": [],
            "Dates": [],
            "Problem Listed Difficulties": [],
        }
        # df = df.append(new_row, ignore_index=True)
        df.loc[len(df)]=new_row
        row_index = [df.index[-1]]  # index of the newly added row

    idx = row_index[0]  # There's only one row per category, presumably

    # Pull out the lists
    problems_solved = df.at[idx, "Problems Solved"]
    difficulties_list = df.at[idx, "Difficulties"]
    dates_list = df.at[idx, "Dates"]
    listed_difficulties = df.at[idx, "Problem Listed Difficulties"]

    # Update them
    problems_solved.append(problem_name)
    difficulties_list.append(difficulty)
    listed_difficulties.append(listed_difficulty)

    if last_touch is None:
        # Use current time as the last touch
        now = datetime.now()
        last_touch_str = format_date(now)
    else:
        # Accept the user-given string
        now = parse_date(last_touch)
        last_touch_str = last_touch

    dates_list.append(last_touch_str)  # Record the date we solved the problem

    # Now we update the "Last Touch" for this category
    df.at[idx, "Last Touch"] = last_touch_str

    # Finally, compute next touch
    next_timedelta = compute_next_touch(difficulties_list)
    next_touch_dt = now + next_timedelta
    df.at[idx, "Next Touch"] = format_date(next_touch_dt)

    # Put the updated lists back into the DataFrame
    df.at[idx, "Problems Solved"] = problems_solved
    df.at[idx, "Difficulties"] = difficulties_list
    df.at[idx, "Dates"] = dates_list
    df.at[idx, "Problem Listed Difficulties"] = listed_difficulties

    # Write it all back to CSV
    write_spaced_df(df, csv_path)

def getstudytopics(csv_path: str) -> List[str]:
    """
    Read in the DataFrame, sort all topics by their Next Touch date/time
    (earliest first), and return the sorted list of topic strings.

    If 'Next Touch' is missing or invalid, treat that row as if it is due now.
    """
    df = read_spaced_df(csv_path)

    # Convert Next Touch to a datetime object for sorting.
    # If invalid or empty, default to datetime.min so it sorts earliest (due now).
    def safe_parse_next_touch(x):
        try:
            return parse_date(x)
        except Exception:
            return datetime.min

    df["Parsed Next Touch"] = df["Next Touch"].apply(safe_parse_next_touch)

    # Sort ascending by Next Touch
    df = df.sort_values("Parsed Next Touch", ascending=True)

    # Return just the topics in order
    return [df["Topic"].tolist(), df["Parsed Next Touch"]]


###############################################################################
# Example usage / demonstration (uncomment to test)
###############################################################################
if __name__ == "__main__":
    csv_file = "my_spaced_data.csv"
    # add(csv_file, 
    #     category="Kadane's Algorithm",
    #     problem_name="Maximum Subarray",
    #     difficulty=2,
    #     listed_difficulty="medium"
    # )
    # # Adding a new problem solution
    # add(csv_file, 
    #     category="Arrays", 
    #     problem_name="Insert Delete GetRandom O(1)", 
    #     difficulty=2, 
    #     listed_difficulty="medium",
    #     # last_touch=None  # or "2025/01/17/12/13/00"
    # )
    # # See which topics are due next
    study_list = getstudytopics(csv_file)
    topics, due_dates = study_list[0], study_list[1]

    # Now print them in a neat, sorted order:
    print("\nExact order of topics to study next:\n")
    for i, (topic, due_date) in enumerate(zip(topics, due_dates), start=1):
        print(f"{i}. {topic} -> Next review on {due_date}")
    
    # df = read_spaced_df(csv_path=csv_file)
    # print(df.columns)
    # print(df[['Problems Solved', 'Problem Listed Difficulties']]) 
    
    # next time you want to do pandas, hard exercize would be to pivot your data about the lsit of problems solved. you should make that function.
