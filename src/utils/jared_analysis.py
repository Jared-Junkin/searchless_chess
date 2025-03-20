import pandas as pd
import os
from typing import List, Tuple
import numpy as np

# determines the win rate across all opponents. currently opponent string is configured to be stockfish
def calcWinRate(df: pd.DataFrame, stockfish_level: int, color: str = "white")->dict:
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
        N = sum(results[opponent]) # total matches played
        wins = results[opponent][0] + results[opponent][1]*0.9
        expected_wins = 1/(1 + 10**((stockfish_ratings[opponent]-start_estimated_rating)/400))
        start_estimated_rating += K_factor * (wins - (N * expected_wins))
        print(f"estimated rating after {N} matches against {opponent}: {start_estimated_rating}")
    return start_estimated_rating
    # there's no way this thing is that strong. clearly I need to play it.

# def calcOneWin(df: pd.DataFrame)->Tuple[int, int, int]:
#     test = df

if __name__ == "__main__":
    
    ############ Code to Calculate Wins across all stockish levels for a given agent whose games are stored at filepath ############
    # filepath = "../logs/stockfish_16layers_ckpt_no_optimizer_pt_vs_stockfish_sweep_5_sec.csv" # this is using the character level language model I trained frmo scratch (estimate ELO 2109.)
    # df = pd.read_csv(filepath)
    # levels = range(1)
    # results = calcWinRate(df=df, stockfish_range=levels)
    # print(F"results: {results}")
    # rating_final = calcELO(results)
    # print(rating_final)
    ############ Code to Calculate Wins across all stockish levels for a given agent whose games are stored at filepath ############
    
    ############ Code to calculate wins for one stockfish level for a given agent. ############
    write_file="./stockfish_sweep.txt"
    results_final = {}
    with open(write_file, 'a') as f:
        f.write("level,wins,draws,losses\n")
        f.close()
    
    for i in range(11):
        filepath = "../pythia/logs/llama3_1repeat50000_vs_stockfish" + str(i) + ".csv"
        df = pd.read_csv(filepath)
        white_perf = calcWinRate(df, stockfish_level=i, color="white")
        
        
        blackFilepath = "../pythia/logs/stockfish" + str(i) + "_vs_llama3_1repeat50000.csv"
        df = pd.read_csv(blackFilepath)
        black_perf = calcWinRate(df, stockfish_level=i, color="black")
        
        # df = pd.concat([df, df_black], ignore_index=True)
        results_final[f"stockfish{i}"] = (black_perf[0]+white_perf[0], black_perf[1]+white_perf[1], black_perf[2]+white_perf[2])
        with open(write_file, 'a') as f:
            f.write(f"{i},{results_final['stockfish'+str(i)][0]},{results_final['stockfish'+str(i)][1]},{results_final['stockfish'+str(i)][2]}\n")
            f.close()
    print(results_final)
    rating_final = calcELO(results=results_final)
    # print(f"final estimated rating is {rating_final}")