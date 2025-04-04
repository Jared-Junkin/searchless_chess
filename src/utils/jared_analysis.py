import pandas as pd
import os
from typing import List, Tuple
import numpy as np

# determines the win rate across all opponents. currently opponent string is configured to be stockfish
def calcWinRate(df: pd.DataFrame, stockfish_range=range(11), agent_color: str = "white")->dict:
    if agent_color == "white":
        matchups = df.groupby("player_two") # group by stockfish level
    elif agent_color == "black":
        matchups = df.groupby("player_one") # group by stockfish level
    else: 
        raise Exception(f"Error: value for agent color (entered '{agent_color}') must be one of: ('black', 'white')")
    results = {}
    # group_names = matchups.groups.keys()
    # print(list(group_names))
    for level in stockfish_range:
        opponent = "Stockfish " + str(level)
        df_group = matchups.get_group(opponent)
        if agent_color == "white":
            wins = sum(df_group["result"].str.count("1-0"))
            draws = sum(df_group["result"].str.count("1/2-1/2"))
            losses = sum(df_group["result"].str.count("0-1"))
        elif agent_color == "black":
            losses = sum(df_group["result"].str.count("1-0"))
            draws = sum(df_group["result"].str.count("1/2-1/2"))
            wins = sum(df_group["result"].str.count("0-1"))
        print(f"Results against Stockfish Level {level}: {wins} wins, {draws} draws, {losses} losses")
        results[opponent] = (wins, draws, losses)
    return results


def calcELO(results: dict)-> float:
    # source: https://github.com/official-stockfish/Stockfish/commit/a08b8d4
    stockfish_ratings = {
        "Stockfish 0": 1320,
        "Stockfish 1": 1467,
        "Stockfish 2": 1608,
        "Stockfish 3": 1742,
        "Stockfish 4": 1922,
        "Stockfish 5": 2203,
        "Stockfish 6": 2363,
        "Stockfish 7": 2499,
        "Stockfish 8": 2596,
        "Stockfish 9": 2702,
        "Stockfish 10": 2788
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
        filepath = "../logs/ckpt200000_pt_vs_stockfish" + str(i) + ".csv"
        print(filepath)
        df = pd.read_csv(filepath)
        
        filepath = "../logs/stockfish" + str(i) + "_vs_ckpt200000_pt" + ".csv"
        df_black = pd.read_csv(filepath)
        results = calcWinRate(df=df, stockfish_range=[i])
        results_black = calcWinRate(df_black, stockfish_range=[i], agent_color="black")
        results_final['Stockfish '+str(i)] = tuple([results['Stockfish '+str(i)][j] + results_black['Stockfish '+str(i)][j] for j in range(len(results['Stockfish '+str(i)]))])
        
        print(F"results for {i} iterations: {results} in white + {results_black} in black = {results_final}")
        with open(write_file, 'a') as f:
            f.write(f"{i}, {results_final['Stockfish '+str(i)][0]}, {results_final['Stockfish '+str(i)][1]}, {results_final['Stockfish '+str(i)][2]}\n")
            # f.write(f"{Checkpoint: {i}, Wins: {results['Stockfish '+str(i)][0]}, Draws: {results['Stockfish '+str(i)][1]}, Losses: {results['Stockfish '+str(i)][2]}}\n")
            f.close()
    print(results_final)
    rating_final = calcELO(results=results_final)
    print(f"\n\nfinal estimated rating of LLM agent is {rating_final}\n")