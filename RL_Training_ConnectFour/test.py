import os
from tqdm import tqdm
import pickle
def generate_xy_pair(episode, mode):
    # Generates XY Pairs
    # By Shifting Indices 
    # Off-By-One
    full_game = []
    full_game_2 = []
    length = len(episode) #10000
    for i in range(length):
        if mode == 0 or mode == 2:
            full_game.append(episode[i][0])
        elif mode == 1:
            full_game.append(episode[i][0][1])
    if mode == 2:
        full_game_2 = [x[1] for x in full_game][1:]

    mcts_vals = []
    for i in range(length):
        mcts_vals.append([full_game[i], episode[i][1]])
    if mode != 2:
        return (full_game[:-1], full_game[1:]), mcts_vals
    else:
        return (full_game[:-1], full_game_2), mcts_vals

for i in range(1, 10):
    if i == 4:
        continue
    with open(rf'training_data/mcts/mcts_vals_mode_0.pkl', 'rb') as f:
        test = pickle.load(f)
        print(test[0])
    # x, y = generate_xy_pair(test[0], 0)
    # print(x) 
    # print(y)
    # x, y = generate_xy_pair(test[0], 1)
    # print(x) 
    # print(y)
    # x, y = generate_xy_pair(test[0], 2)
    # print(x) 
    # print(y)

