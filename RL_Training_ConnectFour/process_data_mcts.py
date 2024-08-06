import pickle
import os
import math
import argparse
from tqdm import tqdm

#path should be what the directory of data is 
path = r'C:\Users\wmasi\Documents\TF-Agent-States\RL_Training_ConnectFour'

def load_data(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

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

    # mcts_vals = []
    # for i in range(length):
    #     mcts_vals.append([full_game[i], episode[i][1]])
    if mode != 2:
        # return (full_game[:-1], full_game[1:]), mcts_vals
        return (full_game[:-1], full_game[1:])
    else:
        # return (full_game[:-1], full_game_2), mcts_vals
        return (full_game[:-1], full_game_2)
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=0, choices=[0, 1, 2], help='Data Mode (state, action, state-action)')
    args = parser.parse_args()

    new_path_first_half = path
    train_output_path_second_half = rf'training_data\mcts\training_games'
    train_output_path = os.path.join(new_path_first_half, train_output_path_second_half)
    q_output_path_second_half = rf'training_data\mcts\mcts_vals_games'
    q_output_path = os.path.join(new_path_first_half, q_output_path_second_half)
    for i in tqdm(range(100)):
        training_hist = []
        mcts_hist = []
        new_path_second_half_1 = rf'Archive\Simulations_Batch{i}.pkl'
        games = load_data(os.path.join(new_path_first_half, new_path_second_half_1))
        for j in range(len(games)):
            # temp_training_hist, temp_mcts_hist = generate_xy_pair(games[j], args.m)
            temp_training_hist = generate_xy_pair(games[j], args.m)
            training_hist.append(temp_training_hist)
            # mcts_hist.append(temp_mcts_hist)
        with open(train_output_path + "_batch_" + str(i) + "_mode_" + str(args.m) + '.pkl', 'wb') as f:
            pickle.dump(training_hist, f)
        # with open(q_output_path + "_batch_" + str(i) + "_mode_" + str(args.m) + '.pkl', 'wb') as f:
        #     pickle.dump(mcts_hist, f)
    full_training_hist = []
    full_mcts_hist = []
    for i in tqdm(range(100)):
        with open(train_output_path + "_batch_" + str(i) + "_mode_" + str(args.m) + '.pkl', 'rb') as f:
            temp_training_hist = pickle.load(f)
            full_training_hist.extend(temp_training_hist)
        # with open(q_output_path + "_batch_" + str(i) + "_mode_" + str(args.m) + '.pkl', 'rb') as f:
        #     temp_mcts_hist = pickle.load(f)
        #     full_mcts_hist.extend(temp_mcts_hist)
    with open(rf'training_data\mcts\training_games_mode_{args.m}.pkl', 'wb') as f:
        pickle.dump(full_training_hist, f)
    # with open(rf'training_data\mcts\mcts_vals_mode_{args.m}.pkl', 'wb') as f:
    #     pickle.dump(full_mcts_hist, f)
           
if __name__ == "__main__":
    main()