# import dqn
import os
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")
from Connect4 import ConnectFourBoard 
from TreeSearch import MCTS
from dqn import DQNAgent, get_random_move, get_valid_locations
import torch
import numpy as np
import random 
from copy import deepcopy

def target_policy(game, tree):
    prob_dist = np.zeros(7, dtype=np.float32)
    valid_moves = [x[1] for x in get_valid_locations(game.board)]
    for i in range(len(valid_moves)):
        prob_dist[valid_moves[i]] = tree.N[tree.children[game][i]]
    prob_dist /= np.sum(prob_dist)
    return torch.tensor(prob_dist)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', action='store_true', help='Train Mode')
    parser.add_argument('-s', action='store_true', help='Store Data')
    parser.add_argument('-g', type=int, default=1000000, help="Number Of Games")
    parser.add_argument('-r', type=int, default=1, help="seed")
    args = parser.parse_args()
   
    if args.t:
        torch.manual_seed(args.r)
        np.random.seed(args.r)
        random.seed(args.r)
        torch.cuda.manual_seed(args.r)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        env = MCTS()
        for i in tqdm(range(0, args.g + 1)):
            env.reset()
            game = ConnectFourBoard()
            full_game = [game.board, game.board]
            play_data = []
            j = 0
            while True:
                for _ in range(50):
                    env.rollout(game)
                best_child = env.choose(game)[0]
                best_move = best_child.last_move
                full_game.append(game.board)
                play_data.append([deepcopy(torch.tensor(full_game[j:j+3])), target_policy(game, env), 0, game.last_move, best_move])
                game = game.make_move(best_move[1])
                if game.is_terminal():
                    break
                j += 1
            value = float(game.find_reward(game.turn))
            for game_state in play_data:
                value = -value
                game_state[2] = value
                env.policy_network.update_values(game_state, 0, args.s)
                env.value_network.update_values(game_state, 1, args.s)
            env.policy_network.reset(i, 0)
            env.value_network.reset(i, 1)
        
    else: 
        env = MCTS()
        results = []
        for i in tqdm(range(0, 1000)):
            env.reset()
            game = ConnectFourBoard()
            while True:
                for _ in range(50):
                    env.rollout(game)
                best_child = env.choose(game)[0]
                best_move = best_child.last_move
                game = game.make_move(best_move[1])
                if game.is_terminal():
                    results.append(game.find_winner(game.board, game.last_move[0], game.last_move[1]))
                    break
                move_2 = get_random_move(game.board)
                game = game.make_move(move_2)
                if game.is_terminal():
                    results.append(game.find_winner(game.board, game.last_move[0], game.last_move[1]))
                    break
        print(results.count(1))
        print(results.count(-1))
        print(results.count(None))

if __name__ == "__main__":
    main()
