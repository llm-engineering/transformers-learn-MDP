# import dqn
import os
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")
from env import Connect4Game
from dqn import DQNAgent, get_random_move
import torch
import numpy as np
import random 

def load_or_initialize_model(model_path, env, params):
    model = DQNAgent(env, params)
    if os.path.exists(model_path):
        model.dqn.load_state_dict(torch.load(model_path))
    return model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', action='store_true', help='Train Mode')
    parser.add_argument('-s', action='store_true', help='Store Data')
    parser.add_argument('-g', type=int, default=1000000, help="Number Of Games")
    args = parser.parse_args()

    params_1 = {
        "N": 1000,
        "M": 10000,
        "gamma": 0.9,
        "epsilon": 0.1,
        "minibatch_size": 32,
        "learning_rate": 1e-5,
        "best_model": None,
        "player": 1
    }

    params_2 = {
        "N": 1000,
        "M": 10000,
        "gamma": 0.9,
        "epsilon": 0.1,
        "minibatch_size": 32,
        "learning_rate": 1e-5,
        "best_model": None,
        "player": -1
    }

    env = Connect4Game()
    agent_1 = load_or_initialize_model("agent_1.pth", env, params_1)
    agent_2 = load_or_initialize_model("agent_2.pth", env, params_2)
   
    if (args.t):
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        agent_1.prepare()
        agent_2.prepare()

        for i in tqdm(range(0, args.g)):
            move_1 = None
            move_2 = None
            output_1 = None
            output_2 = None
            while True:
                output_1, move_1 = agent_1.get_action(True)
                reward_1, done = env.step(move_1, 1)
                reward_2 = env.find_reward(2)
                if done: #hit a terminal state, update Q values for both agents
                    agent_1.update_values(env.state_history[-2], env.state_history[-1], move_1, reward_1, done, output_1, args.s)
                    agent_2.update_values(env.state_history[-3], env.state_history[-1], move_2, reward_2, done, output_2, args.s)
                    break
                elif move_2 is not None: #we calculate states temporal difference as after both agents make a move, so we update for agent_2 since agent_2 just regained control
                    agent_2.update_values(env.state_history[-3], env.state_history[-1], move_2, reward_2, done, output_2, args.s)
                agent_2.update_env(env)
                output_2, move_2 = agent_2.get_action(True)
                reward_2, done = env.step(move_2, 2)
                reward_1 = env.find_reward(1)
                if done: #hit a terminal state, update Q values for both agents
                    agent_1.update_values(env.state_history[-3], env.state_history[-1], move_1, reward_1, done, output_1, args.s)
                    agent_2.update_values(env.state_history[-2], env.state_history[-1], move_2, reward_2, done, output_2, args.s)
                    break
                else:
                    agent_1.update_values(env.state_history[-3], env.state_history[-1], move_1, reward_1, done, output_1, args.s)
                agent_1.update_env(env)
            env.reset()
            agent_1.reset(env)
            agent_2.reset(env)
            if i % 500 == 0: 
                torch.save(agent_1.dqn.state_dict(), 'agent_1.pth')
                torch.save(agent_2.dqn.state_dict(), 'agent_2.pth')
            if i % 5000 == 0: 
                agent_1.record(i)
                agent_2.record(i)
        torch.save(agent_1.dqn.state_dict(), 'agent_1.pth')
        torch.save(agent_2.dqn.state_dict(), 'agent_2.pth') 
        
    else: 
        results = []
        for i in tqdm(range(0, 1000)):
            while True:
                _, move_1 = agent_1.get_action(False)
                reward_1, done = env.step(move_1, 1)
                if done: #hit a terminal state
                    results.append(env.state.find_winner(env.state.board, env.state.last_move[0], env.state.last_move[1]))
                    break
                move_2 = get_random_move(env.state.board)
                reward_2, done = env.step(move_2, 2)
                if done: #hit a terminal state
                    results.append(env.state.find_winner(env.state.board, env.state.last_move[0], env.state.last_move[1]))
                    break
                agent_1.update_env(env)
            env.reset()
            agent_1.reset(env, i, False)
        print(results.count(1))
        print(results.count(2))
        print(results.count(None))

if __name__ == "__main__":
    main()
