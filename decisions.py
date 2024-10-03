import pickle
import torch
import sys
import os
sys.path.append('./')
sys.path.append('../')
sys.path.append('./RL_Training_ConnectFour')

from probe_parallel import LinearProbe, NonLinearProbe
from model import Config, GPTModel
from RL_Training_ConnectFour.env import Connect4Game
from RL_Training_ConnectFour.dqn import get_random_move, get_valid_locations
from tqdm import tqdm

def take_turns(model, layer, probe, device, token_to_idx, mode):
    X = []
    env = Connect4Game()
    move_1 = get_random_move(env.state.board)
    _, done = env.step(move_1, 1)
    if mode != 1:
        X.append(env.state.last_move)
    else:
        X.append(env.state.last_move[1])
    while True:
        move_2 = get_random_move(env.state.board)
        _, done = env.step(move_2, -1)
        # for row in env.state.board:
        #     print(row)
        # print("_________________________")
        if done: #hit a terminal state
            # print("OVER")
            return(env.state.find_winner(env.state.board, env.state.last_move[0], env.state.last_move[1]))
        if mode != 1:
            X.append(env.state.last_move)
        else:
            X.append(env.state.last_move[1])
        
        X_idx = [token_to_idx[token] for token in X]
        X_idx = torch.tensor(X_idx, dtype=torch.long).to(device)
        X_idx = X_idx.unsqueeze(0)
        embedding = model(X_idx, layer)[:, len(X) - 1, :]
        pred = probe.predict(embedding, device)
        possible_actions = [xy[1] for xy in get_valid_locations(env.state.board)]
        mask = torch.full(pred.shape, float('-inf')).to('cuda')
        mask[0][possible_actions] = pred[0][possible_actions]
        move_1 = torch.argmax(mask).item()

        _, done = env.step(move_1, 1)
        # for row in env.state.board:
        #     print(row)
        # print("_________________________")
        if done:
            # print("OVER")
            return(env.state.find_winner(env.state.board, env.state.last_move[0], env.state.last_move[1]))
        if mode != 1:
            X.append(env.state.last_move)
        else:
            X.append(env.state.last_move[1])

def decisions_validate(probe_model_path, gpt_model_path, config, linear, seed, mode, layer, mcts, token_to_idx, random):
    success_count = 0
    total_attempts = 0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            
    if linear:
        probe = LinearProbe(7, 512).to(device)
    else:
        probe = NonLinearProbe(7, 512).to(device)
    probe.load_state_dict(torch.load(probe_model_path, map_location = device))

    model = GPTModel(config).to(device)
    model.load_state_dict(torch.load(gpt_model_path))

    itrs = 100

    for _ in tqdm(range(itrs)):
        total_attempts += 1
        try:
            x = take_turns(model, 6, probe, device, token_to_idx, mode)
            if x == 1:
                success_count += 1
        except (KeyError, AssertionError):
            continue

    success_rate = success_count / total_attempts   
    if not random:
        with open('decision_outputs.txt', 'a') as f:
            f.write(f'Winrate for probe mode {mode} seed {seed} layer {layer} linear {linear} mcts {mcts}: {success_rate}\n')
    else:
        with open('decision_outputs_random.txt', 'a') as f:
            f.write(f'Winrate for random probe seed {seed} linear {linear} mcts {mcts}: {success_rate}\n')

def main():
    for linear in [True, False]:
        #RL
        for s in tqdm(range(3)):
            for m in tqdm(range(2)):
                model_load_path = rf'transformers_trained/RL_mode{m}/best_model/model_mode_{m}_seed_{s}.pth'
                if m == 0:
                    token_to_idx = {(i, j): i * 7 + j + 1 for i in range(6) for j in range(7)}
                    vocab_size = 43
                elif m == 1:
                    token_to_idx = {i: i + 1 for i in range(7)}
                    vocab_size = 8
                elif m == 2:
                    token_to_idx = {(i, j): i * 7 + j + 1 for i in range(6) for j in range(7)} | {i: i + 44 for i in range(7)}
                    vocab_size = 51
                token_to_idx['<pad>'] = 0 
                block_size = 42 
                embed_size = 512
                num_layers = 8
                config = Config(vocab_size, block_size, n_layer=num_layers, n_head=num_layers, n_embd=embed_size)
                for layer in tqdm(range(8)):
                    probe_load_path = rf'transformers_trained/RL_mode{m}/probe_data/{linear}best_model_mode_{m}_seed_{s}_layer_{layer}.pth'
                    if not os.path.exists(probe_load_path):
                        probe_load_path = rf'transformers_trained/RL_mode{m}/probe_data/{linear}best_model_mode_{m}_seed_{s+1}_layer_{layer}.pth'
                    decisions_validate(probe_load_path, model_load_path, config, linear, s, m, layer, False, token_to_idx, False)
        #MCTS
        for s in tqdm(range(3)):
            for m in tqdm(range(2)):
                model_load_path = rf'transformers_trained_mcts/mcts_mode{m}/best_model/model_mode_{m}_seed_{s}.pth'
                if m == 0:
                    token_to_idx = {(i, j): i * 7 + j + 1 for i in range(6) for j in range(7)}
                    vocab_size = 43
                elif m == 1:
                    token_to_idx = {i: i + 1 for i in range(7)}
                    vocab_size = 8
                elif m == 2:
                    token_to_idx = {(i, j): i * 7 + j + 1 for i in range(6) for j in range(7)} | {i: i + 44 for i in range(7)}
                    vocab_size = 51
                token_to_idx['<pad>'] = 0 
                block_size = 42 
                embed_size = 512
                num_layers = 8
                config = Config(vocab_size, block_size, n_layer=num_layers, n_head=num_layers, n_embd=embed_size)
                for layer in tqdm(range(8)):
                    probe_load_path = rf'transformers_trained_mcts/mcts_mode{m}/probe_data/{linear}best_model_mode_{m}_seed_{s}_layer_{layer}.pth'
                    if not os.path.exists(probe_load_path):
                        probe_load_path = rf'transformers_trained_mcts/mcts_mode{m}/probe_data/{linear}best_model_mode_{m}_seed_{s+1}_layer_{layer}.pth'
                    decisions_validate(probe_load_path, model_load_path, config, linear, s, m, layer, True, token_to_idx, False)
        #Random RL 
        for s in tqdm(range(3)):
            model_load_path = rf'transformers_trained/RL_mode0/best_model/model_mode_0_seed_{s}.pth'
            token_to_idx = {(i, j): i * 7 + j + 1 for i in range(6) for j in range(7)}
            vocab_size = 43
            token_to_idx['<pad>'] = 0 
            block_size = 42 
            embed_size = 512
            num_layers = 8
            config = Config(vocab_size, block_size, n_layer=num_layers, n_head=num_layers, n_embd=embed_size)
            probe_load_path = rf'random_probes/rl/{linear}best_model_seed_{s}.pth'
            decisions_validate(probe_load_path, model_load_path, config, linear, s, 0, None, False, token_to_idx, True)

        for s in tqdm(range(3)):
            model_load_path = rf'transformers_trained_mcts/mcts_mode0/best_model/model_mode_0_seed_{s}.pth'
            token_to_idx = {(i, j): i * 7 + j + 1 for i in range(6) for j in range(7)}
            vocab_size = 43
            token_to_idx['<pad>'] = 0 
            block_size = 42 
            embed_size = 512
            num_layers = 8
            config = Config(vocab_size, block_size, n_layer=num_layers, n_head=num_layers, n_embd=embed_size)
            probe_load_path = rf'random_probes/mcts/{linear}best_model_seed_{s}.pth'
            decisions_validate(probe_load_path, model_load_path, config, linear, s, 0, None, True, token_to_idx, True)
            

if __name__ == '__main__':
    main()
