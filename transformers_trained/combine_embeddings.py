import argparse
import pickle
import os
import torch
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=0, choices=[0, 1, 2], help='Data Mode (state, action, state-action)')
    parser.add_argument('-s', type=int, default=0, choices=[0, 1, 2], help='Seed')
    parser.add_argument('-t', action='store_true', help='MCTS')
    path = ''
    args = parser.parse_args()
    if args.t:
        print("mcts")
        total_embeddings = []
        for i in tqdm(range(8)):
            with open(os.path.join(path, rf'seed_{args.s}_process_{i}_mode_{args.m}_embeddings.pkl'), 'rb') as f:
                temp_embeddings = pickle.load(f)
                total_embeddings.extend(temp_embeddings)
            with open(os.path.join(path, rf'seed_{args.s}_mode_{args.m}_embeddings.pkl'), 'wb') as f:
                pickle.dump(total_embeddings, f)
    else:
        print("No mcts")
        total_embeddings = []
        for i in tqdm(range(8)):
            with open(os.path.join(path, rf'seed_{args.s}_process_{i}_mode_{args.m}_embeddings.pkl'), 'rb') as f:
                temp_embeddings = pickle.load(f)
                total_embeddings.extend(temp_embeddings)
            with open(os.path.join(path, rf'seed_{args.s}_mode_{args.m}_embeddings.pkl'), 'wb') as f:
                pickle.dump(total_embeddings, f)

if __name__ == "__main__":
    main()
