import argparse
import pickle
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=0, choices=[0, 1, 2], help='Data Mode (state, action, state-action)')
    parser.add_argument('-s', type=int, default=0, choices=[0, 1, 2], help='Seed')
    parser.add_argument('-t', action='store_true', help='MCTS')
    path = ''
    args = parser.parse_args()
    if (args.t):
        pass
    else:
        total_embeddings = []
        for i in range(8):
            with open(os.path.join(path, rf'RL_mode{args.m}\Seed{args.s}_Process{i}_EmbeddingData.pkl'), 'rb') as f:
                temp_embeddings = pickle.load(f)
                total_embeddings.extend(temp_embeddings)
            with open(os.path.join(path, rf'RL_mode{args.m}\seed_{args.s}_mode_{args.m}_embeddings.pkl'), 'wb') as f:
                pickle.dump(total_embeddings, f)

if __name__ == "__main__":
    main()
