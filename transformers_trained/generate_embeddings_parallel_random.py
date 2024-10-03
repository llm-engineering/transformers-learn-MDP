import torch
import os
import pickle

def generate_embeddings(dataset):
    embeddings_qvalues = []
    i = 0
    for data in dataset:
        if len(data) == 0:
            continue
        qvalues = data[0][1]
        i += 1
        if i % 1000 == 0:
            print(f"{i} done")
        embeddings_qvalues.append((torch.randn(512), torch.tensor(list(qvalues.values()), dtype=torch.float32)))

    print(len(embeddings_qvalues))
    with open(f'random_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings_qvalues, f)

def main():
    path = ''
    path = ''
    with open(os.path.join(path, rf'training_data/1/q_vals_games_1000000_mode_0.pkl'), 'rb') as f:
        qagent1 = pickle.load(f)
    with open(os.path.join(path, rf'training_data/2/q_vals_games_1000000_mode_0.pkl'), 'rb') as f:
        qagent2 = pickle.load(f)
    with open(os.path.join(path, rf'training_data/3/q_vals_games_1000000_mode_0.pkl'), 'rb') as f:
        qagent3 = pickle.load(f)
    with open(os.path.join(path, rf'training_data/5/q_vals_games_1000000_mode_0.pkl'), 'rb') as f:
        qagent5 = pickle.load(f)
    with open(os.path.join(path, rf'training_data/6/q_vals_games_1000000_mode_0.pkl'), 'rb') as f:
        qagent6 = pickle.load(f)
    with open(os.path.join(path, rf'training_data/7/q_vals_games_1000000_mode_0.pkl'), 'rb') as f:
        qagent7 = pickle.load(f)
    with open(os.path.join(path, rf'training_data/8/q_vals_games_1000000_mode_0.pkl'), 'rb') as f:
        qagent8 = pickle.load(f)
    with open(os.path.join(path, rf'training_data/9/q_vals_games_1000000_mode_0.pkl'), 'rb') as f:
        qagent9 = pickle.load(f)
    with open(os.path.join(path, rf'training_data/10/q_vals_games_1000000_mode_0.pkl'), 'rb') as f:
        qagent10 = pickle.load(f)

    qagent = qagent1 + qagent2 + qagent3 + qagent5 + qagent6 + qagent7 + qagent8 + qagent9 + qagent10
    generate_embeddings(qagent)


if __name__ == "__main__":
    main()