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
        qvalues_padded = [0, 0, 0, 0, 0, 0, 0]
        for q in qvalues:
            qvalues_padded[q[0][1]] = q[1]
        i += 1
        if i % 1000 == 0:
            print(f"{i} done")
        embeddings_qvalues.append((torch.randn(512), torch.tensor(qvalues_padded)))
    print(len(embeddings_qvalues))
    with open(f'random_embeddings_mcts.pkl', 'wb') as f:
        pickle.dump(embeddings_qvalues, f)


def main():
    path = ''
    with open(os.path.join(path, rf'training_data/mcts/mcts_vals_mode_0.pkl'), 'rb') as f:
        qagent = pickle.load(f)

    generate_embeddings(qagent)


if __name__ == "__main__":
    main()
