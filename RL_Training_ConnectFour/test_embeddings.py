import pickle
import torch

with open('seed_0_process_0_mode_1_embeddings.pkl', 'rb') as f:
    test = pickle.load(f)
    print(len(test))
    print(test[0])
    print(test[-1])