import torch

from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence

class GameDataset(Dataset):
    
    def __init__(self, data, token_to_idx):
        self.data = data
        self.token_to_idx = token_to_idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        X_sequence, Y_sequence = self.data[idx]

        X_indices = [self.token_to_idx[token] for token in X_sequence]
        Y_indices = [self.token_to_idx[token] for token in Y_sequence]

        return torch.tensor(X_indices, dtype=torch.long), torch.tensor(Y_indices, dtype=torch.long)

def collate_fn(batch):

    Xs, Ys = zip(*batch)

    Xs_padded = pad_sequence(Xs, batch_first=True, padding_value=0)
    Ys_padded = pad_sequence(Ys, batch_first=True, padding_value=0)

    return Xs_padded, Ys_padded
