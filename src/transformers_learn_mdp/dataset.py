import torch

from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
import tqdm 


class EpisodeDataset(Dataset):
    
    def __init__(self, data, token_to_idx, packing_length=30,padding_value=0):
        self.token_to_idx = token_to_idx
        print("Tokenizing and packing the dataset")
        self.packed_data = []

        self.tokenized_data = [[51] + [self.token_to_idx[token] for token in sequence] + [51] for sequence in data]
        # flatten the list and insert padding value at the end of each sequence
        #self.data = []
        #for sequence in self.tokenized_data:
        #    self.data.extend(sequence)
        #    self.data.append(padding_value)
        #del self.tokenized_data
        #self.data = [self.data[i:i+packing_length] for i in range(0, len(self.data), packing_length)]
        self.data = self.tokenized_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        sequence = self.data[idx]

        X_indices =  sequence[:-1]
        Y_indices =  sequence[1:]

        return torch.tensor(X_indices, dtype=torch.long), torch.tensor(Y_indices, dtype=torch.long)

def collate_fn(batch):

    Xs, Ys = zip(*batch)

    Xs_padded = pad_sequence(Xs, batch_first=True, padding_value=0)
    Ys_padded = pad_sequence(Ys, batch_first=True, padding_value=0)

    return Xs_padded, Ys_padded