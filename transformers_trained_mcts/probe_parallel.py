import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, Dataset
from accelerate import load_checkpoint_and_dispatch
from tqdm import tqdm
import os
import gc
import shutil

from accelerate import Accelerator

class ProbeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]
    
class ArgmaxDifferenceLoss(nn.Module):
    def __init__(self):
        super(ArgmaxDifferenceLoss, self).__init__()
    
    def forward(self, batch1, batch2):
        argmax1 = torch.argmax(batch1, dim=1)
        argmax2 = torch.argmax(batch2, dim=1)
        
        loss = (argmax1 != argmax2).float()
        
        return loss.mean()
    
class NonLinearProbe(nn.Module):
    def __init__(self, d, n):
        super(NonLinearProbe, self).__init__()
        self.ln1 = nn.Linear(n, n)
        self.act1 = nn.ReLU()
        self.ln2 = nn.Linear(n, d)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.act1(self.ln1(x))
        x = self.ln2(x)
        return x

    def predict(self, X, device):
        with torch.no_grad():
            X = X.to(device)
            pred = self.forward(X)
            return pred.cpu()
        
class LinearProbe(nn.Module):
    def __init__(self, d, n):
        super(LinearProbe, self).__init__()
        self.ln1 = nn.Linear(n, d)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.ln1(x)
        return x

    def predict(self, X, device):
        with torch.no_grad():
            X = X.to(device)
            pred = self.forward(X)
            return pred.cpu()
        
def train_probe(train_dataset : Dataset, valid_dataset : Dataset, epochs : int, params, linear, mode, seed, layer, accelerator):
    
    d, n = params
    min_loss = float('inf')
    best_epoch = 0

    criterion = nn.MSELoss()

    if linear: 
        probe = LinearProbe(d, n).to(accelerator.device)
    else: 
        probe = NonLinearProbe(d, n).to(accelerator.device)

    optimizer = optim.Adam(probe.parameters(), lr=0.0001, weight_decay=1e-5)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True)

    probe, train_loader, valid_loader = accelerator.prepare(probe, train_loader, valid_loader)

    train_loss_hist = []
    valid_loss_hist = []

    model_path = ''

    for epoch in tqdm(range(epochs)):

        probe.train()
        train_loss = torch.tensor(0.0).to(accelerator.device)
        train_data = torch.tensor(0.0).to(accelerator.device)

        for X, y in train_loader:

            preds = probe(X)

            loss = criterion(preds, y)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() * X.shape[0]
            train_data += X.shape[0]

        accelerator.wait_for_everyone()

        train_loss = accelerator.gather(train_loss).sum()
        train_data = accelerator.gather(train_data).sum()

        valid_loss = validate_probe(probe, valid_loader, accelerator)

        train_loss_hist.append(train_loss / train_data)
        valid_loss_hist.append(valid_loss)

        best_model_path = str(linear) + f"best_model_mode_{mode}_seed_{seed}_layer_{layer}.pth"
        model_save_path = str(linear) + f"_model_{epoch+1}_mode_{mode}_seed_{seed}_layer_{layer}.pth"

        if accelerator.is_main_process:

            print(f'Validation Loss: {valid_loss:.8f}')

            accelerator.save(accelerator.unwrap_model(probe).state_dict(), model_save_path)

        if valid_loss < min_loss:
            min_loss = valid_loss
            model_path = model_save_path

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            with open(str(linear) + f'_train_losses_mode_{mode}_seed_{seed}_layer_{layer}.pkl', 'wb') as f:
                pickle.dump(train_loss_hist, f)
            with open(str(linear) + f'_valid_losses_mode_{mode}_seed_{seed}_layer_{layer}.pkl', 'wb') as f:
                pickle.dump(valid_loss_hist, f)
            shutil.copy(model_path, best_model_path)

    #print(f"Best Epoch: {best_epoch + 1}, Min MSE Loss: {min_loss}, Model Path: {model_path}")

    return model_path, train_loss_hist, valid_loss_hist

def validate_probe(probe, valid_loader, accelerator):

    criterion = nn.MSELoss()
    probe.eval()
    total_loss = torch.tensor(0.0).to(accelerator.device)
    total_data = torch.tensor(0.0).to(accelerator.device)

    with torch.no_grad():

        for X, y in valid_loader:

            preds = probe(X)
            # print(f"Prediction; Size:{preds.size()}\n\n")
            # print(preds)
            # print(f"Targets; Size:{y.size()}\n\n")
            # print(y)
            loss = criterion(preds, y)

            total_loss += loss.item() * X.shape[0]
            total_data += X.shape[0]

    accelerator.wait_for_everyone()

    total_loss = accelerator.gather(total_loss).sum()
    total_data = accelerator.gather(total_data).sum()

    return total_loss / total_data

def test_probe(test_dataset, model_path, params, linear, accelerator):

    d, n = params

    total_loss = torch.tensor(0.0).to(accelerator.device)
    total_data = torch.tensor(0.0).to(accelerator.device)

    if linear: 
        probe = LinearProbe(d, n)
    else : 
        probe = NonLinearProbe(d, n)

    model = load_checkpoint_and_dispatch(
        probe,
        model_path,
        device_map="auto"
    )

    criterion = ArgmaxDifferenceLoss()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    with torch.no_grad():
        for X, y in test_loader:
            preds = probe(X)
            # print(y)
            loss = criterion(preds, y)
            total_loss += loss.item() * X.shape[0]
            total_data += X.shape[0]

    accelerator.wait_for_everyone()

    total_loss = accelerator.gather(total_loss).sum()
    total_data = accelerator.gather(total_data).sum()

    return total_loss / total_data


# def standardize_tensor(tensor):

#     mean = tensor.mean()
#     std = tensor.std()
    
#     standardized_tensor = (tensor - mean) / std
#     return standardized_tensor
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=0, choices=[0, 1, 2], help='Data Mode (state, action, state-action)')
    args = parser.parse_args()
    
    accelerator = Accelerator()
    path = ''
    for s in tqdm(range(3)):
        for layer in tqdm(range(8)):
            if os.path.isfile(f'False_train_losses_mode_{args.m}_seed_{s}_layer_{layer}.pkl'):
                if accelerator.is_main_process:
                    print(f"layer {layer} complete")
                continue
            total_embeddings = []
            for i in tqdm(range(8)):
                with open(os.path.join(path, rf'seed_{s}_process_{i}_mode_{args.m}_layer_{layer}_embeddings.pkl'), 'rb') as f:
                    temp_embeddings = pickle.load(f)
                    total_embeddings.extend(temp_embeddings)
            
            length = len(total_embeddings)
            del temp_embeddings
            train_ratio = 0.8
            valid_ratio = 0.1
            
            train = total_embeddings[:int(train_ratio * length)]
            valid = total_embeddings[int(train_ratio * length):int((train_ratio + valid_ratio) * length)]
            test = total_embeddings[int((train_ratio + valid_ratio) * length):] 

            del total_embeddings
            embeddings_train = [embedding[0].to('cuda:0') for embedding in train if embedding[1][0].size()[0] == 7]
            values_train = [embedding[1][0].to('cuda:0') for embedding in train if embedding[1][0].size()[0] == 7]
            del train
            embeddings_valid = [embedding[0].to('cuda:0') for embedding in valid if embedding[1][0].size()[0] == 7]
            values_valid = [embedding[1][0].to('cuda:0') for embedding in valid if embedding[1][0].size()[0] == 7]
            del valid
            embeddings_test = [embedding[0].to('cuda:0') for embedding in test if embedding[1][0].size()[0] == 7]
            values_test = [embedding[1][0].to('cuda:0') for embedding in test if embedding[1][0].size()[0] == 7]
            del test
            print(len(embeddings_train) + len(embeddings_valid) + len(embeddings_test))

            probe_dataset_train = ProbeDataset(embeddings_train, values_train)
            probe_dataset_valid = ProbeDataset(embeddings_valid, values_valid)
            probe_dataset_test = ProbeDataset(embeddings_test, values_test)
            del embeddings_train
            del values_train
            del embeddings_valid
            del values_valid
            del embeddings_test
            del values_test
            linear_probe_path, _, _ = train_probe(probe_dataset_train, probe_dataset_valid, 100, (7, 512), True, args.m, s, layer, accelerator)
            nonlinear_probe_path, _, _ = train_probe(probe_dataset_train, probe_dataset_valid, 100, (7, 512), False, args.m, s, layer, accelerator)
            test_loss_linear = test_probe(probe_dataset_test, linear_probe_path, (7, 512), True, accelerator)
            test_loss_nonlinear = test_probe(probe_dataset_test, nonlinear_probe_path, (7, 512), False, accelerator)
            if accelerator.is_main_process:
                with open('test_losses.txt', 'a') as file:
                    file.write(f"Layer {layer} Seed {s} Mode {args.m} losses: {test_loss_linear}, {test_loss_nonlinear}")
            del probe_dataset_train
            del probe_dataset_valid
            del probe_dataset_test
            exit()

if __name__ == '__main__':
    main()