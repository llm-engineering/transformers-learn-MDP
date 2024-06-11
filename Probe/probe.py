import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os

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

class ProbeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], torch.tensor(self.y[idx])

def train_probe(train_dataset : Dataset, valid_dataset : Dataset, device, epochs : int, params, model_dir, linear):

    d, n = params
    min_loss = float('inf')
    best_epoch = 0

    criterion = nn.MSELoss()

    if linear: 
        probe = LinearProbe(d, n).to(device)
    else: 
        probe = NonLinearProbe(d, n).to(device)

    optimizer = optim.Adam(probe.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.7)

    torch.manual_seed(0)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True)

    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, 'best_model.pth')

    train_loss_hist = []
    valid_loss_hist = []

    for epoch in range(epochs):

        probe.train()
        train_loss = 0.0
        train_data = 0

        for X, y in train_loader:
            X, y = X.float().to(device), y.float().to(device)
            preds = probe(X)

            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() * X.shape[0]
            train_data += X.shape[0]

        scheduler.step()
        valid_loss = validate_probe(probe, valid_loader, device)

        if valid_loss < min_loss:
            min_loss = valid_loss
            best_epoch = epoch
            torch.save(probe.state_dict(), best_model_path)

        train_loss_hist.append(train_loss / train_data)
        valid_loss_hist.append(valid_loss)

    print(f"Best Epoch: {best_epoch + 1}, Min MSE Loss: {min_loss}")
    return best_model_path, train_loss_hist, valid_loss_hist

def validate_probe(probe, valid_loader, device):

    criterion = nn.MSELoss()
    probe.eval()
    total_loss = 0.0
    total_data = 0

    with torch.no_grad():
        for X, y in valid_loader:
            X, y = X.to(device), y.to(device)
            preds = probe(X)
            loss = criterion(preds, y)

            total_loss += loss.item() * X.shape[0]
            total_data += X.shape[0]

    average_loss = total_loss / total_data
    return average_loss

def test_probe(test_dataset, model_path, params, device, linear):

    d, n = params

    if linear: 
        probe = LinearProbe(d, n)
    else : 
        probe = NonLinearProbe(d, n)

    probe.load_state_dict(torch.load(model_path))
    probe.to(device)
    probe.eval()

    criterion = nn.MSELoss()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    total_loss = 0.0
    total_data = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.float().to(device), y.float().to(device)
            preds = probe(X)
            loss = criterion(preds, y)
            total_loss += loss.item() * X.shape[0]
            total_data += X.shape[0]

    average_mse = total_loss / total_data
    return average_mse