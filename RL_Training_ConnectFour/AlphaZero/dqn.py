from typing import Dict, Any, Tuple
from torchvision import transforms
from torch import nn
from torch import optim
from torch.nn import MSELoss
import torch.nn.functional as F
import os
import numpy as np
import numpy.typing as npt
import torch
import random
import pickle
from accelerate import Accelerator

def get_possible_locations(board):
    locations = []
    for col in range(7):
        for row in range(5, -1, -1):
            if board[row][col] == 0 or row == 0:
                locations.append((row, col))
                break
    return locations

def get_valid_locations(board):
    locations = []
    for col in range(7):
        for row in range(5, -1, -1):
            if board[row][col] == 0:
                locations.append((row, col))
                break
    return locations

def get_random_move(board):
    locations = []
    for col in range(7):
        for row in range(5, -1, -1):
            if board[row][col] == 0:
                locations.append(col)
                break
    return random.choice(locations)

class DQNAgent():
    def __init__(
            self, 
            output_channels
    ):        
        self.accelerator = Accelerator()
        self.dqn = DQN(output_channels).to('cuda')
        self.replay_buffer = ReplayMemory(
            capacity=1000,
            device=self.accelerator.device,
        )
        self.optimizer = optim.AdamW(
            self.dqn.parameters(),
            lr=0.001,
            betas=(0.9, 0.95),
        )
        self.criterion = MSELoss().to(self.accelerator.device)
        self.history = []
        self.total_history = []
        self.accelerator = Accelerator()

    def prepare(self):
        self.dqn, self.optimizer = self.accelerator.prepare(self.dqn, self.optimizer)

    # def get_action(self, random_enabled):
    #     output = self.dqn(torch.tensor(self.env.state.board))
    #     if random_enabled and random.random() < self.epsilon:
    #         action = self.env.action_space_sample()
    #     else:
    #         possible_actions = [xy[1] for xy in get_valid_locations(self.env.state.board)]
    #         mask = torch.full(output.shape, float('-inf')).to('cuda')
    #         mask[0][possible_actions] = output[0][possible_actions]
    #         action = torch.argmax(mask).item()
    #     return output, action

    def update_values(self, game_state, val_or_pol, to_log):
        self.replay_buffer.push(game_state[0], game_state[1], game_state[2])
        minibatch = self.replay_buffer.sample(32)
        outputs, labels = [], []
        for sample in minibatch:
            if val_or_pol == 0:
                target = sample["policy"]
            else:
                target = sample["value"]
            output = self.dqn(sample["current_state"])
            outputs.append(output)
            labels.append(target)
        # print(outputs)
        # print(labels)
        outputs = torch.stack(outputs).to(self.accelerator.device)
        if val_or_pol == 0: 
            labels = torch.stack(labels).to(self.accelerator.device)
        else: 
            labels = torch.tensor(labels).to(self.accelerator.device)
        loss = self.criterion(outputs, labels)
        #print(type(loss))
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.optimizer.step()

        if to_log:
            self.history.append((get_valid_locations(np.array(game_state[0])), game_state[-1]))

    
    def reset(self, i, val_or_pol):
        if len(self.history) > 0:
            self.total_history.append(self.history)
        self.history = []
        if i % 5000 == 0:
            self.record(i)
        self.save_networks(val_or_pol)

    def record(self, i):
        if len(self.total_history) != 0:
            file_path = os.path.join("connect_4_data_alphazero", "data_games_" + str(i) + ".pkl")
            with open(file_path, 'wb') as file:
                pickle.dump(self.total_history, file)
            self.total_history = []

    def save_networks(self, val_or_pol):
        if val_or_pol == 0:
            torch.save(self.dqn.state_dict(), "policy_network.pth")
        else:
            torch.save(self.dqn.state_dict(), "value_network.pth")

class DQN(nn.Module):
    def __init__(self, output_channels = 7):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(2, 2), stride=1, padding=0)
        self.fc1 = nn.Linear(128 * 5 * 6, 64)  # Adjusted for the output size of the convolution
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, output_channels) 

    def forward(self, x):
        x = x.type(torch.float32)
        x = x.unsqueeze(0)
        x = x.to('cuda')
        x = F.relu(self.conv1(x))
        x = x.view(-1, 128 * 5 * 6)  # Flatten the output from conv layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x
    

class ReplayMemory():

    def __init__(self, capacity: int, device: str):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device

    def push(
            self,
            state: torch.Tensor, 
            policy: torch.Tensor, 
            value: int, 
    ):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = {
            "current_state": state.to(self.device),
            "policy": policy.to(self.device),
            "value": value
        }
        self.position = (self.position + 1) % self.capacity

    def sample(
            self, 
            batch_size: int,
    ):
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(
            self
    ):
        return len(self.memory)
    
    