from typing import Callable, Dict, Any
import torch
from Connect4 import ConnectFourBoard
import numpy as np

def get_possible_moves(board):
    return [index for index, value in enumerate(board[0]) if value == 0]

class Connect4Game:
    def __init__(self):
        self.state = ConnectFourBoard()
        self.state_history = [self.state.board]

    def step(self, action, player):
        self.state = self.state.make_move(action)
        reward = 0
        if self.state.is_terminal():
            reward = self.state.find_reward(player) * 10 + 0.5
        self.state_history.append(self.state.board)
        return reward, self.state.is_terminal() 
    
    def action_space_sample(self):
        possible_moves = get_possible_moves(self.state.board)
        return np.random.choice(possible_moves) 
    
    def find_reward(self, player):
        if not self.state.is_terminal():
            return 0
        return self.state.find_reward(player) * 10 + 0.5
    
    def reset(self):
        self.state = ConnectFourBoard()
        self.state_history = [self.state.board]



        