import numpy as np
import random
import pickle
import copy 
import os
from tqdm import tqdm
from collections import defaultdict
from TreeSearch import MCTS
from Connect4 import ConnectFourBoard

num_rows = 6
num_cols = 7
# board = np.zeros((num_rows, num_cols))

# def is_legal_column(col):
#     empty_col = False
#     for i in range(num_rows - 1, -1, -1):
#         if col[i] == 0: #column should be empty from this point forward
#             empty_col = True
#         elif col[i] != 0 and empty_col:
#             return False
#     return True

# def is_legal(state):
#     for col in num_cols:
#         if not is_legal_column(state[:, col]):
#             return False
#     return True 

def get_possible_moves(board):
    return [index for index, value in enumerate(board[0]) if value == 0]

# def load_data(filename):
#     try:
#         with open(filename, 'rb') as file:
#             return pickle.load(file)
#     except FileNotFoundError:
#         return []

# def make_move(state, action, player):
#     state[:, action] = make_move_column(state[:, action], player)
#     return state
    
# def make_move_column(col, player):
#     for i in range(num_rows - 1, -1, -1):
#         if col[i] == 0:
#             col[i] = player
#             break
#     return col

# def acquire_reward(state, player):
#     if four_in_row(state, player):
#         return 1
#     elif len(get_possible_moves(state)) == 0:
#         return 0.5
#     elif four_in_row(state, (player % 2) + 1):
#         return -1
#     else:
#         return 0

# def is_terminal_state(state)
#     return acquire_reward(state, 1) != 0
    
# def four_in_row(arr, player):
#     rows, cols = arr.shape
    
#     for row in range(rows):
#         for col in range(cols - 3):
#             if arr[row, col] == arr[row, col+1] == arr[row, col+2] == arr[row, col+3] == player:
#                 return True

#     for col in range(cols):
#         for row in range(rows - 3):
#             if arr[row, col] == arr[row+1, col] == arr[row+2, col] == arr[row+3, col] == player:
#                 return True

#     for row in range(rows - 3):
#         for col in range(cols - 3):
#             if arr[row, col] == arr[row+1, col+1] == arr[row+2, col+2] == arr[row+3, col+3] == player:
#                 return True

#     for row in range(3, rows):
#         for col in range(cols - 3):
#             if arr[row, col] == arr[row-1, col+1] == arr[row-2, col+2] == arr[row-3, col+3] == player:
#                 return True

#     return False
def default_dict():
    return {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

class Agent:
    def __init__(self, player, board, Q_val = None, MCTS_factor = 1, MCTS_decay = 0.99998, random = False):
        self.states = [] 
        self.state = board
        self.lr = 0.2
        self.exp_rate = 0.05
        self.decay_gamma = 0.9
        self.player = player
        self.tree = MCTS()
        self.MCTS_factor = MCTS_factor
        self.MCTS_decay = MCTS_decay
        self.random = random

        self.Q_values = defaultdict(default_dict)
        if Q_val is not None:
            self.Q_values = Q_val
    
    def chooseAction(self):
        if self.random:
            return np.random.choice(get_possible_moves(self.state.board))
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(get_possible_moves(self.state.board))
            return action
        else:
            for _ in range(100):
                self.tree.rollout(self.state) #perform MCTS rollout

            if self.state not in self.tree.children: #we're at a spot we haven't seen yet -- Q learning and MCTS at this point should do the same thing
                action = self.state.find_random_child()[1]
                return action          
            else: 
                mx_nxt_reward = float('-inf')
                potential_actions = []
                for child in self.tree.children[self.state]:
                    if self.tree.N[child] == 0:
                        child_score = float("-inf")
                    else:
                        child_score = self.tree.Q[child] / self.tree.N[child]
                    action = getattr(child, 'last_move')[1]
                    board_str = str(self.state.board)
                    nxt_reward = (1 - self.MCTS_factor) * self.Q_values[board_str][action] + self.MCTS_factor * child_score
                    if nxt_reward > mx_nxt_reward:
                        mx_nxt_reward = nxt_reward
                        potential_actions = [action]
                    elif nxt_reward == mx_nxt_reward:
                        potential_actions.append(action)
                if len(potential_actions) == 1:
                    return potential_actions[0]
                elif len(potential_actions) > 1:
                    return potential_actions[random.randrange(len(potential_actions))]
        
    def newGame(self, board, iteration):
        file_path = os.path.join("connect_4_data", "data_agent_" + str(self.player) + "_game_" + str(iteration) + ".pkl")
        if iteration is not None:
            with open(file_path, 'wb') as file:
                pickle.dump(self.states, file)
            if iteration % 10000 == 0:
                file_path = os.path.join("connect_4_q_vals", "data_agent_q_values_" + str(self.player) + "_game_" + str(iteration) + ".pkl")
                with open(file_path, 'wb') as file:
                    pickle.dump(self.Q_values, file)
        self.states = []
        self.state = board
        self.MCTS_factor *= self.MCTS_decay
        self.tree = MCTS()

    def update_values(self, move, reward, offset, board):
        board_str = str(self.state.board)
        self.Q_values[board_str][move] = self.Q_values[board_str][move] + self.lr * (self.decay_gamma * reward - self.Q_values[board_str][move])
        # print("hi")
        # print(self.player)
        # print(self.state.move_history)
        self.states.append([(board.move_history[:offset], copy.deepcopy(self.Q_values[board_str])), board.move_history[offset]])

# Q_vals_1 = None
# with open("connect_4_q_vals/data_agent_q_values_1_game_150000.pkl", 'rb') as file:
#     Q_vals_1 = pickle.load(file)
# Q_vals_2 = None
# with open("connect_4_q_vals/data_agent_q_values_2_game_150000.pkl", 'rb') as file:
#     Q_vals_2 = pickle.load(file)

# board = ConnectFourBoard()
# agent_1 = Agent(1, board, Q_vals_1, 0, 0, False)
# agent_2 = Agent(2, board, Q_vals_2, 0, 0, False)
# for i in tqdm(range(150001, 200001)):
#     move_1 = None
#     move_2 = None
#     while True:
#         move_1 = agent_1.chooseAction()
#         board = board.make_move(move_1)
#         if board.terminal: #hit a terminal state, update Q values for both agents
#             reward_1 = board.find_reward(1)
#             reward_2 = board.find_reward(2)
#             agent_1.update_values(move_1, reward_1, -1, board)
#             agent_2.update_values(move_2, reward_2, -2, board)
#             break
#         elif move_2 is not None: #we calculate states temporal difference as after both agents make a move, so we update for agent_2 since agent_2 just regained control
#             board_str = str(board.board)
#             next_action_2 = np.argmax([agent_2.Q_values[board_str][a] for a in get_possible_moves(board.board)])
#             reward_2 = agent_2.Q_values[board_str][next_action_2]
#             agent_2.update_values(move_2, reward_2, -2, board)
#             agent_2.state = board
#         move_2 = agent_2.chooseAction()
#         board = board.make_move(move_2)
#         if board.terminal: #hit a terminal state, update Q values for both agents
#             reward_1 = board.find_reward(1)
#             reward_2 = board.find_reward(2)
#             agent_1.update_values(move_1, reward_1, -2, board)
#             agent_2.update_values(move_2, reward_2, -1, board)
#             break
#         else:
#             board_str = str(board.board)
#             next_action_1 = np.argmax([agent_1.Q_values[board_str][a] for a in get_possible_moves(board.board)])
#             reward_1 = agent_1.Q_values[board_str][next_action_1]
#             agent_1.update_values(move_1, reward_1, -2, board)
#             agent_1.state = board
#     board = ConnectFourBoard()
#     agent_1.newGame(board, i)
#     agent_2.newGame(board, i)

# with open("connect_4_q_vals/data_agent_q_values_1_game_0.pkl", 'rb') as file:
#     test = pickle.load(file)
#     print(test)

# with open("connect_4_data/data_agent_1_game_0.pkl", 'rb') as file:
#     test = pickle.load(file)
#     print(test)



'''testing''' 

Q_vals = None
with open("connect_4_q_vals/data_agent_q_values_1_game_200000.pkl", 'rb') as file:
    Q_vals = pickle.load(file)

board = ConnectFourBoard()
agent_1 = Agent(1, board, Q_vals, 0)
agent_2 = Agent(2, board, None, 1, 1, True)
move_1 = None
move_2 = None
winners = []
#print(Q_vals)
for i in tqdm(range(1000)):
    while True:
        move_1 = agent_1.chooseAction()
        board = board.make_move(move_1)
        # for row in board.board:
        #     print(row)
        # print("_____________________________")
        # print(str(board.board))
        if board.terminal: #hit a terminal state, update Q values for both agents
            reward_1 = board.find_reward(1)
            reward_2 = board.find_reward(2)
            # agent_1.update_values(move_1, reward_1)
            # agent_2.update_values(move_2, reward_2)
            winners.append(board.winner)
            break
        elif move_2 is not None: #we calculate states temporal difference as after both agents make a move, so we update for agent_2 since agent_2 just regained control
            board_str = str(board.board)
            next_action_2 = np.argmax([agent_2.Q_values[board_str][a] for a in get_possible_moves(board.board)])
            reward_2 = agent_2.Q_values[board_str][next_action_2]
            # agent_2.update_values(move_2, reward_2)
            agent_2.state = board
        move_2 = agent_2.chooseAction()
        board = board.make_move(move_2)
        # for row in board.board:
        #     print(row)
        # print("_____________________________")
        if board.terminal: #hit a terminal state, update Q values for both agents
            reward_1 = board.find_reward(1)
            reward_2 = board.find_reward(2)
            # agent_1.update_values(move_1, reward_1)
            # agent_2.update_values(move_2, reward_2)
            winners.append(board.winner)
            break
        else:
            board_str = str(board.board)
            next_action_1 = np.argmax([agent_1.Q_values[board_str][a] for a in get_possible_moves(board.board)])
            reward_1 = agent_1.Q_values[board_str][next_action_1]
            # agent_1.update_values(move_1, reward_1)
            agent_1.state = board
    board = ConnectFourBoard()
    agent_1.newGame(board, None)
    agent_2.newGame(board, None)

print(winners.count(1))
print(winners.count(2))
print(winners.count(None))

# with open("connect_4_data/data_agent_1_game_50000.pkl", 'rb') as file:
#     test = pickle.load(file)
#     print(test)

# with open("connect_4_data/data_agent_2_game_50000.pkl", 'rb') as file:
#     test = pickle.load(file)
#     print(test)