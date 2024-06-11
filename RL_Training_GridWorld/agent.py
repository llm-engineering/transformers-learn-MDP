import numpy as np
import random
import pickle

from typing import Tuple 

def is_legal(state : Tuple[int], x_max : int, y_max : int):

    """
    Checks if the current state is legal

    Parameters
    ----------
    state : Tuple[int]
        the current location of the agent (x_position, y_position)
    x_max : int
        the maximum x position
    y_max : int
        the maximum y position
    """

    return (state[0] >= 0) and (state[1] >= 0) and (state[0] < x_max) and (state[1] < y_max)

def load_data(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return []

def make_move(state : Tuple[int], action : str, x_max : int, y_max : int):

    """
    Returns the new state given current state and action

    Parameters
    ----------
    state : Tuple[int]
        the current location of the agent (x_position, y_position)
    action : str
        the action to take : up, down, left, right
    x_max : int
        the maximum x position
    y_max : int
        the maximum y position
    """

    new_state = None

    if action == "up":
        new_state = (state[0], state[1] + 1)
    elif action == "down":
        new_state = (state[0], state[1] - 1)
    elif action == "left":
        new_state = (state[0] - 1, state[1])
    elif action == "right":
        new_state = (state[0] + 1, state[1])
        
    if is_legal(new_state, x_max, y_max):
        return new_state
    
    return state

class Agent:

    def __init__(self, start : Tuple[int], goal : Tuple[int], x_max : int, y_max : int, reward : int):

        """
        Initializes Q-learning Agent

        Parameters
        ----------
        start : Tuple[int]
            starting position of the agent
        goal : Tuple[int]
            goal the agent needs to reach
        x_max : int
            the maximum x position
        y_max : int
            the maximum y position
        reward : int
            reward associated upon reaching the goal
        """

        self.states = []
        self.start = start
        self.goal = goal
        self.x_max = x_max
        self.y_max = y_max

        self.actions = ["up", "down", "left", "right"]
        self.state = start
        self.lr = 0.2
        self.epsilon = 0.1
        self.decay_gamma = 0.9

        self.Q_values = {(i, j): {a: 0 for a in self.actions} for i in range(x_max) for j in range(y_max)}


        for a in self.actions:
            self.Q_values[goal][a] = reward
    
    def chooseAction(self):

        # Epsilon Greedy Policy
        # Makes Random Move with Probability Epsilon

        if np.random.uniform(0, 1) <= self.epsilon:

            action = np.random.choice(self.actions)
            return action
        
        # Choose move w/max Q-Value
        # If tie then choose random

        else:

            mx_nxt_reward = float('-inf')
            potential_actions = []

            for a in self.actions:

                current_position = self.state
                nxt_reward = self.Q_values[current_position][a]

                if nxt_reward > mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
                    potential_actions = [a]

                elif nxt_reward == mx_nxt_reward:
                    potential_actions.append(a)

            if len(potential_actions) == 1:
                return potential_actions[0]
            
            elif len(potential_actions) > 1:
                return potential_actions[random.randrange(len(potential_actions))]

    def train(self, num_episodes : int):

        """
        trains a RL agent a chosen number of episodes

        Parameters
        ----------
        num_episodes : int
            number of episodes to train the RL agent
        """

        i = 0
        episodes = []

        while i < num_episodes:

            action = self.chooseAction()
            next_state = make_move(self.state, action, self.x_max, self.y_max)
            next_action = self.actions[np.argmax([self.Q_values[next_state][a] for a in self.actions])]

            self.Q_values[self.state][action] = self.Q_values[self.state][action] + self.lr * (-1 + self.decay_gamma * self.Q_values[next_state][next_action] - self.Q_values[self.state][action])
            self.states.append([(self.state, {'up': self.Q_values[self.state]['up'], 'down': self.Q_values[self.state]['down'], 'left': self.Q_values[self.state]['left'], 'right': self.Q_values[self.state]['right']}), action])

            self.state = next_state

            if self.state == self.goal:

                episodes.append(self.states)
                self.states = []
                self.state = self.start
                i += 1

        return episodes