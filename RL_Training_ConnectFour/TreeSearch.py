from collections import defaultdict
import math

class MCTS:

    def __init__(self, alpha=1):

        # total reward for each node
        self.Q = defaultdict(int)  

        # total visits for each node
        self.N = defaultdict(int)  

        # children of each node
        self.children = dict()  

        # Trade-Off Between Exploration/Exploitation
        self.alpha = alpha

    def choose(self, node):

        if node.is_terminal():
            raise RuntimeError(f"Terminal Node: {node}")

        if node not in self.children:
            return node.find_random_child()

        monte_carlo_candidates = []
        for child in self.children[node]:
            if self.N[child] == 0:
                child_score = float("-inf")
            else:
                child_score = self.Q[child] / self.N[child]
            move_representation = getattr(child, 'last_move', 'Unknown move')
            monte_carlo_candidates.append([move_representation, child_score])

        best_child = max(self.children[node], key=lambda n: float("-inf") if self.N[n] == 0 else self.Q[n] / self.N[n])
        
        return (best_child, monte_carlo_candidates)

    def rollout(self, node):
        
        # Rollout For 1 Iteration

        path = self.select(node)
        leaf = path[-1]
        self.expand(leaf)
        reward = self.simulate(leaf)
        self.backpropagate(path, reward)

    def select(self, node):

        # Find An Unexplored Descendant

        path = []
        while True:
            path.append(node)

            # Returns Unexplored or Terminal Node
            if node not in self.children or not self.children[node]:
                return path
            
            # Finds Unexplored Nodes, if any
            unexplored = self.children[node] - self.children.keys()

            if unexplored:
                chosen = unexplored.pop()
                path.append(chosen)
                return path
            
            node = self.UCT(node)  

    def expand(self, node):

        # Already Contained in Dictionary
        if node in self.children:
            return  
        
        # Update Dictionary
        self.children[node] = node.find_children()

    def simulate(self, node):
        invert_reward = True

        while not node.is_terminal():
            node = node.find_random_child()[0]
            invert_reward = not invert_reward

        reward = node.reward()
        return 1 - reward if invert_reward else reward

    def backpropagate(self, path, reward):
        
        # Back Propagating Values

        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward   

    def UCT(self, node):
        
        # UCT Selection
        # All children Should Be Expanded
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            return self.Q[n] / self.N[n] + self.alpha * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)

