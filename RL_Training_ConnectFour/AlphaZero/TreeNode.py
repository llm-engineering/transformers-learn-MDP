from abc import ABC, abstractmethod

class Node(ABC):
    
    # Representation of Board State

    @abstractmethod
    def find_children(self):
        # Returns All Successors of Board State
        return set()

    @abstractmethod
    def find_random_child(self):
        # Returns Random Successor of Board State
        return None

    @abstractmethod
    def is_terminal(self):
        # Returns True if Terminal Node
        return True

    @abstractmethod
    def reward(self):
        # 0/1 Loss; 0.5 for tie
        return 0