import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from typing import List, Optional

def plot_connect4(board: List[List[int]], filename: Optional[str] = None):

    """
    Plots a Connect4 board for visualization.

    Parameters
    ----------
    board : List[List[int]]
        Current state of the board, represented as a list of lists of integers,
        where 0 represents an empty cell, 1 represents a chip for Player 1, 
        and 2 represents a chip for Player 2.

    filename : str, optional
        Path to save the image file. If None, the plot is not saved.
    """

    colors = {0: 'white', 1: 'red', 2: 'yellow'}
    _, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal')

    for y in range(6):
        for x in range(7):
            color = colors[board[y][x]]
            circle = patches.Circle((x, y), radius=0.45, color=color, ec='black')
            ax.add_patch(circle)

    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(5.5, -0.5)

    ax.set_xticks(np.arange(-0.5, 7, 1), minor=False)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=False)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)

    ax.set_xticks(np.arange(0, 7, 1), minor=True)
    ax.set_yticks(np.arange(0, 6, 1), minor=True)

    ax.set_xticklabels([str(i + 1) for i in range(7)], minor=True, fontsize=14)
    ax.set_yticklabels([str(6 - i) for i in range(6)], minor=True, fontsize=14)

    ax.set_xticklabels([], minor=False)
    ax.set_yticklabels([], minor=False)

    ax.tick_params(axis='both', which='both', length=0)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save the figure

    plt.show()
