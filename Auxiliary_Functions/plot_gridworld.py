import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional

def plot_gridworld(path : List[Tuple[int]], filename : Optional[str] = None):

    """
    Plots a GridWorld board visualizing the trajectory of an agent.

    Parameters
    ----------
    path : List[Tuple[int, int]]
        A list of tuples representing the trajectory of the agent from the beginning to the end.
        Each tuple contains (x, y) coordinates on a 9x9 grid.

    filename : str, optional
        If provided, the plot is saved to the given file path. If None, the plot is only displayed.
    """

    dim = 9

    _, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.set_facecolor('white')   

    grid = np.full((dim, dim), 0.9)   

    _ = ax.matshow(grid, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax.xaxis.set_ticks_position('bottom')
    ax.invert_yaxis()

    ax.set_xticks(np.arange(-0.5, dim, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, dim, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)   

    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.plot(path[0][0], path[0][1], 'ro', markersize=10)
    ax.plot(path[-1][0], path[-1][1], 'bo', markersize=10)
    ax.plot(*zip(*[(p[0], p[1]) for p in path]), 'k-', linewidth=2) 

    for (m, n), (m_next, n_next) in zip(path, path[1:]):
        ax.annotate("", xy=(m_next, n_next), xytext=(m, n),
                    arrowprops=dict(arrowstyle="->, head_length=0.6, head_width=0.3", lw=1.5, color='black'))
        
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save the figure
        
    plt.show()