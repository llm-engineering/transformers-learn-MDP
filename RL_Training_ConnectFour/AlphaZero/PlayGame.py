import random 
from TreeSearch import MCTS
from Connect4 import ConnectFourBoard

def play_game(seed = None):

    if seed is not None:
        random.seed(seed)

    plays = []
    tree = MCTS()   
    board = ConnectFourBoard()   
    
    while not board.terminal:   
        
        valid_columns = [col for col in range(7) if board.board[0][col] is 0]
        col = random.choice(valid_columns) if valid_columns else None
        
        if col is not None:
            # Generate a random move from 
            # The List of Valid Columns
            # Random Player Always Starts First
            board = board.make_move(col)
            plays.append(board.last_move)
        
        if board.terminal:  # Check if the game has ended
            break
        
        # Perform MCTS rollouts
        for _ in range(200):
            tree.rollout(board)
        
        # Monte Carlo Tree Search Move
        board, candidates = tree.choose(board)
        plays.append((board.last_move, candidates))
        
        if board.terminal:   
            break

    return plays
