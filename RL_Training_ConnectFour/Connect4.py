class ConnectFourBoard:
    def __init__(self):
        self.board = [[0 for _ in range(7)] for _ in range(6)]  # 7x6 board
        self.turn = 1  # True for Player 1's turn, False for Player 2's turn
        self.winner = None
        self.terminal = False
        self.last_move = None
        self.move_history = []

    # def find_children(self):
    #     if self.terminal:
    #         return set()
    #     children = set()
    #     for col in range(7):
    #         if self.board[0][col] == 0:  # Check if the top cell of the column is empty
    #             children.add(self.make_move(col))
    #     return children

    # def find_random_child(self):
    #     from random import choice
    #     if self.terminal:
    #         return None
    #     empty_columns = [col for col in range(7) if self.board[0][col] == 0]
    #     col = choice(empty_columns)
    #     return (self.make_move(col), col)

    def reward(self):
        if not self.terminal:
            raise RuntimeError("reward called on nonterminal board")
        if self.winner == self.turn:
            return 1
        elif self.winner is None:
            return 0.5
        else:
            return 0
        
    def find_reward(self, player):
        if self.winner is None:
            return 0
        elif self.winner == player:
            return 1
        else:
            return -1

    def is_terminal(self):
        return self.terminal
 
    def make_move(self, col):
        for row in range(5, -1, -1):  # Start from the bottom of the column
            if self.board[row][col] == 0:
                new_board = [row[:] for row in self.board]
                new_board[row][col] = self.turn
                new_turn = -1 * self.turn
                new_winner = self.find_winner(new_board, row, col)
                new_terminal = (new_winner is not None) or all(new_board[0][col] != 0 for col in range(7))
                new_move_history = self.move_history + [(row, col)]

                new_game = ConnectFourBoard()
                new_game.board = new_board
                new_game.turn = new_turn
                new_game.winner = new_winner
                new_game.terminal = new_terminal
                #print(new_terminal)
                new_game.last_move = (row, col)
                new_game.move_history = new_move_history
                #print(new_game.move_history)
                return new_game
        # for row in self.board:
        #     print(row)
        # print(col)
        raise ValueError("Column is full")

    def find_winner(self, board, last_row, last_col):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        player = board[last_row][last_col]

        for d_row, d_col in directions:
            count = 1
            row, col = last_row + d_row, last_col + d_col
            while 0 <= row < 6 and 0 <= col < 7 and board[row][col] == player:
                count += 1
                row += d_row
                col += d_col

            row, col = last_row - d_row, last_col - d_col
            while 0 <= row < 6 and 0 <= col < 7 and board[row][col] == player:
                count += 1
                row -= d_row
                col -= d_col

            if count >= 4:
                return player

        return None

    # def to_pretty_string(self):
    #     symbols = {1: 'X', 2: 'O', 0: '.'}
    #     rows = [" ".join(symbols[cell] for cell in row) for row in self.board]
    #     return "\n" + "\n".join(rows) + "\n"
