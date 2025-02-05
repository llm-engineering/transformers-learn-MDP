from typing import List


def actions_to_col_row(actions, board_height=6):
    """
    Converts a sequence of Connect4 column moves into (column, row) pairs.

    Args:
        actions (list): List of column indices (0-6) representing moves.
        board_height (int): Number of rows in Connect4 (default: 6).

    Returns:
        list of tuples: [(col, row), ...] where row is where the piece lands.
    """
    heights = [0] * 7  # Track how filled each column is
    col_row_sequence = []

    for col in actions:
        row = board_height - 1 - heights[col]  # Compute the landing row
        if row < 0:
            raise ValueError(f"Invalid move: Column {col} is full!")

        col_row_sequence.append((row, col))
        heights[col] += 1  # Update column height

    return col_row_sequence


def information_parser(info: List[str]):
    """

    
    """
    # 
    parsed_info = []

    for line in info:
        temp = []
        raw = line.split(",")
        counter = 0
        while counter < len(raw):

            leap_steps = int(raw[counter]) * 2
            counter += 1

            q_values = {}
            fragment = raw[counter:counter + leap_steps ]
            zip_object = zip(fragment[::2], fragment[1::2])
            for key, value in zip_object:
                q_values[int(key)] = float(value)
            counter += leap_steps

            temp.append((q_values, int(raw[counter])))
            counter += 1

        parsed_info.append(temp)
    
    return parsed_info