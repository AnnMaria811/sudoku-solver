from copy import deepcopy


class InvalidBoardError(ValueError):
    pass


def solve_board(board):
    validate_board(board)
    working = deepcopy(board)
    if not _backtrack(working):
        raise InvalidBoardError("This board has no valid solution")
    return working


def validate_board(board):
    if len(board) != 9 or any(len(row) != 9 for row in board):
        raise InvalidBoardError("Board must be 9x9")

    for row in board:
        _check_group(row)

    for col in range(9):
        _check_group([board[row][col] for row in range(9)])

    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = [board[r][c] for r in range(box_row, box_row + 3) for c in range(box_col, box_col + 3)]
            _check_group(box)


def _check_group(values):
    digits = [v for v in values if v != 0]
    if any(v < 0 or v > 9 for v in values):
        raise InvalidBoardError("Digits must be between 0 and 9")
    if len(digits) != len(set(digits)):
        raise InvalidBoardError("Duplicate digits in the same row, column, or box")


def _backtrack(board):
    empty = _find_empty(board)
    if empty is None:
        return True

    row, col = empty
    for digit in range(1, 10):
        if _can_place(board, row, col, digit):
            board[row][col] = digit
            if _backtrack(board):
                return True
            board[row][col] = 0

    return False


def _find_empty(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return row, col
    return None


def _can_place(board, row, col, digit):
    if digit in board[row]:
        return False
    for r in range(9):
        if board[r][col] == digit:
            return False
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == digit:
                return False
    return True
