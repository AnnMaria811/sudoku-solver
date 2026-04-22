from copy import deepcopy


class InvalidBoardError(ValueError):
    """Raised when the recognized Sudoku board is structurally invalid."""


Board = list[list[int]]


def solve_board(board: Board) -> Board:
    validate_board(board)
    working = deepcopy(board)
    if not _solve_in_place(working):
        raise InvalidBoardError("board has no valid solution")
    return working


def validate_board(board: Board) -> None:
    if len(board) != 9 or any(len(row) != 9 for row in board):
        raise InvalidBoardError("board must be 9x9")

    # check each row for duplicate digits
    for row in board:
        _validate_group(row)

    # check each column for duplicate digits
    for col in range(9):
        _validate_group([board[row][col] for row in range(9)])

    # check each 3x3 box for duplicate digits
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            _validate_group(
                [
                    board[row][col]
                    for row in range(box_row, box_row + 3)
                    for col in range(box_col, box_col + 3)
                ]
            )


def _validate_group(values: list[int]) -> None:
    digits = [value for value in values if value != 0]
    if any(value < 0 or value > 9 for value in values):
        raise InvalidBoardError("digits must be between 0 and 9")
    if len(digits) != len(set(digits)):
        raise InvalidBoardError("board contains duplicate digits")


def _solve_in_place(board: Board) -> bool:
    # find the next empty cell (value 0)
    empty = _find_empty(board)
    if empty is None:
        return True  # no empty cells left, the puzzle is solved!

    row, col = empty

    # try placing each digit 1-9 in this cell
    for candidate in range(1, 10):
        if _is_valid_move(board, row, col, candidate):
            board[row][col] = candidate

            # recursively try to solve the rest of the board
            if _solve_in_place(board):
                return True

            # this digit didn't lead to a solution, undo and try the next one
            board[row][col] = 0

    # none of the digits worked, so we need to backtrack
    return False


def _find_empty(board: Board) -> tuple[int, int] | None:
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return row, col
    return None


def _is_valid_move(board: Board, row: int, col: int, candidate: int) -> bool:
    # check the row
    if candidate in board[row]:
        return False

    # check the column
    if any(board[r][col] == candidate for r in range(9)):
        return False

    # check the 3x3 box that this cell belongs to
    box_row = row - row % 3
    box_col = col - col % 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == candidate:
                return False

    return True
