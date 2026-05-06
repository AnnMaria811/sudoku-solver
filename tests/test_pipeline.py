from pathlib import Path

import cv2
import numpy as np

from sudoku_solver.pipeline import run_pipeline


PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def test_pipeline_solves_puzzle(tmp_path):
    image_path = tmp_path / "board.png"
    _make_sudoku_image(image_path, PUZZLE, tilt=8.0)

    run_pipeline(image_path=image_path, output_dir=tmp_path / "out")

    assert _read_grid(tmp_path / "out" / "10_solved_grid.txt") == SOLUTION
    assert (tmp_path / "out" / "11_solution_overlay.png").exists()
    assert (tmp_path / "out" / "run_summary.json").exists()


def test_pipeline_stops_after_warp(tmp_path):
    image_path = tmp_path / "board.png"
    _make_sudoku_image(image_path, PUZZLE, tilt=0.0)

    run_pipeline(image_path=image_path, output_dir=tmp_path / "out", stop_after_stage="warp")

    assert (tmp_path / "out" / "06_warped_grid.png").exists()
    assert not (tmp_path / "out" / "09_recognized_grid.txt").exists()


def _make_sudoku_image(path, board, tilt):
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    top_left = (140, 140)
    board_size = 720
    cell = board_size // 9

    for i in range(10):
        thickness = 5 if i % 3 == 0 else 2
        offset = i * cell
        cv2.line(image, (top_left[0] + offset, top_left[1]), (top_left[0] + offset, top_left[1] + board_size), (0, 0, 0), thickness)
        cv2.line(image, (top_left[0], top_left[1] + offset), (top_left[0] + board_size, top_left[1] + offset), (0, 0, 0), thickness)

    for r, row in enumerate(board):
        for c, val in enumerate(row):
            if val == 0:
                continue
            x = top_left[0] + c * cell + 18
            y = top_left[1] + (r + 1) * cell - 18
            cv2.putText(image, str(val), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 0, 0), 4)

    if tilt:
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, tilt, 1.0)
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    assert cv2.imwrite(str(path), image)


def _read_grid(path):
    return [[int(v) for v in line.split()] for line in path.read_text().splitlines()]
