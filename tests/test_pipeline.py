from __future__ import annotations

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


def test_run_pipeline_solves_clean_synthetic_board(tmp_path: Path) -> None:
    image_path = tmp_path / "synthetic.png"
    _write_sudoku_image(image_path, PUZZLE, rotate_degrees=8.0)

    run_pipeline(image_path=image_path, output_dir=tmp_path / "output")

    solved_grid = _read_grid(tmp_path / "output" / "10_solved_grid.txt")
    assert solved_grid == SOLUTION
    assert (tmp_path / "output" / "11_solution_overlay.png").exists()
    assert (tmp_path / "output" / "run_summary.json").exists()


def test_run_pipeline_can_stop_after_warp_stage(tmp_path: Path) -> None:
    image_path = tmp_path / "synthetic.png"
    _write_sudoku_image(image_path, PUZZLE, rotate_degrees=0.0)

    run_pipeline(
        image_path=image_path,
        output_dir=tmp_path / "output",
        stop_after_stage="warp",
    )

    assert (tmp_path / "output" / "06_warped_grid.png").exists()
    assert not (tmp_path / "output" / "09_recognized_grid.txt").exists()


def _write_sudoku_image(path: Path, board: list[list[int]], rotate_degrees: float) -> None:
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    top_left = (140, 140)
    board_size = 720
    cell = board_size // 9

    for index in range(10):
        thickness = 5 if index % 3 == 0 else 2
        offset = index * cell
        cv2.line(
            image,
            (top_left[0] + offset, top_left[1]),
            (top_left[0] + offset, top_left[1] + board_size),
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            image,
            (top_left[0], top_left[1] + offset),
            (top_left[0] + board_size, top_left[1] + offset),
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

    for row_idx, row in enumerate(board):
        for col_idx, value in enumerate(row):
            if value == 0:
                continue
            x = top_left[0] + col_idx * cell + 18
            y = top_left[1] + (row_idx + 1) * cell - 18
            cv2.putText(
                image,
                str(value),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.9,
                (0, 0, 0),
                4,
                lineType=cv2.LINE_AA,
            )

    if rotate_degrees:
        center = (image.shape[1] // 2, image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, rotate_degrees, 1.0)
        image = cv2.warpAffine(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    assert cv2.imwrite(str(path), image)


def test_run_pipeline_raises_on_inconsistent_ocr(tmp_path: Path, monkeypatch) -> None:
    from sudoku_solver import pipeline as _pipeline
    from sudoku_solver.ocr import TemplateMatch

    def _bad_recognize(digit_tiles, matcher, crops_dir, config=None):
        from sudoku_solver.pipeline import PipelineError
        raise PipelineError("OCR produced an inconsistent board: unable to resolve a consistent candidate board")

    monkeypatch.setattr(_pipeline, "_recognize_board", _bad_recognize)

    image_path = tmp_path / "synthetic.png"
    _write_sudoku_image(image_path, PUZZLE, rotate_degrees=0.0)

    import pytest
    with pytest.raises(_pipeline.PipelineError, match="OCR produced an inconsistent board"):
        run_pipeline(image_path=image_path, output_dir=tmp_path / "output")


def _read_grid(path: Path) -> list[list[int]]:
    rows = []
    for line in path.read_text().splitlines():
        rows.append([int(value) for value in line.split()])
    return rows
