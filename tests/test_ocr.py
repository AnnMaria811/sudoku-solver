from __future__ import annotations

import numpy as np

from sudoku_solver.ocr import (
    TemplateMatch,
    TemplateMatcher,
    is_blank_cell,
    resolve_candidate_board,
)


ONE = np.array(
    [
        [0, 1, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 1],
    ],
    dtype=np.uint8,
)

TWO = np.array(
    [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
    ],
    dtype=np.uint8,
)


def test_template_matcher_returns_best_matching_digit() -> None:
    matcher = TemplateMatcher({"1": ONE, "2": TWO}, blank_threshold=0.02)

    result = matcher.match(ONE.copy())

    assert result == TemplateMatch(digit=1, score=1.0)


def test_is_blank_cell_detects_sparse_cells() -> None:
    blank = np.zeros((32, 32), dtype=np.uint8)
    blank[0, 0] = 1

    assert is_blank_cell(blank, filled_ratio_threshold=0.01)


def test_is_blank_cell_detects_cells_with_digit_content() -> None:
    digit = np.zeros((32, 32), dtype=np.uint8)
    digit[8:24, 14:18] = 1

    assert not is_blank_cell(digit, filled_ratio_threshold=0.01)


def test_resolve_candidate_board_prefers_consistent_candidates() -> None:
    candidate_grid = [[[] for _ in range(9)] for _ in range(9)]
    candidate_grid[0][0] = [TemplateMatch(digit=1, score=0.9)]
    candidate_grid[0][1] = [TemplateMatch(digit=1, score=0.95), TemplateMatch(digit=3, score=0.85)]
    candidate_grid[1][0] = [TemplateMatch(digit=2, score=0.8)]

    resolved = resolve_candidate_board(candidate_grid)

    assert resolved[0][0] == 1
    assert resolved[0][1] == 3
    assert resolved[1][0] == 2


def test_resolve_candidate_board_raises_when_no_consistent_assignment_possible() -> None:
    import pytest

    candidate_grid = [[[] for _ in range(9)] for _ in range(9)]
    # force an impossible conflict: same digit locked in every column of row 0
    for col in range(9):
        candidate_grid[0][col] = [TemplateMatch(digit=1, score=0.9)]

    with pytest.raises(ValueError, match="unable to resolve"):
        resolve_candidate_board(candidate_grid)
