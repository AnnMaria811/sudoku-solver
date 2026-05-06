import numpy as np

from sudoku_solver.ocr import dice_score, generate_digit_templates, is_empty_cell, match_digit


def test_match_digit_identifies_correct_digit():
    templates = generate_digit_templates(32)
    result = match_digit(templates[5].copy(), templates)
    assert result == 5


def test_match_digit_returns_zero_for_blank_cell():
    templates = generate_digit_templates(32)
    assert match_digit(np.zeros((32, 32), dtype=np.uint8), templates) == 0


def test_is_empty_cell_sparse():
    blank = np.zeros((32, 32), dtype=np.uint8)
    blank[0, 0] = 1
    assert is_empty_cell(blank)


def test_is_empty_cell_with_content():
    cell = np.zeros((32, 32), dtype=np.uint8)
    cell[8:24, 14:18] = 1
    assert not is_empty_cell(cell)


def test_dice_score_identical():
    img = np.zeros((32, 32), dtype=np.uint8)
    img[8:24, 8:24] = 1
    assert dice_score(img, img) == 1.0


def test_dice_score_no_overlap():
    a = np.zeros((32, 32), dtype=np.uint8)
    a[0:8, 0:8] = 1
    b = np.zeros((32, 32), dtype=np.uint8)
    b[24:32, 24:32] = 1
    assert dice_score(a, b) == 0.0
