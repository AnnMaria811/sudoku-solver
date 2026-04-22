from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TemplateMatch:
    digit: int
    score: float


def is_blank_cell(cell: np.ndarray, filled_ratio_threshold: float = 0.02) -> bool:
    if cell.size == 0:
        return True
    filled_ratio = float(np.count_nonzero(cell)) / float(cell.size)
    return filled_ratio <= filled_ratio_threshold


class TemplateMatcher:
    def __init__(self, templates: dict[str, np.ndarray], blank_threshold: float = 0.02) -> None:
        if not templates:
            raise ValueError("templates cannot be empty")
        # normalise all templates to binary (0 or 1) on creation
        self.templates = {label: self._normalize(template) for label, template in templates.items()}
        self.blank_threshold = blank_threshold

    def match(self, cell: np.ndarray) -> TemplateMatch | None:
        rankings = self.rank(cell)
        if not rankings:
            return None
        return rankings[0]

    def rank(self, cell: np.ndarray, limit: int | None = None) -> list[TemplateMatch]:
        normalized = self._normalize(cell)

        # if the cell is mostly empty, skip matching entirely
        if is_blank_cell(normalized, self.blank_threshold):
            return []

        matches = []
        for label, template in self.templates.items():
            if template.shape != normalized.shape:
                template = self._resize_nearest(template, normalized.shape)
            score = _dice_score(template, normalized)
            matches.append(TemplateMatch(digit=int(label), score=round(score, 6)))

        # sort by score descending so the best match is first
        matches.sort(key=lambda m: m.score, reverse=True)

        if limit is not None:
            return matches[:limit]
        return matches

    @staticmethod
    def _normalize(cell: np.ndarray) -> np.ndarray:
        if cell.ndim != 2:
            raise ValueError("expected a 2D grayscale/binary cell")
        return (cell > 0).astype(np.uint8)

    @staticmethod
    def _resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        src_h, src_w = image.shape
        dst_h, dst_w = shape
        y_idx = np.clip((np.arange(dst_h) * src_h / dst_h).astype(int), 0, src_h - 1)
        x_idx = np.clip((np.arange(dst_w) * src_w / dst_w).astype(int), 0, src_w - 1)
        return image[np.ix_(y_idx, x_idx)]


def load_templates(templates_dir: Path | None, size: int) -> dict[str, np.ndarray]:
    # try loading custom templates from disk first
    if templates_dir is not None and templates_dir.exists():
        templates = _load_templates_from_directory(templates_dir, size)
        if templates:
            return templates
    # fall back to synthetically generated templates
    return generate_default_templates(size=size)


def generate_default_templates(size: int) -> dict[str, np.ndarray]:
    """Draw each digit 1-9 using OpenCV's font and use those as templates."""
    templates = {}
    canvas_size = size * 2

    for digit in range(1, 10):
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        text = str(digit)
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)
        x = (canvas_size - text_size[0]) // 2
        y = (canvas_size + text_size[1]) // 2
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 5, lineType=cv2.LINE_AA)
        _, canvas = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY)
        templates[str(digit)] = center_digit_component(canvas, size)

    return templates


def center_digit_component(component: np.ndarray, canvas_size: int) -> np.ndarray:
    """Crop out the digit, scale it to fit, and centre it on a fixed-size canvas."""
    if component.size == 0 or np.count_nonzero(component) == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # find the bounding box of the digit pixels
    y_indices, x_indices = np.where(component > 0)
    top = int(y_indices.min())
    bottom = int(y_indices.max())
    left = int(x_indices.min())
    right = int(x_indices.max())
    glyph = component[top : bottom + 1, left : right + 1]

    # scale so the longest edge fills most of the canvas
    longest_edge = max(glyph.shape)
    scale = (canvas_size - 8) / max(longest_edge, 1)
    new_w = max(1, int(round(glyph.shape[1] * scale)))
    new_h = max(1, int(round(glyph.shape[0] * scale)))

    resized = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # place the resized glyph in the centre of the canvas
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y_offset = (canvas_size - new_h) // 2
    x_offset = (canvas_size - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return (canvas > 0).astype(np.uint8)


def _load_templates_from_directory(templates_dir: Path, size: int) -> dict[str, np.ndarray]:
    templates = {}
    for digit in range(1, 10):
        for suffix in ("png", "jpg", "jpeg"):
            candidate = templates_dir / f"{digit}.{suffix}"
            if not candidate.exists():
                continue
            image = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            templates[str(digit)] = center_digit_component(binary, size)
            break
    return templates


def _dice_score(template: np.ndarray, candidate: np.ndarray) -> float:
    """
    Dice similarity coefficient — measures how much two binary images overlap.
    Score of 1.0 means perfect match, 0.0 means no overlap at all.
    Formula: 2 * |intersection| / (|template| + |candidate|)
    """
    template = (template > 0).astype(np.uint8)
    candidate = (candidate > 0).astype(np.uint8)

    intersection = int(np.count_nonzero(template & candidate))
    total = int(np.count_nonzero(template)) + int(np.count_nonzero(candidate))

    if total == 0:
        return 0.0
    return (2.0 * intersection) / total


def resolve_candidate_board(
    candidate_grid: list[list[list[TemplateMatch]]],
    *,
    validate_board: callable | None = None,
) -> list[list[int]]:
    board = [[0 for _ in range(9)] for _ in range(9)]

    # collect only the cells that have at least one candidate digit
    constrained_cells = [
        (row, col, candidates)
        for row in range(9)
        for col in range(9)
        if (candidates := candidate_grid[row][col])
    ]

    # sort so cells with fewer options are placed first — reduces backtracking
    constrained_cells.sort(key=lambda item: (len(item[2]), -item[2][0].score))

    def backtrack(index: int) -> bool:
        if index == len(constrained_cells):
            # all cells placed — check if the full board is solvable
            return validate_board(board) if validate_board else True

        row, col, candidates = constrained_cells[index]
        for candidate in candidates:
            if not _can_place(board, row, col, candidate.digit):
                continue
            board[row][col] = candidate.digit
            if backtrack(index + 1):
                return True
            board[row][col] = 0

        return False

    if not backtrack(0):
        raise ValueError("unable to resolve a consistent candidate board")

    return [row[:] for row in board]


def _can_place(board: list[list[int]], row: int, col: int, digit: int) -> bool:
    # check the row
    if digit in board[row]:
        return False

    # check the column
    if any(board[r][col] == digit for r in range(9)):
        return False

    # check the 3x3 box
    box_row = row - row % 3
    box_col = col - col % 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == digit:
                return False

    return True
