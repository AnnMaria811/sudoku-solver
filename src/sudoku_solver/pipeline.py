import json
from pathlib import Path

import cv2
import numpy as np

from .cell_extraction import extract_cells
from .config import PipelineConfig
from .grid_detection import detect_grid
from .ocr import TemplateMatcher, load_templates, resolve_candidate_board
from .preprocess import preprocess_image
from .solver import InvalidBoardError, solve_board
from .warp import warp_grid


STAGE_ORDER = ("preprocess", "detect", "warp", "ocr", "solve")


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot complete successfully."""


def run_pipeline(
    image_path: Path,
    output_dir: Path,
    stop_after_stage: str | None = None,
    save_stages: bool = True,
    templates_dir: Path | None = None,
) -> None:
    config = PipelineConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"unable to read image: {image_path}")

    summary = {
        "image_path": str(image_path),
        "output_dir": str(output_dir),
        "save_stages": save_stages,
        "stop_after_stage": stop_after_stage or "solve",
    }

    if save_stages:
        _write_image(output_dir / "01_original.png", image)

    # --- Stage 1: Preprocess ---
    print("Step 1: Preprocessing the image...")
    preprocessed = preprocess_image(image, config)
    if save_stages:
        _write_image(output_dir / "02_preprocessed.png", preprocessed.normalized)

    if _should_stop(stop_after_stage, "preprocess"):
        _write_summary(output_dir / "run_summary.json", summary | {"completed_stage": "preprocess"})
        return

    # --- Stage 2: Detect the grid ---
    print("Step 2: Detecting the Sudoku grid...")
    detection = detect_grid(preprocessed.binary, image, config)
    if save_stages:
        _write_image(output_dir / "03_grid_mask.png", detection.mask)
        _write_image(output_dir / "04_lines_or_contour.png", detection.contour_overlay)
        _write_image(output_dir / "05_corners_overlay.png", detection.corners_overlay)

    if _should_stop(stop_after_stage, "detect"):
        _write_summary(output_dir / "run_summary.json", summary | {"completed_stage": "detect"})
        return

    # --- Stage 3: Warp the grid to a flat square ---
    print("Step 3: Straightening the grid...")
    warped = warp_grid(image, detection.corners, config)
    if save_stages:
        _write_image(output_dir / "06_warped_grid.png", warped.color)

    if _should_stop(stop_after_stage, "warp"):
        _write_summary(
            output_dir / "run_summary.json",
            summary | {"completed_stage": "warp", "contour_area": detection.contour_area},
        )
        return

    # --- Stage 4: OCR — read the digits from each cell ---
    print("Step 4: Reading digits from cells...")
    cells = extract_cells(warped.binary, warped.color, config)
    if save_stages:
        _write_image(output_dir / "07_cells_overlay.png", cells.overlay)

    digit_crops_dir = output_dir / "08_digit_crops"
    digit_crops_dir.mkdir(exist_ok=True)
    templates = load_templates(templates_dir, config.digit_canvas_size)
    matcher = TemplateMatcher(templates, blank_threshold=config.blank_filled_ratio_threshold)
    recognized_board, confidences = _recognize_board(cells.digit_tiles, matcher, digit_crops_dir if save_stages else None, config)
    _write_grid(output_dir / "09_recognized_grid.txt", recognized_board)

    if _should_stop(stop_after_stage, "ocr"):
        _write_summary(
            output_dir / "run_summary.json",
            summary | {
                "completed_stage": "ocr",
                "recognized_board": recognized_board,
                "mean_confidence": _mean(confidences),
            },
        )
        return

    # --- Stage 5: Solve the puzzle ---
    print("Step 5: Solving the puzzle...")
    try:
        solved_board = solve_board(recognized_board)
    except InvalidBoardError as exc:
        _write_summary(
            output_dir / "run_summary.json",
            summary | {
                "completed_stage": "ocr",
                "recognized_board": recognized_board,
                "mean_confidence": _mean(confidences),
                "contour_area": detection.contour_area,
                "error": str(exc),
            },
        )
        raise PipelineError(f"OCR produced an invalid board: {exc}") from exc

    _write_grid(output_dir / "10_solved_grid.txt", solved_board)
    _assert_givens_preserved(recognized_board, solved_board)

    solution_overlay = _render_solution_overlay(warped.color, recognized_board, solved_board)
    if save_stages:
        _write_image(output_dir / "11_solution_overlay.png", solution_overlay)

    _write_summary(
        output_dir / "run_summary.json",
        summary | {
            "completed_stage": "solve",
            "recognized_board": recognized_board,
            "solved_board": solved_board,
            "mean_confidence": _mean(confidences),
            "contour_area": detection.contour_area,
        },
    )

    print("Done!")


def _recognize_board(
    digit_tiles: list[np.ndarray],
    matcher: TemplateMatcher,
    digit_crops_dir: Path | None,
    config: PipelineConfig | None = None,
) -> tuple[list[list[int]], list[float]]:
    if config is None:
        config = PipelineConfig()

    min_score = config.ocr_min_score
    min_accept = config.ocr_min_accept_score
    margin = config.ocr_score_margin

    candidate_grid = [[[] for _ in range(9)] for _ in range(9)]
    confidence_map: dict[tuple[int, int, int], float] = {}

    for row in range(9):
        for col in range(9):
            index = row * 9 + col
            tile = digit_tiles[index]

            if digit_crops_dir is not None:
                _write_image(digit_crops_dir / f"cell_{row + 1}_{col + 1}.png", tile * 255)

            rankings = matcher.rank(tile, limit=4)

            # skip this cell if there's no confident match
            if not rankings or rankings[0].score < min_score:
                continue

            # keep any candidate that's close enough to the best score
            accepted = [
                match
                for match in rankings
                if match.score >= max(min_accept, rankings[0].score - margin)
            ]
            if not accepted:
                continue

            candidate_grid[row][col] = accepted
            for match in accepted:
                confidence_map[(row, col, match.digit)] = match.score

    try:
        resolved_board = resolve_candidate_board(candidate_grid, validate_board=_is_solvable_board)
    except ValueError as exc:
        raise PipelineError(f"OCR produced an inconsistent board: {exc}") from exc

    confidences = [
        confidence_map.get((row, col, resolved_board[row][col]), 0.0)
        for row in range(9)
        for col in range(9)
    ]
    return resolved_board, confidences


def _render_solution_overlay(
    warped_color: np.ndarray,
    recognized_board: list[list[int]],
    solved_board: list[list[int]],
) -> np.ndarray:
    overlay = warped_color.copy()
    cell_size = warped_color.shape[0] // 9

    for row in range(9):
        for col in range(9):
            x = col * cell_size + int(cell_size * 0.28)
            y = (row + 1) * cell_size - int(cell_size * 0.22)

            if recognized_board[row][col] != 0:
                # draw OCR-read givens in blue so we can visually audit them
                cv2.putText(
                    overlay,
                    str(recognized_board[row][col]),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (200, 100, 0),
                    3,
                    lineType=cv2.LINE_AA,
                )
            else:
                # draw solver-filled digits in green
                cv2.putText(
                    overlay,
                    str(solved_board[row][col]),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 160, 0),
                    3,
                    lineType=cv2.LINE_AA,
                )

    return overlay


def _assert_givens_preserved(recognized_board: list[list[int]], solved_board: list[list[int]]) -> None:
    for row in range(9):
        for col in range(9):
            given = recognized_board[row][col]
            if given != 0 and solved_board[row][col] != given:
                raise PipelineError(
                    f"solver did not preserve given at ({row},{col}): "
                    f"expected {given}, got {solved_board[row][col]}"
                )


def _write_grid(path: Path, board: list[list[int]]) -> None:
    lines = [" ".join(str(value) for value in row) for row in board]
    path.write_text("\n".join(lines) + "\n")


def _write_image(path: Path, image: np.ndarray) -> None:
    success = cv2.imwrite(str(path), image)
    if not success:
        raise OSError(f"unable to write image: {path}")


def _write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2))


def _should_stop(stop_after_stage: str | None, current_stage: str) -> bool:
    return stop_after_stage == current_stage


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _is_solvable_board(board: list[list[int]]) -> bool:
    try:
        solve_board(board)
    except InvalidBoardError:
        return False
    return True
