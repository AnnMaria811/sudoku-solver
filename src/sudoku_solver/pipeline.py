import json
from pathlib import Path

import cv2

from .cell_extraction import extract_cells
from .config import PipelineConfig
from .grid_detection import detect_grid
from .ocr import load_templates, match_digit
from .preprocess import preprocess_image
from .solver import InvalidBoardError, solve_board
from .warp import warp_grid


class PipelineError(RuntimeError):
    pass


def run_pipeline(image_path, output_dir, stop_after_stage=None, save_stages=True, templates_dir=None):
    config = PipelineConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    if save_stages:
        _save_image(output_dir / "01_original.png", image)

    print("Step 1: Preprocessing...")
    gray, normalized, binary = preprocess_image(image, config)
    if save_stages:
        _save_image(output_dir / "02_preprocessed.png", normalized)

    if stop_after_stage == "preprocess":
        _save_summary(output_dir / "run_summary.json", {"completed_stage": "preprocess"})
        return

    print("Step 2: Detecting the grid...")
    mask, contour_overlay, corners_overlay, corners, contour_area = detect_grid(binary, image, config)
    if save_stages:
        _save_image(output_dir / "03_grid_mask.png", mask)
        _save_image(output_dir / "04_lines_or_contour.png", contour_overlay)
        _save_image(output_dir / "05_corners_overlay.png", corners_overlay)

    if stop_after_stage == "detect":
        _save_summary(output_dir / "run_summary.json", {"completed_stage": "detect"})
        return

    print("Step 3: Straightening the grid...")
    warped_color, warped_binary = warp_grid(image, corners, config)
    if save_stages:
        _save_image(output_dir / "06_warped_grid.png", warped_color)

    if stop_after_stage == "warp":
        _save_summary(output_dir / "run_summary.json", {"completed_stage": "warp"})
        return

    print("Step 4: Reading digits...")
    digit_tiles, cells_overlay = extract_cells(warped_binary, warped_color, config)
    if save_stages:
        _save_image(output_dir / "07_cells_overlay.png", cells_overlay)

    digit_crops_dir = output_dir / "08_digit_crops"
    if save_stages:
        digit_crops_dir.mkdir(exist_ok=True)

    templates = load_templates(templates_dir, config.digit_canvas_size)

    recognized_board = [[0] * 9 for _ in range(9)]
    for row in range(9):
        for col in range(9):
            tile = digit_tiles[row * 9 + col]
            if save_stages:
                _save_image(digit_crops_dir / f"cell_{row + 1}_{col + 1}.png", tile * 255)
            recognized_board[row][col] = match_digit(tile, templates, config.ocr_min_score)

    _save_grid(output_dir / "09_recognized_grid.txt", recognized_board)

    if stop_after_stage == "ocr":
        _save_summary(output_dir / "run_summary.json", {"completed_stage": "ocr", "recognized_board": recognized_board})
        return

    print("Step 5: Solving...")
    try:
        solved_board = solve_board(recognized_board)
    except InvalidBoardError as exc:
        _save_summary(output_dir / "run_summary.json", {"completed_stage": "ocr", "error": str(exc)})
        raise PipelineError(f"OCR produced an invalid board: {exc}") from exc

    _save_grid(output_dir / "10_solved_grid.txt", solved_board)

    solution_overlay = _draw_solution(warped_color, recognized_board, solved_board)
    if save_stages:
        _save_image(output_dir / "11_solution_overlay.png", solution_overlay)

    _save_summary(output_dir / "run_summary.json", {
        "completed_stage": "solve",
        "recognized_board": recognized_board,
        "solved_board": solved_board,
    })

    print("Done!")


def _draw_solution(warped_color, recognized_board, solved_board):
    overlay = warped_color.copy()
    cell_size = warped_color.shape[0] // 9

    for row in range(9):
        for col in range(9):
            x = col * cell_size + int(cell_size * 0.28)
            y = (row + 1) * cell_size - int(cell_size * 0.22)

            if recognized_board[row][col] != 0:
                cv2.putText(overlay, str(recognized_board[row][col]), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 100, 0), 3, lineType=cv2.LINE_AA)
            else:
                cv2.putText(overlay, str(solved_board[row][col]), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 160, 0), 3, lineType=cv2.LINE_AA)

    return overlay


def _save_grid(path, board):
    lines = [" ".join(str(v) for v in row) for row in board]
    path.write_text("\n".join(lines) + "\n")


def _save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")


def _save_summary(path, data):
    path.write_text(json.dumps(data, indent=2))
