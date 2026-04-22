from dataclasses import dataclass

import cv2
import numpy as np

from .config import PipelineConfig
from .ocr import center_digit_component


@dataclass
class CellArtifacts:
    digit_tiles: list[np.ndarray]
    occupancy_tiles: list[np.ndarray]
    overlay: np.ndarray


def extract_cells(warped_binary: np.ndarray, warped_color: np.ndarray, config: PipelineConfig) -> CellArtifacts:
    cell_size = warped_binary.shape[0] // 9
    margin = max(4, int(cell_size * config.cell_margin_ratio))
    min_component_area = max(12, int((cell_size * cell_size) * config.min_component_area_ratio))

    digit_tiles = []
    occupancy_tiles = []
    overlay = warped_color.copy()

    for row in range(9):
        for col in range(9):
            # compute the pixel bounds of this cell
            y0 = row * cell_size
            x0 = col * cell_size
            y1 = (row + 1) * cell_size
            x1 = (col + 1) * cell_size

            # crop the cell and strip the border margin to remove grid lines
            cell = warped_binary[y0:y1, x0:x1]
            inner = cell[margin : cell.shape[0] - margin, margin : cell.shape[1] - margin]

            # keep only the largest blob — that should be the digit if there is one
            occupancy = _largest_component(inner, min_component_area=min_component_area)

            # centre the digit on a fixed-size canvas for consistent template matching
            digit_tiles.append(center_digit_component(occupancy, config.digit_canvas_size))
            occupancy_tiles.append(occupancy)

            # colour the cell green if it looks occupied, red if it looks empty
            colour = (0, 255, 0) if np.count_nonzero(occupancy) else (255, 0, 0)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, 1)

    return CellArtifacts(digit_tiles=digit_tiles, occupancy_tiles=occupancy_tiles, overlay=overlay)


def _largest_component(image: np.ndarray, *, min_component_area: int) -> np.ndarray:
    if image.size == 0 or np.count_nonzero(image) == 0:
        return np.zeros_like(image)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(image)

    # find the component with the biggest area (skip label 0 which is background)
    best_index = 0
    best_area = 0
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_index = label_idx

    # ignore tiny specks that are probably noise
    if best_area < min_component_area:
        return np.zeros_like(image)

    component = np.zeros_like(image)
    component[labels == best_index] = 255
    return component
