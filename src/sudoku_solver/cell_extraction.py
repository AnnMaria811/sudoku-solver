import cv2
import numpy as np

from .config import PipelineConfig
from .ocr import centre_on_canvas


def extract_cells(warped_binary, warped_color, config):
    cell_size = warped_binary.shape[0] // 9
    margin = max(4, int(cell_size * config.cell_margin_ratio))
    min_blob_area = max(12, int(cell_size * cell_size * config.min_component_area_ratio))

    digit_tiles = []
    overlay = warped_color.copy()

    for row in range(9):
        for col in range(9):
            y0, y1 = row * cell_size, (row + 1) * cell_size
            x0, x1 = col * cell_size, (col + 1) * cell_size

            cell = warped_binary[y0:y1, x0:x1]
            inner = cell[margin: cell.shape[0] - margin, margin: cell.shape[1] - margin]
            cleaned = _largest_blob(inner, min_blob_area)

            digit_tiles.append(centre_on_canvas(cleaned, config.digit_canvas_size))

            colour = (0, 200, 0) if np.count_nonzero(cleaned) > 0 else (0, 0, 200)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, 1)

    return digit_tiles, overlay


def _largest_blob(image, min_area):
    if image.size == 0 or np.count_nonzero(image) == 0:
        return np.zeros_like(image)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(image)

    best_label, best_area = 0, 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label

    if best_area < min_area:
        return np.zeros_like(image)

    result = np.zeros_like(image)
    result[labels == best_label] = 255
    return result
