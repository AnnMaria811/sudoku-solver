import cv2
import numpy as np

from .config import PipelineConfig
from .geometry import order_quad_corners


def detect_grid(binary, image, config):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = image.shape[0] * image.shape[1] * config.min_board_area_ratio

    best_corners = None
    best_contour = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, perimeter * config.contour_epsilon_ratio, True)

        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect).astype(np.float32)

        if area > best_area:
            best_area = area
            best_corners = order_quad_corners(corners)
            best_contour = contour

    if best_corners is None:
        raise ValueError("Could not find the sudoku grid in the image")

    mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    contour_overlay = image.copy()
    cv2.drawContours(contour_overlay, [best_contour], -1, (0, 255, 0), thickness=3)

    corners_overlay = contour_overlay.copy()
    for i, point in enumerate(best_corners):
        x, y = int(point[0]), int(point[1])
        cv2.circle(corners_overlay, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(corners_overlay, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return mask, contour_overlay, corners_overlay, best_corners, float(best_area)
