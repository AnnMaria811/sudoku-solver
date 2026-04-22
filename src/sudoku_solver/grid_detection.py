from dataclasses import dataclass

import cv2
import numpy as np

from .config import PipelineConfig
from .geometry import order_quad_corners


@dataclass
class GridDetectionArtifacts:
    mask: np.ndarray
    contour_overlay: np.ndarray
    corners_overlay: np.ndarray
    corners: np.ndarray
    contour_area: float


def detect_grid(binary: np.ndarray, image: np.ndarray, config: PipelineConfig) -> GridDetectionArtifacts:
    # find all contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # the grid must take up a reasonable portion of the image
    min_area = image.shape[0] * image.shape[1] * config.min_board_area_ratio

    best_corners = None
    best_contour = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # simplify the contour shape to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, perimeter * config.contour_epsilon_ratio, True)

        if len(approx) == 4:
            # we got a nice quadrilateral
            candidate = approx.reshape(4, 2).astype(np.float32)
        else:
            # fall back to the minimum bounding rectangle
            rect = cv2.minAreaRect(contour)
            candidate = cv2.boxPoints(rect).astype(np.float32)

        ordered = order_quad_corners(candidate)

        # prefer candidates that are closer to square (the grid should be roughly square)
        width = np.linalg.norm(ordered[1] - ordered[0])
        height = np.linalg.norm(ordered[3] - ordered[0])
        if min(width, height) == 0:
            continue

        aspect_penalty = abs(1.0 - (width / height))
        score = float(area) - (aspect_penalty * area * 0.25)

        if score > best_score:
            best_score = score
            best_corners = ordered
            best_contour = contour

    if best_corners is None or best_contour is None:
        raise ValueError("unable to detect a Sudoku grid")

    # create a filled mask of the grid area
    mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    # draw the detected contour on the original image for debugging
    contour_overlay = image.copy()
    cv2.drawContours(contour_overlay, [best_contour], -1, (0, 255, 0), thickness=3)

    # mark the four corners with numbered circles
    corners_overlay = contour_overlay.copy()
    for idx, point in enumerate(best_corners):
        x, y = point.astype(int)
        cv2.circle(corners_overlay, (x, y), 10, (0, 0, 255), thickness=-1)
        cv2.putText(
            corners_overlay,
            str(idx + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )

    return GridDetectionArtifacts(
        mask=mask,
        contour_overlay=contour_overlay,
        corners_overlay=corners_overlay,
        corners=best_corners,
        contour_area=float(cv2.contourArea(best_contour)),
    )
