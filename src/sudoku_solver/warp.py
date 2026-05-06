import cv2
import numpy as np

from .config import PipelineConfig
from .preprocess import preprocess_image


def warp_grid(image, corners, config):
    size = config.warp_size

    destination = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    warped_color = cv2.warpPerspective(image, matrix, (size, size))

    _gray, _normalized, warped_binary = preprocess_image(warped_color, config)

    return warped_color, warped_binary
