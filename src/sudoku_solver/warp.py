from dataclasses import dataclass

import cv2
import numpy as np

from .config import PipelineConfig
from .preprocess import preprocess_image


@dataclass
class WarpArtifacts:
    color: np.ndarray
    gray: np.ndarray
    binary: np.ndarray
    matrix: np.ndarray
    inverse_matrix: np.ndarray


def warp_grid(image: np.ndarray, corners: np.ndarray, config: PipelineConfig) -> WarpArtifacts:
    # define the four destination corners of the output square
    size = config.warp_size
    destination = np.array(
        [
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1],
        ],
        dtype=np.float32,
    )

    # compute the perspective transform that maps the grid corners to the destination square
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)

    # also compute the inverse so we can map back if needed
    inverse_matrix = cv2.getPerspectiveTransform(destination, corners.astype(np.float32))

    # apply the warp to get a flat, top-down view of the grid
    color = cv2.warpPerspective(image, matrix, (size, size))

    # preprocess the warped image to get binary and grayscale versions
    preprocessed = preprocess_image(color, config)

    return WarpArtifacts(
        color=color,
        gray=preprocessed.gray,
        binary=preprocessed.binary,
        matrix=matrix,
        inverse_matrix=inverse_matrix,
    )
