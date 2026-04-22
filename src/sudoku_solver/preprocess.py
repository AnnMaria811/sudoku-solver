from dataclasses import dataclass

import cv2
import numpy as np

from .config import PipelineConfig


@dataclass
class PreprocessArtifacts:
    gray: np.ndarray
    normalized: np.ndarray
    binary: np.ndarray


def preprocess_image(image: np.ndarray, config: PipelineConfig) -> PreprocessArtifacts:
    # convert the colour image to grayscale so we can work with pixel intensities
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply CLAHE to even out the lighting across the image
    # this helps when parts of the puzzle are in shadow or overexposed
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)

    # blur the image slightly to reduce noise before thresholding
    blurred = cv2.GaussianBlur(
        normalized,
        (config.gaussian_kernel_size, config.gaussian_kernel_size),
        0,
    )

    # adaptive threshold turns the image into black and white
    # "adaptive" means the threshold is calculated per region, not globally,
    # which works much better for uneven lighting
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config.adaptive_block_size,
        config.adaptive_c,
    )

    # morphological close fills in small gaps in the grid lines
    kernel = np.ones(
        (config.morphological_kernel_size, config.morphological_kernel_size),
        dtype=np.uint8,
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return PreprocessArtifacts(gray=gray, normalized=normalized, binary=binary)
