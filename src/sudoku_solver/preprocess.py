import cv2
import numpy as np

from .config import PipelineConfig


def preprocess_image(image, config):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)

    blurred = cv2.GaussianBlur(normalized, (config.gaussian_kernel_size, config.gaussian_kernel_size), 0)

    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config.adaptive_block_size,
        config.adaptive_c,
    )

    kernel = np.ones((config.morphological_kernel_size, config.morphological_kernel_size), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return gray, normalized, binary
