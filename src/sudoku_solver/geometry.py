import numpy as np


def order_quad_corners(points: np.ndarray) -> np.ndarray:
    """Return the four corners ordered: top-left, top-right, bottom-right, bottom-left."""
    if points.shape != (4, 2):
        raise ValueError("expected four 2D points")

    points = points.astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    # top-left has the smallest x+y, bottom-right has the largest x+y
    sums = points[:, 0] + points[:, 1]
    ordered[0] = points[np.argmin(sums)]   # top-left
    ordered[2] = points[np.argmax(sums)]   # bottom-right

    # top-right has the smallest y-x, bottom-left has the largest y-x
    diffs = points[:, 1] - points[:, 0]
    ordered[1] = points[np.argmin(diffs)]  # top-right
    ordered[3] = points[np.argmax(diffs)]  # bottom-left

    return ordered
