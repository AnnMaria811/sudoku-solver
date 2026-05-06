import numpy as np


def order_quad_corners(points):
    points = points.astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points[:, 0] + points[:, 1]
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]

    diffs = points[:, 1] - points[:, 0]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]

    return ordered
