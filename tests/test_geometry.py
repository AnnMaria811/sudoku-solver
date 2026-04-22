from __future__ import annotations

import numpy as np

from sudoku_solver.geometry import order_quad_corners


def test_order_quad_corners_returns_clockwise_points_from_top_left() -> None:
    points = np.array(
        [
            [400.0, 100.0],
            [110.0, 390.0],
            [390.0, 410.0],
            [90.0, 120.0],
        ],
        dtype=np.float32,
    )

    ordered = order_quad_corners(points)

    np.testing.assert_allclose(
        ordered,
        np.array(
            [
                [90.0, 120.0],
                [400.0, 100.0],
                [390.0, 410.0],
                [110.0, 390.0],
            ],
            dtype=np.float32,
        ),
    )
