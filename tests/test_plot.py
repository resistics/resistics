import pytest
from typing import List
import numpy as np


@pytest.mark.parametrize(
    "y, max_pts, x_expected, y_expected",
    [
        ([0, 1, 3, 4, 2, 3, 4, 3, 4, 5, 5, 5], 5, [0, 3, 4, 9, 11], [0, 4, 2, 5, 5]),
    ],
)
def test_lttb_downsample(
    y: List, max_pts: int, x_expected: List, y_expected: List
) -> None:
    """Test lttb downsampling"""
    from resistics.plot import lttb_downsample

    x = np.arange(len(y))
    y = np.array(y)
    nx, ny = lttb_downsample(x, y, max_pts=max_pts)
    np.testing.assert_array_equal(nx, x_expected)
    np.testing.assert_array_equal(ny, y_expected)


@pytest.mark.parametrize(
    "y, max_pts, x_expected, y_expected",
    [
        (
            [0, 1, 3, 4, 2, 3, 4, 3, 4, 5, 5, 5],
            5,
            [0, 3, 4, 9, 11],
            [0, 4, 2, 5, 5],
        ),
        (
            [0, 1, 3, 4, 2, 3, 4, 3, 4, 5, 5, 5],
            None,
            np.arange(12),
            [0, 1, 3, 4, 2, 3, 4, 3, 4, 5, 5, 5],
        ),
    ],
)
def test_apply_lttb(y: List, max_pts: int, x_expected: List, y_expected: List) -> None:
    """Testing the helper function"""
    from resistics.plot import apply_lttb

    x_new, y_new = apply_lttb(np.array(y), max_pts)
    np.testing.assert_array_equal(x_expected, x_new)
    np.testing.assert_array_equal(y_expected, y_new)
