"""Test resistics.common.smooth"""


def test_getSmoothingLength() -> None:
    """Test getting smoothing length"""
    from resistics.common.smooth import getSmoothingLength

    assert getSmoothingLength(64) == 5
    assert getSmoothingLength(128) == 9
    assert getSmoothingLength(1598) == 101
    assert getSmoothingLength(1598, 21) == 77


def test_smooth1d() -> None:
    """Test single dimension smoothing"""
    from resistics.common.smooth import smooth1d
    import numpy as np
    import pytest

    with pytest.raises(ValueError):
        x = np.array([10])
        smooth1d(x, 11)
    with pytest.raises(ValueError):
        x = np.array([10, 5, 7])
        smooth1d(x, 7)
    # window length < 3
    x = np.array([10, 5, 7, 9, 7, 8])
    assert np.array_equal(smooth1d(x, 2), x)
    # do a smooth
    x = np.array([10, 5, 7, 9, 7, 8, 6, 7, 4, 2, 4, 6])
    expected = np.array([8.75, 6.75, 7, 8, 7.75, 7.25, 6.75, 6, 4.25, 3, 4, 5.5])
    assert len(x) == len(expected)
    assert np.array_equal(smooth1d(x, 5), expected)


# def test_smooth2d() -> None:
#     from resistics.common.smooth import smooth2d

#     kernel = np.outer(signal.hanning(winLen[0], 8), signal.gaussian(winLen[1], 8))
#     # pad to help the boundaries
#     padded = np.pad(x, ((winLen[0], winLen[0]), (winLen[1], winLen[1])), mode="edge")
#     # 2d smoothing
#     blurred = signal.fftconvolve(padded, kernel, mode="same")
#     return blurred[
#         winLen[0] : winLen[0] + x.shape[0], winLen[1] : winLen[1] + x.shape[1]
#     ]
