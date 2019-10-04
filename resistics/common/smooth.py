import numpy as np
import scipy.signal as signal
from typing import List

from resistics.common.math import intdiv


def smooth1d(x: np.ndarray, winLen: int = 11, window: str = "hanning"):
    """Smooth in 1 dimension

    Parameters
    ----------
    x : np.ndarray
        The data array to be smoothed
    winLen : int
        The number of samples to smooth across
    window : str
        The window function
	
    Notes
    ----- 
    The data is padded before the smoothing to ensure that the output data is the same size as the input. 

    todo:
    The window parameter could be the window itself if an array instead of a string
	"""

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < winLen:
        raise ValueError("Input vector needs to be bigger than window size.")
    if winLen < 3:
        return x

    s = np.pad(x, (winLen, winLen), mode="edge")
    if window == "flat":  # moving average
        w = np.ones(winLen, "d")
    else:
        w = eval("signal." + window + "(winLen)")

    # assume winLen is odd
    off = winLen + intdiv((winLen - 1), 2)
    if winLen % 2 == 0:  # check if even and recalc off
        off = winLen + intdiv(winLen, 2)

    y = np.convolve(s, w / w.sum(), mode="full")
    return y[off : off + x.size]


def smooth2d(x: np.ndarray, winLen: List[int] = [5, 5], window: str = "hanning"):
    """Smooth in 2-D

    This smooths in a window and across windows too

    Parameters
    ----------
    x : np.ndarray
        A 2-D array to be smoothed
    winLen : List[int]
        A two element list with winLen[0] being smoothing across windows and winLen[1] the smoothing within a window
    window : str
        The window function to use
    """

    kernel = np.outer(signal.hanning(winLen[0], 8), signal.gaussian(winLen[1], 8))
    # pad to help the boundaries
    padded = np.pad(x, ((winLen[0], winLen[0]), (winLen[1], winLen[1])), mode="edge")
    # 2d smoothing
    blurred = signal.fftconvolve(padded, kernel, mode="same")
    return blurred[
        winLen[0] : winLen[0] + x.shape[0], winLen[1] : winLen[1] + x.shape[1]
    ]
