import numpy as np
import math
from typing import List


def getEvaluationFreq(fs: float, minFreq: float) -> np.ndarray:
    """Calculate evaluation frequencies with mimum allowable frequency

    Highest frequency is Nyquist/2. Lowest allowable frequency is provided by user
    f_i = f_max / pow(2,(i-1)/2)
    
    Parameters
    ----------
    fs : float
        Sampling frequency
    minFreq : float
        Minimum allowable frequency    

    Returns
    -------
    out : np.ndarray
        Array of evaluation frequencies
    """
    fmax: float = fs / 4
    freq: List = []
    i: int = 1
    f: float = fmax
    while f > minFreq:
        f = fmax / math.pow(2, (i - 1.0) / 2.0)
        freq.append(f)
        i = i + 1
    return np.array(freq)


def getEvaluationFreqSize(fs: float, numFreq: int) -> np.ndarray:
    """Calculate evaluation frequencies with maximum size

    Highest frequency is Nyquist/2 and number of evaluation frequencies is set by user
    f_i = f_max / pow(2,(i-1)/2)
    
    Parameters
    ----------
    fs : float
        Sampling frequency
    numFreq : int
        Number of evaluation frequencies    

    Returns
    -------
    out : np.ndarray
        Array of evaluation frequencies
    """
    fmax: float = fs / 4
    freq: List = []
    i: int = 1
    f: float = fmax
    while i <= numFreq:
        f = fmax / math.pow(2, (i - 1.0) / 2.0)
        freq.append(f)
        i = i + 1
    return np.array(freq)

