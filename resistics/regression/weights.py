"""
Weight functions to use for weighted least squares. The idea is to downweight samples which have high residual values as these are likely outliers and should be removed. 
There are various weighting functions that can be used. 
"""
import numpy as np
from typing import Union


def getWeights(
    r: np.ndarray, weight: str = "bisquare", k: Union[None, float] = None
) -> np.ndarray:
    """Robust weighting schemes
    
    Parameters
    ----------
    r : np.ndarray
        Residuals from least squares
    weight : str
        The type of weighting to use. Default is bisquare.
    k : float, None
        The tuning constant for the weights function

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    weight = weight.lower()
    if weight == "bisquare":
        return bisquare(r, k)
    elif weight == "huber":
        return huber(r, k)
    elif weight == "hampel":
        return hampel(r, k)
    elif weight == "trimmedmean":
        return trimmedMean(r, k)
    elif weight == "andrewswave":
        return andrewsWave(r, k)
    elif weight == "leastsquares":
        return leastSquares(r)
    else:
        return bisquare(r, k)


def bisquare(r: np.ndarray, k: Union[None, float] = None) -> np.ndarray:
    """Bisquare location weights
    
    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float, None
        Tuning parameter. If None, a standard value will be used.

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    if k is None:
        k = 4.685
    ones = np.ones(shape=(r.size), dtype="complex")
    threshR = np.minimum(ones, np.absolute(r / k))
    return np.power((1 - np.power(threshR, 2)), 2).real


def huber(r: np.ndarray, k: Union[None, float] = None) -> np.ndarray:
    """Huber location weights
    
    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float
        Tuning parameter. If None, a standard value will be used.

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    if k is None:
        k = 1.345
    weights = np.ones(shape=r.size, dtype="complex")
    for idx, val in enumerate(np.absolute(r)):
        if val > k:
            # relying on numpy doing the right thing when dividing by zero
            weights[idx] = k / val
    return weights.real


def hampel(r: np.ndarray, k: Union[None, float] = None) -> np.ndarray:
    """Hampel location weights
    
    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float
        Tuning parameter. If None, a standard value will be used.

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    if k is None:
        k = 8
    b = k / 2
    a = k / 4
    weights = np.ones(shape=r.size, dtype="complex")
    for idx, val in enumerate(np.absolute(r)):
        if val > a and val <= b:
            weights[idx] = a / val
        elif val > b and val <= k:
            weights[idx] = a * (k - val) / (val * (k - b))
        elif val > k:
            weights[idx] = 0
    return weights.real


def trimmedMean(r: np.ndarray, k: Union[None, float] = None) -> np.ndarray:
    """Trimmed mean location weights
    
    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float
        Tuning parameter. If None, a standard value will be used.

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    if k is None:
        k = 2
    weights = np.zeros(shape=r.size, dtype="complex")
    indices = np.where(np.absolute(r) <= k)
    weights[indices] = 1
    return weights.real


def andrewsWave(r: np.ndarray, k: Union[None, float] = None) -> np.ndarray:
    """Andrews Wave location weights
    
    Parameters
    ----------
    r : np.ndarray
        Residuals
    k : float
        Tuning parameter. If None, a standard value will be used.

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    if k is None:
        k = 1.339
    weights = np.zeros(shape=r.size, dtype="complex")
    testVal = k * np.pi
    for idx, val in enumerate(np.absolute(r)):
        if val < testVal:
            weights[idx] = np.sin(val / k) / (val / k)
    return weights.real


def leastSquares(r: np.ndarray):
    """Least squares weights, which are all equal to 1

    Parameters
    ----------
    r : np.ndarray
        Residuals

    Returns
    -------
    weights : np.ndarray
        The robust weights
    """
    return np.ones(shape=(r.size), dtype="complex")
