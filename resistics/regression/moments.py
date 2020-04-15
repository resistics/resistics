"""
These are measures of location and scale.
Location is the first moment of a probability distribution. Often this is taken to be the mean or median and sometimes mode depending on the context. 
Scale is the second moment of a probability distribution and is a measure of the spread or dispersion. For Gaussian distributions, this is usually the variance or standard deviation.
When nominally Gaussian data with large outliers is encountered, the mean and standard deviation are adversely affected. Therefore, it can be useful to use more robust measures of location and scale. 
The simplest robust measures of location and scale are median and the MAD respectively.
"""
import numpy as np


def getLocation(data: np.ndarray, location: str = "median") -> float:
    """Get an estimate of the location. This is the first moment of a distribution
    
    Parameters
    ----------
    np.ndarray
        Data for which to estimate location
    
    Returns
    -------
    float
        The estimate of location
    """
    if location == "median":
        return np.median(data)
    else:
        return np.median(data)


def getScale(data: np.ndarray, scale: str = "mad0") -> float:
    """Get an estimate of the scale. This is the second moment of a distribution
    
    Parameters
    ----------
    np.ndarray
        Data for which to estimate location
    
    Returns
    -------
    float
        The estimate of scale
    """
    if scale == "mad":
        return mad(data)
    elif scale == "mad0":
        return mad0(data)
    else:
        return mad0(data)


def mad(data: np.ndarray) -> float:
    """Median absolute deviation from median

    Median deviation from the median. A common robust measure of scale. The standard deviation is not robust against outliers, hence use the MAD. The median is used as the location.
    
    Parameters
    ----------
    np.ndarray
        Data for which to calculate MAD
    
    Returns
    -------
    float
        The MAD    
    """
    absData = np.absolute(data)
    mad = np.median(np.absolute(absData - np.median(absData)))
    return mad / 0.67448975019608171


def mad0(data: np.ndarray) -> float:
    """Median absolute deviation using an estimate of the location as 0

    When the location estimate is zero (rather than the median), the MAD essentially reduces to a median. This should be over non-zero data as do not want to return zero. Useful for calculating variance of residuals.

    Parameters
    ----------
    np.ndarray
        Data for which to calculate MAD. This is often residuals when using 0 as an estimate of location. 
    
    Returns
    -------
    float
        The MAD using zero as an esimate of location   
    """
    absData = np.absolute(data)
    inputIndices = np.where(absData != 0.0)
    mad = np.median(absData[inputIndices])
    return mad / 0.67448975019608171
