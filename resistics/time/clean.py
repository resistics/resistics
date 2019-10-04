import numpy as np
import math
from typing import Dict


def removeZeros(data: Dict):
    """Remove a stretch of zeros in the data

    This function finds a stretch of zeros and tries to fill them in with better data i.e. interpolated data or some such.

    Parameters
    ----------
    data : Dict
        Dictionary of data with channel as key and a np.ndarray as value

    Returns
    -------
    Dict
        Dictionary of data with channel as key and a np.ndarray as value (with zero stretches removed)
    """
    for chan in data:
        data[chan] = removeZerosSingle(data[chan])
    return data


def removeZerosSingle(data: np.ndarray) -> np.ndarray:
    """Remove a stretch of zeros in a data array

    This function finds a stretch of zeros and tries to fill them in with better data i.e. interpolated data or some such.

    Parameters
    ----------
    data : np.ndarray
        Array of data

    Returns
    -------
    np.ndarray
        Array of data with zeros removed
    """
    eps = 0.000000001  # use this because of floating point precision
    # set an x array
    x = np.arange(data.size)
    # find zero locations - this returns a tuple, take the first index
    zeroLocs = np.where(np.absolute(data) < eps)[0]
    if len(zeroLocs) == 0:
        return data  # no zeros to remove

    # now want to find consecutive zeros
    grouped = groupConsecutive(zeroLocs)
    indicesToFix = []
    # now find groups of 3+
    for g in grouped:
        if g.size >= 20:
            indicesToFix = indicesToFix + list(g)
    # now have the indices we want to fix
    # can go about interpolating values there
    indicesToFix = np.array(sorted(indicesToFix))
    mask = np.ones(data.size, np.bool)
    mask[indicesToFix] = 0
    data[indicesToFix] = np.interp(indicesToFix, x[mask], data[mask])
    return data


def groupConsecutive(vals: np.ndarray, stepsize: int = 1):
    """Takes an array of values and splits it into consecutive sections of stepsize

    In general, the stepsize is 1.
    
    Parameters
    ----------
    vals : np.ndarray
        A set of values to split into consecutive sections
    stepsize : int
        The stepsize between values that means they are consecutive

    Examples
    --------
    An array of [1,2,3,5,6,7,10,12,13] would be split into consecutive sections [1,2,3], [5,6,7], [10], [12,13]
    """
    return np.split(vals, np.where(np.diff(vals) != stepsize)[0] + 1)


def removeNans(data: Dict):
    """Remove NaNs in the data

    This function finds NaNs in the data and tries to fill them in with better data i.e. interpolated data or some such.

    Parameters
    ----------
    data : Dict
        Dictionary of data with channel as key and a np.ndarray as value

    Returns
    -------
    Dict
        Dictionary of data with channel as key and a np.ndarray as value (with zero stretches removed)
    """
    # find nan in the dataset and removes the values
    for chan in data:
        data[chan] = removeNansSingle(data[chan])
    return data


def removeNansSingle(data):
    """Remove NaNs in a data array

    This function finds NaNs in the np.ndarray and tries to fill them in with better data i.e. interpolated data or some such.

    Parameters
    ----------
    data : np.ndarray
        Array of data

    Returns
    -------
    np.ndarray
        Array of data with zeros removed
    """
    # set an x array
    x = np.arange(data.size)
    # find locations of nans - this is a bool array with True in locations with nan values
    nanLocs = np.isnan(data)
    # if no nans, do nothing
    if not np.any(nanLocs):
        return data  # no nans to remove
    # create mask
    mask = np.ones(data.size, np.bool)
    mask[nanLocs] = 0  # using numpy indexing with bool arrays
    # no need to group, want to remove every nan
    data[nanLocs] = np.interp(x[nanLocs], x[mask], data[mask])
    return data
