import numpy as np
import math
from typing import Dict, Union

from resistics.time.data import TimeData


def removeZeros(timeData: TimeData, conzeros: int = 20) -> TimeData:
    """Remove a stretch of zeros in the data

    This function finds a stretch of zeros and fills them with interpolated data. The function will return a new TimeData object.

    Parameters
    ----------
    timeData : TimeData
        A TimeData instance
    conzeros : int, optional
        How many consecutive zeros (in samples) are required before they are considered to be zeros to remove

    Returns
    -------
    TimeData
        A new TimeData object with zeros removed
    """
    data = {}
    for chan in timeData:
        data[chan] = removeZerosChan(timeData[chan], conzeros)
    comments = timeData.comments + [
        "Sections of {:d} consecutive zeros have been interpolated".format(conzeros)
    ]
    return TimeData(
        timeData.sampleFreq, timeData.startTime, timeData.stopTime, data, comments,
    )


def removeZerosChan(data: np.ndarray, conzeros: int = 20) -> np.ndarray:
    """Remove a stretch of zeros in a data array

    This function finds a stretch of zeros and tries to fill them in with better data i.e. interpolated data or some such.

    Parameters
    ----------
    data : np.ndarray
        Array of data
    conzeros : int, optional
        How many consecutive zeros (in samples) are required before they are considered to be zeros to remove

    Returns
    -------
    np.ndarray
        Array of data with zeros removed
    """
    # find close to zero locations
    eps = 0.000000001
    zeroLocs = np.where(np.absolute(data) < eps)[0]
    if len(zeroLocs) == 0:
        return data
    # find consecutive zeros
    grouped = groupConsecutive(zeroLocs)
    indicesToFix = []
    # find groups of 20 or more
    for g in grouped:
        if g.size >= conzeros:
            indicesToFix = indicesToFix + list(g)
    # no zero groups big enough to fix
    if len(indicesToFix) == 0:
        return data
    # have indices to fix can interpolate values there
    x = np.arange(data.size)
    indicesToFix = np.array(sorted(indicesToFix))
    mask = np.ones(data.size, dtype=np.bool)
    # set indices to fix to False
    mask[indicesToFix] = 0
    data[indicesToFix] = np.interp(indicesToFix, x[mask], data[mask])
    return data


def groupConsecutive(vals: np.ndarray, stepsize: int = 1) -> np.ndarray:
    """Takes an array of values and splits it into consecutive sections of stepsize

    In general, the stepsize is 1.
    
    Parameters
    ----------
    vals : np.ndarray
        A set of values to split into consecutive sections
    stepsize : int
        The stepsize between values that means they are consecutive
    
    Returns
    -------
    np.ndarray[np.ndarray]
        Array of sections of consecutive numbers

    Examples
    --------
    An array of [1,2,3,5,6,7,10,12,13] would be split into consecutive sections [1,2,3], [5,6,7], [10], [12,13]
    """
    return np.split(vals, np.where(np.diff(vals) != stepsize)[0] + 1)


def removeNans(timeData: TimeData) -> TimeData:
    """Remove NaNs in the data

    This function finds NaNs in the data and fills them with interpolated data. A new TimeData object will be returned.

    Parameters
    ----------
    timeData : TimeData
        A TimeData instance 

    Returns
    -------
    TimeData
        A TimeData object with nans removed
    """
    data = {}
    for chan in timeData:
        data[chan] = removeNansChan(timeData[chan])
    comments = timeData.comments + ["NaN values in data have been interpolated"]
    return TimeData(
        timeData.sampleFreq, timeData.startTime, timeData.stopTime, data, comments,
    )


def removeNansChan(data: np.ndarray) -> np.ndarray:
    """Remove NaNs in a data array

    Find NaNs in an np.ndarray and fill them in with better data i.e. interpolated data or some such.

    Parameters
    ----------
    data : np.ndarray
        Array of data

    Returns
    -------
    np.ndarray
        Array of data with nans removed
    """
    # find locations of nans in a bool array
    nanLocs = np.isnan(data)
    if not np.any(nanLocs):
        return data
    # create mask and set nan locations to False
    mask = np.ones(data.size, np.bool)
    mask[nanLocs] = 0
    # interpolate all nan locations
    x = np.arange(data.size)
    data[nanLocs] = np.interp(x[nanLocs], x[mask], data[mask])
    return data
