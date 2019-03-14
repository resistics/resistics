from datetime import datetime, timedelta
import calendar
import numpy as np
import math


def gIndex2datetime(
    gIndex: int, refTime: datetime, fs: float, windowSize: int, windowOverlap: int
):
    """Global index to datetime convertor

    Global index 0 corresponds to reference time

    Parameters
    ----------
    gIndex : int
        Globel index
    refTime : datetime.datetime
        Reference time   
    fs : float
        Sampling frequency in Hz
    windowSize : int
        Size of windows
    windowOverlap : int
        Size of window overlaps     

    Returns
    -------
    startTime : datetime.datetime
        Start time of global window gIndex
    endTime : datetime.datetime
        End time of global window gIndex        
    """

    # global index 0 starts at refTime
    timeOffset = 1.0 * (windowSize - windowOverlap) / fs
    totalOffset = gIndex * timeOffset
    startTime = refTime + timedelta(seconds=totalOffset)
    # windowSize - 1 because inclusive of start sample
    endTime = startTime + timedelta(seconds=1.0 * (windowSize - 1) / fs)
    return startTime, endTime


def gArray2datetime(
    gArray: np.ndarray,
    refTime: datetime,
    fs: float,
    windowSize: int,
    windowOverlap: int,
):
    """Global index array to datetime convertor

    Global index 0 corresponds to reference time

    Parameters
    ----------
    gArray : np.ndarray
        Globel indices array
    refTime : datetime.datetime
        Reference time   
    fs : float
        Sampling frequency in Hz
    windowSize : int
        Size of windows
    windowOverlap : int
        Size of window overlaps     

    Returns
    -------
    startTime : np.ndarray of datetime.datetime
        Start times of global windows
    endTime : np.ndarray of datetime.datetime
        End times of global windows         
    """

    arrSize = gArray.size
    startTime = np.zeros(shape=(arrSize), dtype=datetime)
    endTime = np.zeros(shape=(arrSize), dtype=datetime)
    for i in range(0, arrSize):
        startTime[i], endTime[i] = gIndex2datetime(
            gArray[i], refTime, fs, windowSize, windowOverlap
        )
    return startTime, endTime


def gIndex2timestamp(
    gIndex: int, refTime: datetime, fs: float, windowSize: int, windowOverlap: int
):
    """Global index to timestamp convertor

    Global index 0 corresponds to reference time

    Parameters
    ----------
    gIndex : int
        Globel index
    refTime : datetime.datetime
        Reference time   
    fs : float
        Sampling frequency in Hz
    windowSize : int
        Size of windows
    windowOverlap : int
        Size of window overlaps     

    Returns
    -------
    startTime : UNIX timestamp
        Start time of global window gIndex
    endTime : UNIX timestamp
        End time of global window gIndex        
    """

    startTime, endTime = gIndex2datetime(gIndex, refTime, fs, windowSize, windowOverlap)
    return calendar.timegm(startTime.timetuple()), calendar.timegm(endTime.timetuple())


def gArray2timestamp(
    gArray: np.ndarray,
    refTime: datetime,
    fs: float,
    windowSize: int,
    windowOverlap: int,
):
    """Global index array to timestamp convertor

    Global index 0 corresponds to reference time

    Parameters
    ----------
    gArray : np.ndarray
        Globel indices array
    refTime : datetime.datetime
        Reference time   
    fs : float
        Sampling frequency in Hz
    windowSize : int
        Size of windows
    windowOverlap : int
        Size of window overlaps     

    Returns
    -------
    startTime : UNIX timestamp
        Start times of global windows
    endTime : UNIX timestamp
        End times of global windows         
    """

    arrSize = gArray.size
    startTime = np.zeros(shape=(arrSize), dtype=datetime)
    endTime = np.zeros(shape=(arrSize), dtype=datetime)
    for i in range(0, arrSize):
        startTime[i], endTime[i] = gIndex2timestamp(
            gArray[i], refTime, fs, windowSize, windowOverlap
        )
    return startTime, endTime


def datetime2gIndex(
    refTime: datetime, inTime: datetime, fs: float, windowSize: int, windowOverlap: int
):
    """Datetime to global index convertor

    Global index 0 corresponds to reference time. This returns the global index of the time window nearest to inTime

    Parameters
    ----------
    refTime : datetime.datetime
        Reference time  
    inTime : datetime.datetime
        Time for which you want closest global index 
    fs : float
        Sampling frequency in Hz
    windowSize : int
        Size of windows
    windowOverlap : int
        Size of window overlaps     

    Returns
    -------
    gIndex : int
        Global window index closest to inTime
    firstWindowTime : datetime.datetime
        Datetime of the global window        
    """

    # need to return the next one close
    # calculate
    deltaRefStart = inTime - refTime
    winStartIncrement = (windowSize - windowOverlap) / fs
    # calculate number of windows started before reference time
    # and then by taking the ceiling, find the global index of the first window in the data
    gIndex = int(math.ceil(deltaRefStart.total_seconds() / winStartIncrement))
    # calculate start time of first global window
    offsetSeconds = gIndex * winStartIncrement
    # calculate the first window time
    firstWindowTime = refTime + timedelta(seconds=offsetSeconds)
    return gIndex, firstWindowTime
