import numpy as np
import scipy.interpolate as interp
from datetime import datetime, timedelta
from typing import Dict

from resistics.time.data import TimeData
from resistics.common.math import intdiv
from resistics.common.print import errorPrint


def interpolateToSecond(timeData: TimeData) -> TimeData:
    """Interpolate data to be on the second

    Some formats of time data (e.g. SPAM) do not start on the second with their sampling. This method interpolates so that sampling starts on the second and improves interoperability with other recording formats. As an example, consider recording at 10 Hz, with the first sample at 0.05 seconds. Then the sample times will be:

    .. code-block::text

        0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 ...

    Interpolating to the second will change the sample times to:

    .. code-block::text 

        0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00 1.10 1.20 ...

    .. warning::
        
        Do not use this method on data recording with a sampling frequency of less than 1Hz

    Parameters
    ----------
    timeData : TimeData
        Time data to interpolate onto the second
    
    Returns
    -------
    TimeData
        A new TimeData object interpolated to start on the second
    """
    startTimeInterp, numSamplesInterp, dataInterp = interpolateToSecondData(
        timeData.data, timeData.sampleFreq, timeData.startTime
    )
    # calculate the stop time
    sampleFreqInterp = timeData.sampleFreq
    stopTimeInterp = startTimeInterp + timedelta(
        seconds=(1.0 / sampleFreqInterp) * (numSamplesInterp - 1)
    )
    # the comments
    newcomment = "Time data interpolated to nearest second. New start time {}, new end time {}, new number of samples {}".format(
        startTimeInterp, stopTimeInterp, numSamplesInterp
    )
    commentsInterp = timeData.comments + [newcomment]
    # create a new time data object
    return TimeData(
        sampleFreqInterp, startTimeInterp, stopTimeInterp, dataInterp, commentsInterp
    )


def interpolateToSecondData(
    data: Dict[str, np.ndarray], sampleFreq: float, startTime: datetime
) -> Dict[str, np.ndarray]:
    """Interpolate data to be on the second

    Interpolates the sampling so that it coincides with full seconds. The function also shifts the start point to the next full second. This function will truncate the data to the previous full second.
    
    .. warning::
        
        Do not use this method on data recording with a sampling frequency of less than 1Hz
    
    Parameters
    ----------
    data : Dict
        Dictionary with channel as keys and data as values
    sampleFreq : float
        Sampling frequency of the data
    startTime : datetime
        Time of first sample
    
    Returns
    -------
    data : Dict
        Dictionary with channel as keys and data as values

    todo:
    This function needs to be more robust for low (< 1Hz) sample frequencies as the use of microseconds and seconds makes no sense for this    
    """
    # data properties
    chans = list(data.keys())
    samplePeriod = 1.0 / sampleFreq
    # set initial vals
    numSamples = data[chans[0]].size

    # now caluclate the interpolation
    microseconds = startTime.time().microsecond
    # check if the dataset already begins on a second
    if microseconds == 0:
        return startTime, numSamples, data  # do nothing, already on the second
    # now turn microseconds into a decimal
    microseconds = microseconds / 1000000.0
    # now calculate the number of complete samples till the next second
    eps = 0.000000001
    test = microseconds
    samplesToDrop = 0
    # this loop will always either calculate till the full second or the next sample passed the full second
    while test < 1.0 - eps:
        test += samplePeriod
        samplesToDrop += 1

    # if this is exact, i.e. integer number of samples to next second, just need to drop samples
    multiple = (1.0 - microseconds) / samplePeriod
    if np.absolute(multiple - samplesToDrop) < eps:  # floating point arithmetic
        dataInterp = {}  # create a new dictionary for data
        for chan in chans:
            dataInterp[chan] = data[chan][samplesToDrop:]
        # update the other data
        numSamplesInterp = numSamples - samplesToDrop
        startTimeInterp = startTime + timedelta(
            seconds=1.0 * samplesToDrop / sampleFreq
        )
        return startTimeInterp, numSamplesInterp, dataInterp

    # if here, then we have calculated one extra for samplesToDrop
    samplesToDrop -= 1

    # now the number of samples to the next full second is not an integer
    # interpolation will have to be performed
    shift = (multiple - samplesToDrop) * samplePeriod
    sampleShift = shift / samplePeriod
    x = np.arange(0, numSamples)
    xInterp = np.arange(samplesToDrop, numSamples - 1) + sampleShift
    # calculate return vars
    numSamplesInterp = xInterp.size
    startTimeInterp = (
        startTime
        + timedelta(seconds=1.0 * samplesToDrop / sampleFreq)
        + timedelta(seconds=shift)
    )

    # interpolation
    dataInterp = {}
    for chan in chans:
        tck = interp.splrep(x, data[chan], s=0)
        dataInterp[chan] = interp.splev(xInterp, tck, der=0)

    return startTimeInterp, numSamplesInterp, dataInterp


def fillGap(timeData1: TimeData, timeData2: TimeData) -> TimeData:
    """Fill gap between time series
    
    Fill gaps between two different recordings. The intent is to fill the gap when recording has been interrupted and there are two data files. Both times series must have the same sampling frequency. The missing timestamps and samples will be added to the timeseries and filled in using interpolation.

    Parameters
    ----------
    timeDat1 : TimeData
        Time series data
    timeData2 : TimeData
        Time series data

    Returns
    -------
    TimeData
        Time series data with gap filled
    """
    from resistics.time.clean import removeNansChan

    if timeData1.sampleFreq != timeData2.sampleFreq:
        errorPrint(
            "fillGap",
            "fillGap requires both timeData objects to have the same sample rate",
            quitrun=True,
        )
        return False
    sampleFreq = timeData1.sampleFreq
    sampleRate = 1.0 / sampleFreq
    timeDataFirst = timeData1
    timeDataSecond = timeData2
    if timeData1.startTime > timeData2.stopTime:
        timeDataFirst = timeData2
        timeDataSecond = timeData1
    # time data start and end times are inclusive
    gapStart = timeDataFirst.stopTime + timedelta(seconds=sampleRate)
    gapEnd = timeDataSecond.startTime - timedelta(seconds=sampleRate)
    # number of samples in the gap - add 1 because inclusive
    numSamplesGap = int(round((gapEnd - gapStart).total_seconds() * sampleFreq)) + 1
    # interpolate
    newData = {}
    for chan in timeDataFirst.chans:
        missing = np.empty(shape=(numSamplesGap))
        missing[:] = np.nan
        combined = np.concatenate([timeDataFirst[chan], missing, timeDataSecond[chan]])
        newData[chan] = removeNansChan(combined)
    comment = (
        ["-----------------------------", "TimeData1 comments"]
        + timeDataFirst.comments
        + ["-----------------------------", "TimeData2 comments"]
        + timeDataSecond.comments
    )
    comment += ["-----------------------------"] + [
        "Gap filled from {} to {}".format(gapStart, gapEnd)
    ]
    return TimeData(
        sampleFreq, timeDataFirst.startTime, timeDataSecond.stopTime, newData, comment,
    )

