import numpy as np
import scipy.interpolate as interp
from datetime import datetime, timedelta
from typing import Dict

# import from package
from resistics.dataObjects.timeData import TimeData
from resistics.utilities.utilsMath import intdiv
from resistics.utilities.utilsPrint import errorPrint


def interpolateToSecond(timeData: TimeData) -> TimeData:
    """Interpolate data to be on the second

    Some formats of time data (e.g. SPAM) do not start on the second with their sampling. This method interpolates so that sampling starts on the second and improves interoperability with other recording formats. 

    Parameters
    ----------
    timeData : TimeData
        Time data to interpolate onto the second
    
    Returns
    -------
    TimeData
        Time data interpolated to start on the second
    """

    startTimeInterp, numSamplesInterp, dataInterp = interpolateToSecondData(
        timeData.data, timeData.sampleFreq, timeData.startTime
    )
    timeData.numSamples = numSamplesInterp
    timeData.startTime = startTimeInterp
    # calculate end timeEnd
    timeData.stopTime = timeData.startTime + timedelta(
        seconds=(1.0 / timeData.sampleFreq) * (timeData.numSamples - 1)
    )
    timeData.data = dataInterp
    timeData.addComment(
        "Time data interpolated to nearest second. New start time {}, new end time {}, new number of samples {} ".format(
            timeData.startTime, timeData.stopTime, timeData.numSamples
        )
    )
    return timeData


def interpolateToSecondData(
    data: Dict[str, np.ndarray], sampleFreq: float, startTime: datetime
) -> Dict[str, np.ndarray]:
    """Interpolate data to be on the second

    Interpolates the sampling so that it coincides with full seconds. The function also shifts the start point to the next full second
    WARNING: Do not use this method on data recording with a sampling frequency of less than 1Hz
    
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

    Notes
    -----
    This function will truncate the data to the next second.

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

    # do the interpolation
    dataInterp = {}
    for chan in chans:
        # interpFunc = interp.InterpolatedUnivariateSpline(x, data[chan])
        # dataInterp[chan] = interpFunc(xInterp)
        tck = interp.splrep(x, data[chan], s=0)
        dataInterp[chan] = interp.splev(xInterp, tck, der=0)

    # need to calculate how much the
    return startTimeInterp, numSamplesInterp, dataInterp


def fillGap(timeData1, timeData2):
    """Fill gap between time series
    
    Fill gaps between two different recordings. The intent is to fill the gap when recording has been interrupted and there are two data files. Both times series must have the same sampling frequency.

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

    if timeData1.sampleFreq != timeData2.sampleFreq:
        errorPrint(
            "fillGap",
            "fillGap requires both timeData objects to have the same sample rate",
            quitRun=True,
        )
        return False
    sampleFreq = timeData1.sampleFreq
    sampleRate = 1.0 / sampleFreq
    timeDataFirst = timeData1
    timeDataSecond = timeData2
    if timeData1.startTime > timeData2.stopTime:
        timeDataFirst = timeData2
        timeDataSecond = timeData1
    # now want to do a simple interpolation between timeDataFirst and timeDataSecond
    # recall, these times are inclusive, so want to do the samples in between
    # this is mostly for clarity of programming
    gapStart = timeDataFirst.stopTime + timedelta(seconds=sampleRate)
    gapEnd = timeDataSecond.startTime - timedelta(seconds=sampleRate)
    # calculate number of samples in the gap
    numSamplesGap = (
        int(round((gapEnd - gapStart).total_seconds() * sampleFreq)) + 1
    )  # add 1 because inclusive
    # now want to interpolate
    newData = {}
    for chan in timeDataFirst.chans:
        startVal = timeDataFirst.data[chan][-1]
        endVal = timeDataSecond.data[chan][0]
        increment = 1.0 * (endVal - startVal) / (numSamplesGap + 2)
        fillData = np.zeros(shape=(numSamplesGap), dtype=timeDataFirst.data[chan].dtype)
        for i in range(0, numSamplesGap):
            fillData[i] = startVal + (i + 1) * increment
        newData[chan] = np.concatenate(
            [timeDataFirst.data[chan], fillData, timeDataSecond.data[chan]]
        )
    # return a new time data object
    # deal with the comment
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
        sampleFreq=sampleFreq,
        startTime=timeDataFirst.startTime,
        stopTime=timeDataSecond.stopTime,
        data=newData,
        comments=comment,
    )
