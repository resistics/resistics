import numpy as np
import scipy.signal as signal
from datetime import datetime, timedelta
from typing import Dict, Union

from resistics.time.data import TimeData
from resistics.common.print import generalPrint, arrayToStringInt
from resistics.common.math import intdiv


def lowPass(timeData: TimeData, cutoff: float, order: int = 5) -> TimeData:
    """Lowpass butterworth filter for time data
    
    Parameters
    ----------
    timeData : TimeData
        timeData to filter
    cutoff : float
        Cutoff frequency in Hz
    order : int
        The filter order, by default 5

    Returns
    -------
    TimeData
        Filtered time data
    """
    # create the filter
    normalisedCutoff = 2 * cutoff / timeData.sampleFreq
    b, a = signal.butter(order, normalisedCutoff, btype="lowpass", analog=False)
    # filter
    data = filterData(timeData.data, b, a)
    comments = timeData.comments + [
        "Low pass filter applied with cutoff {} Hz".format(cutoff)
    ]
    return TimeData(
        timeData.sampleFreq, timeData.startTime, timeData.stopTime, data, comments,
    )


def highPass(timeData: TimeData, cutoff: float, order: int = 5) -> TimeData:
    """Highpass butterworth filter for time data
    
    Parameters
    ----------
    timeData : TimeData
        timeData to filter
    cutoff : float
        Cutoff frequency in Hz
    order : int
        The filter order, by default 5     

    Returns
    -------
    TimeData
        Filtered time data
    """
    # create the filter
    normalisedCutoff = 2 * cutoff / timeData.sampleFreq
    b, a = signal.butter(order, normalisedCutoff, btype="highpass", analog=False)
    # filter
    data = filterData(timeData.data, b, a)
    comments = timeData.comments + [
        "High pass filter applied with cutoff {} Hz".format(cutoff)
    ]
    return TimeData(
        timeData.sampleFreq, timeData.startTime, timeData.stopTime, data, comments,
    )


def bandPass(
    timeData: TimeData, cutoffLow: float, cutoffHigh: float, order: int = 5
) -> TimeData:
    """Bandpass butterworth filter for time data
    
    Parameters
    ----------
    timeData : TimeData
        timeData to filter
    cutoffLow : float
        Cutoff frequency in Hz for the low side
    cutoffHigh : float
        Cutoff frequency in Hz for the high side
    order : int
        The filter order, by default 5      

    Returns
    -------
    TimeData
        Filtered time data
    """
    # create the filter
    normalisedCutoffLow = 2 * cutoffLow / timeData.sampleFreq
    normalisedCutoffHigh = 2 * cutoffHigh / timeData.sampleFreq
    b, a = signal.butter(
        order,
        [normalisedCutoffLow, normalisedCutoffHigh],
        btype="bandpass",
        analog=False,
    )
    # filter
    data = filterData(timeData.data, b, a)
    comments = timeData.comments + [
        "Band pass filter applied with cutoffs {} Hz and {} Hz".format(
            cutoffLow, cutoffHigh
        )
    ]
    return TimeData(
        timeData.sampleFreq, timeData.startTime, timeData.stopTime, data, comments,
    )


def filterData(data: Dict[str, np.ndarray], b, a) -> Dict[str, np.ndarray]:
    """Butterworth filter for array data
    
    Parameters
    ----------
    data : Dict
        Dictionary with channel as keys and data as values
    b : array_like
        The numerator coefficient vector of the filter
    a : array_like
        The denominator coefficient vector of the filter. If a[0] is not 1, then both a and b are normalized by a[0]

    Returns
    -------
    data : Dict
        Dictionary with channel as keys and data as values
    """
    filteredData = {}
    for c in data:
        filteredData[c] = signal.filtfilt(b, a, data[c], method="gust", irlen=500)
    return filteredData


def notchFilter(
    timeData: TimeData, notch: float, band: Union[float, None] = None, order: int = 2
) -> TimeData:
    """Bandpass butterworth filter for time data

    Parameters
    ----------
    timeData : TimeData
        timeData to filter
    notch : float
        Frequency to notch filter in Hz
    band : Union[float, None], optional
        The bandwidth around the centerline freqency that you wish to filter, by default 10 % of sampling frequency
    order : int, optional
        The filter order, by default 2

    Returns
    -------
    TimeData
        Filtered time data
    """
    if band is None:
        band = timeData.sampleFreq / 10
    # set parameters and create filter
    low = notch - (band / 2)
    normalisedLow = 2 * low / timeData.sampleFreq
    high = notch + (band / 2)
    normalisedHigh = 2 * high / timeData.sampleFreq
    filter_type = "bessel"
    b, a = signal.iirfilter(
        order,
        [normalisedLow, normalisedHigh],
        btype="bandstop",
        analog=False,
        ftype=filter_type,
    )
    # filter
    data = {}
    for chan in timeData:
        data[chan] = signal.lfilter(b, a, timeData[chan])
    comments = timeData.comments + [
        "Notch filter applied at {} Hz with band {} Hz".format(notch, band)
    ]
    return TimeData(
        timeData.sampleFreq, timeData.startTime, timeData.stopTime, data, comments,
    )


def resample(timeData: TimeData, resampFreq: float) -> TimeData:
    """Resample time data
    
    Resample the data using the polyphase method which does not assume periodicity
    Calculate the upsample and then the downsampling rate and using polyphase filtering, the final sample rate is: 
    (up / down) * original sample rate
    Therefore, to get a sampling frequency of resampFreq, want:
    (resampFreq / sampleFreq) * sampleFreq
    Use the fractions library to get up and down as integers which they are required to be.

    Parameters
    ----------
    timeData : TimeData
        timeData to filter
    resampFreq : float
        The frequency to resample to

    Returns
    -------
    TimeData
        Filtered time data
    """
    from fractions import Fraction

    # get the resampling fraction and limit denominator because this is most probably a float
    frac = Fraction(1.0 / timeData.sampleFreq).limit_denominator()
    frac = Fraction(frac * int(resampFreq))
    frac.limit_denominator()
    # otherwise, normal polyphase filtering
    data = {}
    for chan in timeData:
        data[chan] = signal.resample_poly(
            timeData[chan], frac.numerator, frac.denominator
        )
    # new TimeData parameters
    numSamples = data[timeData.chans[0]].size
    startTime = timeData.startTime
    stopTime = startTime + timedelta(seconds=(1.0 / resampFreq) * (numSamples - 1))
    comments = timeData.comments + [
        "Time data resampled from {:.6f} Hz to {:.6f} Hz".format(
            timeData.sampleFreq, resampFreq
        )
    ]
    return TimeData(resampFreq, startTime, stopTime, data, comments)


def downsample(timeData: TimeData, downsampleFactor: int) -> Dict[str, np.ndarray]:
    """Decimate TimeData 

    A new TimeData instance will be returned. If downsample factor is greater than 13, downsampling will be performed in multiple operations. See scipy decimate for further information.
    
    Parameters
    ----------
    timeData : TimeData
        TimeData instance to downsample
    downsampleFactor : int
        The factor to downsample the data by

    Returns
    -------
    TimeData
        Downsampled TimeData instance
    """
    if downsampleFactor == 1:
        timeData = timeData.copy()
        timeData.addComment("Downsample factor is 1, no downsampling performed")
        return timeData
    # manage the downsampling factors
    if downsampleFactor > 13:
        downsamples = factorise(downsampleFactor)
        generalPrint(
            "downsample",
            "Downsample factor {} greater than 13. Downsampling will be performed in multiple steps of {}".format(
                downsampleFactor, arrayToStringInt(downsamples)
            ),
        )
    else:
        downsamples = [downsampleFactor]
    # downsample
    data = {}
    for idx, factor in enumerate(downsamples):
        for chan in timeData:
            if idx == 0:
                data[chan] = signal.decimate(timeData[chan], factor, zero_phase=True)
            else:
                data[chan] = signal.decimate(data[chan], factor, zero_phase=True)
    # return new TimeData
    sampleFreq = timeData.sampleFreq / downsampleFactor
    startTime = timeData.startTime
    numSamples = data[timeData.chans[0]].size
    stopTime = startTime + timedelta(seconds=(1.0 / sampleFreq) * (numSamples - 1))
    comments = timeData.comments + [
        "Time data decimated from {:.6f} Hz to {:.6f} Hz, new start time {}, new end time {}".format(
            timeData.sampleFreq, sampleFreq, startTime, stopTime,
        )
    ]
    return TimeData(sampleFreq, startTime, stopTime, data, comments)


def factorise(number: int):
    """Factorise a number to avoid too large a downsample factor

    Parameters
    ----------
    number : int
        The number to factorise
    
    Returns
    -------
    List[int]
        The downsampling factors to use

    Notes
    -----
    There's a few pathological cases here that are being ignored. For example, what if the downsample factor is the product of two primes greater than 13.
    """
    import primefac

    factors = list(primefac.primefac(number))
    downsamples = []

    val = 1
    for f in factors:
        test = val * f
        if test > 13:
            downsamples.append(val)
            val = 1
        val = val * f
    # logic: on the last value of f, val*f is tested
    # if this is greater than 13, the previous val is added, which leaves one factor leftover
    # if not greater than 13, then this is not added either
    # so append the last value. the only situation in which this fails is the last factor itself is over 13.
    downsamples.append(val)
    return downsamples
