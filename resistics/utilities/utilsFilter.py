import numpy as np
import scipy.signal as signal
from datetime import datetime, timedelta
from typing import Dict

# import from package
from resistics.dataObjects.timeData import TimeData
from resistics.utilities.utilsPrint import generalPrint, arrayToStringInt
from resistics.utilities.utilsMath import intdiv


def normalise(timeData: TimeData, inplace: bool = True) -> TimeData:
    """Normalise time data
    
    Parameters
    ----------
    timeData : TimeData
        timeData to normalise
    inplace : bool, optional
        Whether to manipulate the data inplace

    Returns
    -------
    TimeData
        Normalised time data
    """

    if not inplace:
        timeData = timeData.copy()
    timeData.data = normaliseData(timeData.data)
    timeData.addComment("Data normalised")
    return timeData


def normaliseData(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalise array data
    
    Normalisation is done by dividing by the result of numpy.norm of the data

    Parameters
    ----------
    data : Dict
        Dictionary with channel as keys and data as values

    Returns
    -------
    Dict
        Dictionary with channel as keys and normalised data as values
    """

    for c in data:
        data[c] = data[c] / np.linalg.norm(data[c])
    return data


def lowPass(timeData: TimeData, cutoff: float, inplace: bool = True) -> TimeData:
    """Lowpass butterworth filter for time data
    
    Parameters
    ----------
    timeData : TimeData
        timeData to filter
    cutoff : float
        Cutoff frequency in Hz
    inplace : bool, optional
        Whether to manipulate the data inplace

    Returns
    -------
    TimeData
        Filtered time data
    """

    if not inplace:
        timeData = timeData.copy()
    timeData.data = lowPassData(timeData.data, timeData.sampleFreq, cutoff)
    timeData.addComment("Low pass filter applied with cutoff {} Hz".format(cutoff))
    return timeData


def lowPassData(
    data: Dict[str, np.ndarray], sampleFreq: float, cutoff: float, order: int = 5
) -> Dict[str, np.ndarray]:
    """Lowpass butterworth filter for array data
    
    Parameters
    ----------
    data : Dict
        Dictionary with channel as keys and data as values
    cutoff : float
        Cutoff frequency in Hz

    Returns
    -------
    data : Dict
        Dictionary with channel as keys and data as values
    """

    # create the filter
    normalisedCutoff = 2.0 * cutoff / sampleFreq
    b, a = signal.butter(order, normalisedCutoff, btype="lowpass", analog=False)
    # filter each channel
    return filterData(data, b, a)


def highPass(timeData: TimeData, cutoff: float, inplace: bool = True) -> TimeData:
    """Highpass butterworth filter for time data
    
    Parameters
    ----------
    timeData : TimeData
        timeData to filter
    cutoff : float
        Cutoff frequency in Hz
    inplace : bool, optional
        Whether to manipulate the data inplace        

    Returns
    -------
    TimeData
        Filtered time data
    """

    if not inplace:
        timeData = timeData.copy()
    timeData.data = highPassData(timeData.data, timeData.sampleFreq, cutoff)
    timeData.addComment("High pass filter applied with cutoff {} Hz".format(cutoff))
    return timeData


def highPassData(
    data: Dict[str, np.ndarray], sampleFreq: float, cutoff: float, order: int = 5
):
    """Highpass butterworth filter for array data
    
    Parameters
    ----------
    data : Dict
        Dictionary with channel as keys and data as values
    cutoff : float
        Cutoff frequency in Hz

    Returns
    -------
    data : Dict
        Dictionary with channel as keys and data as values
    """

    # create the filter
    normalisedCutoff = 2.0 * cutoff / sampleFreq
    b, a = signal.butter(order, normalisedCutoff, btype="highpass", analog=False)
    return filterData(data, b, a)


def bandPass(timeData: TimeData, cutoffLow: float, cutoffHigh: float, inplace: bool = True) -> TimeData:
    """Bandpass butterworth filter for time data
    
    Parameters
    ----------
    timeData : TimeData
        timeData to filter
    cutoff : float
        Cutoff frequency in Hz
    inplace : bool, optional
        Whether to manipulate the data inplace        

    Returns
    -------
    TimeData
        Filtered time data
    """

    if not inplace:
        timeData = timeData.copy()
    timeData.data = bandPassData(
        timeData.data, timeData.sampleFreq, cutoffLow, cutoffHigh
    )
    timeData.addComment(
        "Band pass filter applied with cutoffs {} Hz and {} Hz".format(
            cutoffLow, cutoffHigh
        )
    )
    return timeData


def bandPassData(
    data: Dict[str, np.ndarray],
    sampleFreq: float,
    cutoffLow: float,
    cutoffHigh: float,
    order: int = 5,
):
    """Bandpass butterworth filter for array data
    
    Parameters
    ----------
    data : Dict
        Dictionary with channel as keys and data as values
    cutoff : float
        Cutoff frequency in Hz

    Returns
    -------
    data : Dict
        Dictionary with channel as keys and data as values
    """

    # create the filter
    normalisedCutoffLow = 2.0 * cutoffLow / sampleFreq
    normalisedCutoffHigh = 2.0 * cutoffHigh / sampleFreq
    b, a = signal.butter(
        order,
        [normalisedCutoffLow, normalisedCutoffHigh],
        btype="bandpass",
        analog=False,
    )
    return filterData(data, b, a)


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


def notchFilter(timeData: TimeData, notch: float, inplace: bool = True) -> TimeData:
    """Bandpass butterworth filter for time data
    
    Parameters
    ----------
    timeData : TimeData
        timeData to filter
    notch : float
        Frequency to notch filter in Hz
    inplace : bool, optional
        Whether to manipulate the data inplace        

    Returns
    -------
    TimeData
        Filtered time data
    """

    if not inplace:
        timeData = timeData.copy()
    timeData.data = notchFilterData(
        timeData.data, timeData.sampleFreq, notch, notch / 5.0
    )
    timeData.addComment("Notch filter applied at {} Hz".format(notch))
    return timeData


def notchFilterData(
    data: Dict[str, np.ndarray], sampleFreq: float, notch: float, band: float
) -> Dict[str, np.ndarray]:
    """Notch filter array data

    Parameters
    ----------
    data : Dict
        Dictionary with channel as keys and data as values
    sampleFreq : float
        Sampling frequency in Hz
    notch : float
        Frequency to notch in Hz
    band : float   
        The bandwidth around the centerline freqency that you wish to filter

    Returns
    -------
    data : Dict
        Dictionary with channel as keys and data as values    
    """

    # set parameters
    nyq = sampleFreq / 2.0
    low = notch - band / 2.0
    high = notch + band / 2.0
    low = low / nyq
    high = high / nyq
    # filter
    order = 2
    filter_type = "bessel"
    filteredData = {}
    for c in data:
        b, a = signal.iirfilter(
            order, [low, high], btype="bandstop", analog=False, ftype=filter_type
        )
        filteredData[c] = signal.lfilter(b, a, data[c])
    return filteredData


def resample(timeData: TimeData, resampFreq: float, inplace: bool = True) -> TimeData:
    """Resample time data
    
    Parameters
    ----------
    timeData : TimeData
        timeData to filter
    resampFreq : float
        The frequency to resample to
    inplace : bool, optional
        Whether to manipulate the data inplace        

    Returns
    -------
    TimeData
        Filtered time data
    """

    if not inplace:
        timeData = timeData.copy()
    timeData.data = resampleData(timeData.data, timeData.sampleFreq, resampFreq)
    # update the time info
    timeData.sampleFreq = resampFreq
    timeData.numSamples = timeData.data[timeData.chans[0]].size
    timeData.stopTime = timeData.startTime + timedelta(
        seconds=(1.0 / timeData.sampleFreq) * (timeData.numSamples - 1)
    )
    return timeData


def resampleData(
    data: Dict[str, np.ndarray], sampleFreq: float, sampleFreqNew: float
) -> Dict[str, np.ndarray]:
    """Resample array data
    
    Resample the data using the polyphase method which does not assume periodicity
    Calculate the upsample and then the downsampling rate and using polyphase filtering, the final sample rate is: 
    (up / down) * original sample rate
    Therefore, to get a sampling frequency of sampleFreqNew, want:
    (sampleFreqNew / sampleFreq) * sampleFreq
    Use the fractions library to get up and down as integers which they are required to be.

    Parameters
    ----------
    data : Dict
        Dictionary with channel as keys and data as values
    sampleFreq : float
        The current sampling frequency in Hz
    sampleFreqNew : float
        The sampling frequency in Hz to resample to

    Returns
    -------
    data : Dict
        Dictionary with channel as keys and data as values
    """

    from fractions import Fraction
    frac = Fraction(
        1.0 / sampleFreq
    ).limit_denominator()  # because this is most probably a float
    frac = Fraction(frac * int(sampleFreqNew))
    frac.limit_denominator()


    # otherwise, normal polyphase filtering
    resampleData = {}
    for c in data:
        resampleData[c] = signal.resample_poly(
            data[c], frac.numerator, frac.denominator
        )
    return resampleData


def downsampleData(data: Dict[str, np.ndarray], downsampleFactor: int) -> Dict[str, np.ndarray]:
    """Decimate array data
    
    Parameters
    ----------
    data : Dict
        Dictionary with channel as keys and data as values
    downsampleFactor : int
        The factor to downsample the data by

    Returns
    -------
    data : Dict
        Dictionary with channel as keys and data as values
    """
    
    # if downsampleFactor is 1, nothing to do
    if downsampleFactor == 1:
        return data

    # a downsample factor should not be greater than 13
    # hence factorise downsampleFactors that are greater than this
    if downsampleFactor > 13:
        downsamples = factorise(downsampleFactor)
        generalPrint(
            "Decimation",
            "Downsample factor {} greater than 13. Downsampling will be performed in multiple steps of {}".format(
                downsampleFactor, arrayToStringInt(downsamples)
            ),
        )
    else:
        downsamples = [downsampleFactor]

    # downsample for each factor in downsamples
    for factor in downsamples:
        for c in data:
            data[c] = signal.decimate(data[c], factor, zero_phase=True)
    return data


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

