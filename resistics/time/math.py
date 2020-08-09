import numpy as np
from typing import Dict, Union

from resistics.time.data import TimeData
from resistics.common.print import errorPrint


def polarityReversal(timeData: TimeData, reversal: Dict[str, bool]) -> TimeData:
    """Multiply the data by -1 (polarity reversal)
    
    Parameters
    ----------
    timeData : TimeData
        timeData to normalise
    reversal : Dict[str, bool]
        Keys are channels and values are boolean flags for reversing

    Returns
    -------
    TimeData
        Normalised time data
    """
    data = {}
    for chan, revbool in reversal.items():
        data[chan] = timeData[chan] * -1 if revbool else timeData[chan]
    comments = timeData.comments + [
        "Polarity reversal with parameters: {}".format(reversal)
    ]
    return TimeData(
        timeData.sampleFreq, timeData.startTime, timeData.stopTime, data, comments,
    )


def scale(timeData: TimeData, scalars: Dict[str, float]) -> TimeData:
    """Scale the data by an arbitrary amount

    If a channel does not appear in scalars, the channel will not be scaled.
    
    Parameters
    ----------
    timeData : TimeData
        timeData to scale
    scalars : Dict[str, float]
        Keys are channels and values are floats to scale with

    Returns
    -------
    TimeData
        Normalised time data
    """
    data = {}
    for chan in timeData:
        if chan in scalars:
            data[chan] = timeData[chan] * scalars[chan]
        else:
            data[chan] = timeData[chan]
    comments = timeData.comments + ["Time data scaled with scalars: {}".format(scalars)]
    return TimeData(
        timeData.sampleFreq, timeData.startTime, timeData.stopTime, data, comments,
    )


def normalise(timeData: TimeData) -> TimeData:
    """Normalise time data
    
    Parameters
    ----------
    timeData : TimeData
        timeData to normalise

    Returns
    -------
    TimeData
        Normalised time data
    """
    data = {}
    for chan in timeData:
        data[chan] = timeData[chan] / np.linalg.norm(timeData[chan])
    comments = timeData.comments + ["Data channels normalised"]
    return TimeData(
        timeData.sampleFreq, timeData.startTime, timeData.stopTime, data, comments,
    )

