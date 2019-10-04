import numpy as np
from typing import Dict, Union

from resistics.time.data import TimeData
from resistics.common.print import errorPrint


def polarityReversal(
    timeData: TimeData, reversal: Dict[str, bool], inplace: bool = True
) -> TimeData:
    """Multiply the data by -1 (polarity reversal)
    
    Parameters
    ----------
    timeData : TimeData
        timeData to normalise
    reversal : Dict[str, bool]
        Keys are channels and values are boolean flags for reversing
    inplace : bool, optional
        Whether to manipulate the data inplace

    Returns
    -------
    TimeData
        Normalised time data
    """
    if not inplace:
        timeData = timeData.copy()
    timeData.data = polarityReversalData(timeData.data, reversal)
    timeData.addComment("Polarity reversal with parameters: {}".format(reversal))
    return timeData


def polarityReversalData(
    data: Dict[str, np.ndarray], reversal: Dict[str, bool]
) -> Dict[str, np.ndarray]:
    """Polarity reverse data or simply multiply by -1
    
    Parameters
    ----------
    data : Dict
        Dictionary with channel as keys and data as values
    reversal : Dict[str, bool]
        Keys are channels and values are boolean flags for reversing
        
    Returns
    -------
    Dict
        Dictionary with channel as keys and normalised data as values
    """
    for c in data:
        if c in reversal and reversal[c]:
            data[c] = data[c] * -1
    return data


def scale(
    timeData: TimeData, scalars: Dict[str, bool], inplace: bool = True
) -> TimeData:
    """Scale the data by an arbitrary amount
    
    Parameters
    ----------
    timeData : TimeData
        timeData to normalise
    scalars : Dict[str, float]
        Keys are channels and values are boolean flags for reversing
    inplace : bool, optional
        Whether to manipulate the data inplace

    Returns
    -------
    TimeData
        Normalised time data
    """
    if not inplace:
        timeData = timeData.copy()
    timeData.data = scaleData(timeData.data, scalars)
    timeData.addComment("Time data scaled with scalars: {}".format(scalars))
    return timeData


def scaleData(
    data: Dict[str, np.ndarray], scalars: Dict[str, bool]
) -> Dict[str, np.ndarray]:
    """Polarity reverse data or simply multiply by -1
    
    Parameters
    ----------
    data : Dict
        Dictionary with channel as keys and data as values
    scalars : Dict[str, float]
        Keys are channels and values are flaots
        
    Returns
    -------
    Dict
        Dictionary with channel as keys and normalised data as values
    """
    for c in data:
        if c in scalars:
            data[c] = data[c] * scalars[c]
    return data
