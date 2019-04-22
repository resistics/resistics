import numpy as np
from typing import Dict, Union

# import from package
from resistics.dataObjects.timeData import TimeData
from resistics.utilities.utilsPrint import errorPrint


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


def intdiv(nom: Union[int, float], div: Union[int, float]) -> int:
    """Return an integer result of division

    The division is expected to be exact and ensures an integer return rather than float.
    Code execution will exit if division is not exact
    
    Parameters
    ----------
    nom : int, float
        Nominator
    div : int, float
        Divisor    

    Returns
    -------
    out : int
        Result of division
    """

    if nom % div == 0:
        return nom // div
    else:
        errorPrint(
            "utilsMath::intdiv",
            "intdiv assumes exits upon having a remainder to make sure errors are not propagated through the code",
            quitRun=True,
        )
        return 0

