from typing import Dict, Union
import numpy as np


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
    from resistics.common.print import errorPrint

    if nom % div == 0:
        return nom // div
    else:
        errorPrint(
            "utilsMath::intdiv",
            "intdiv assumes exact division and exits upon having a remainder to make sure errors are not propagated through the code",
            quitRun=True,
        )
        return 0


def getFrequencyArray(fs: float, samples: int) -> np.ndarray:
    """Calculate the frequency array for a real fourier transform

    Frequency array goes from 0Hz to Nyquist. Nyquist = fs/2
    
    Parameters
    ----------
    fs : float
        Sampling frequency
    samples : int
        Number of samples    

    Returns
    -------
    frequencies : np.ndarray
        Array of rfft frequencies
    """
    nyquist: float = 1.0 * fs / 2.0
    return np.linspace(0, nyquist, samples)


def forwardFFT(data: np.ndarray, norm: bool = True):
    """Forward real fourier transform
    
    Parameters
    ----------
    data : np.ndarray
        Time array to be transformed   
    norm : bool, optional
        Normalisation option

    Returns
    -------
    fourierData : np.ndarray
        Fourier transformed data
    """
    import numpy.fft as fft

    if not norm:
        return fft.rfft(data, axis=0)
    return fft.rfft(data, norm="ortho", axis=0)


def inverseFFT(data: np.ndarray, length: int, norm: bool = True):
    """Inverse real fourier transform
    
    Parameters
    ----------
    data : np.ndarray
        Time array to be transformed
    length : int
        Length of output time data (to remove padding)
    norm : bool, optional
        Normalisation option

    Returns
    -------
    timeData : np.ndarray
        Inverse fourier transformed data
    """
    import numpy.fft as fft

    if not norm:
        return fft.irfft(data, n=length)
    return fft.irfft(data, n=length, norm="ortho")


def padNextPower2(size: int) -> int:
    """Calculate the amount of padding to next power of 2
    
    Parameters
    ----------
    size : float
        Size of array to be padded   

    Returns
    -------
    padSize : int
        Amout of extra padding required to increase to next power of 2
    """
    import math

    next2Power = math.ceil(math.log(size, 2))
    next2Size = math.pow(2, int(next2Power))
    return int(next2Size) - size
