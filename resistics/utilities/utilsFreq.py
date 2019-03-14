import numpy as np
import numpy.fft as fft
import math


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

    next2Power = math.ceil(math.log(size, 2))
    next2Size = math.pow(2, int(next2Power))
    return int(next2Size) - size
