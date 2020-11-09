from typing import Dict, Union
import numpy as np


def eps() -> float:
    """Small number

    Returns
    -------
    float
        A small number for quitting robust regression
    """
    return 0.0001


def intdiv(nom: Union[int, float], div: Union[int, float]) -> int:
    """Return an integer result of division

    The division is expected to be exact and ensures an integer return rather than float.
    Code execution will exit if division is not exact

    Parameters
    ----------
    nom : Union[int, float]
        Nominator
    div : Union[int, float]
        Divisor

    Returns
    -------
    int
        Result of division

    Raises
    ------
    ValueError
        Raises a value error if division leaves a remainder
    """
    if nom % div == 0:
        return nom // div
    raise ValueError(f"{nom} divided by {div} leaves a remainder")


def frequency_array(fs: float, nsamples: int) -> np.ndarray:
    """Calculate the frequency array for a real fourier transform

    Frequency array goes from 0Hz to Nyquist. Nyquist = fs/2

    Parameters
    ----------
    fs : float
        Sampling frequency
    nsamples : int
        Number of samples

    Returns
    -------
    frequencies : np.ndarray
        Array of rfft frequencies
    """
    nyquist: float = 1.0 * fs / 2.0
    return np.linspace(0, nyquist, nsamples)


def pad_to_power2(nsamples: int) -> int:
    """Calculate the amount of padding to next power of 2

    Parameters
    ----------
    nsamples : int
        Size of array to be padded

    Returns
    -------
    int
        Amout of padding samples required to increase to next power of 2
    """
    import math

    next_power = math.ceil(math.log(nsamples, 2))
    next_size = math.pow(2, int(next_power))
    return int(next_size) - nsamples


def fft(data: np.ndarray, norm: bool = True):
    """Forward real fourier transform

    Parameters
    ----------
    data : np.ndarray
        Time array to be transformed
    norm : bool, optional
        Normalization mode. Default is None, meaning no normalization on the forward transforms and scaling by 1/n on the ifft. For norm="ortho", both directions are scaled by 1/sqrt(n).

    Returns
    -------
    np.ndarray
        Fourier transformed data
    """
    import numpy.fft as fft

    if not norm:
        return fft.rfft(data, axis=0)
    return fft.rfft(data, norm="ortho", axis=0)


def ifft(data: np.ndarray, nsamples: int, norm: bool = True):
    """Inverse real fourier transform

    Parameters
    ----------
    data : np.ndarray
        Time array to be transformed
    nsamples : int
        Length of output time data (to remove padding)
    norm : bool, optional
        Normalization mode. Default is None, meaning no normalization on the forward transforms and scaling by 1/n on the ifft. For norm="ortho", both directions are scaled by 1/sqrt(n).

    Returns
    -------
    np.ndarray
        Inverse fourier transformed data
    """
    import numpy.fft as fft

    if not norm:
        return fft.irfft(data, n=nsamples)
    return fft.irfft(data, n=nsamples, norm="ortho")
