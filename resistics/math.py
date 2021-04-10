from typing import Union, Tuple, List
import pandas as pd
import numpy as np

from resistics.common import ResisticsProcess


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

    Examples
    --------
    .. doctest::

        >>> from resistics.math import intdiv
        >>> intdiv(12, 3)
        4
    """
    if nom % div == 0:
        return nom // div
    raise ValueError(f"{nom} divided by {div} leaves a remainder")


def prime_factorisation(n: int) -> List[int]:
    """
    Factorise an integer into primes

    Parameters
    ----------
    n : int
        The integer to factorise

    Returns
    -------
    List[int]
        List of factors
    """
    import math

    prime_list = []
    # turn n into odd number
    while (n % 2) == 0:
        prime_list.append(2)
        n = n // 2
    if n == 1:
        return prime_list
    # odd divisors
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while (n % i) == 0:
            prime_list.append(i)
            n = n // i
    if n > 2:
        prime_list.append(n)
    return prime_list


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

    Examples
    --------
    .. doctest::

        >>> from resistics.math import pad_to_power2
        >>> pad_to_power2(14)
        2
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


def frequency_array(fs: float, n_samples: int) -> np.ndarray:
    """Calculate the frequency array for a real fourier transform

    Frequency array goes from 0Hz to Nyquist. Nyquist = fs/2

    Parameters
    ----------
    fs : float
        Sampling frequency
    n_samples : int
        Number of samples

    Returns
    -------
    np.ndarray
        Array of rfft frequencies
    """
    return np.linspace(0, fs / 2, n_samples)


def smooth_length(nsamples: int, proportion: float = 16.0) -> int:
    """Get smoothing length as a proportion of the number of samples in the data

    Parameters
    ----------
    nsamples : int
        The number of samples in the data
    proportion : float, optional
        The proportion of the data that should be in a single smoothing window

    Returns
    -------
    int
        The smoothing window size in samples

    Examples
    --------
    .. doctest::

        >>> from resistics.math import smooth_length
        >>> smooth_length(128, 16)
        9
    """
    length = int(nsamples // proportion)
    if length <= 3:
        return 3
    # ensure odd number
    if length % 2 == 0:
        return length + 1
    return length


class Smoother(ResisticsProcess):
    def __init__(self, length: int, window: str = "hann"):
        """Initialise the smoother with a length and window

        Parameters
        ----------
        smooth_lenght : int
            Smoothing length
        window : str, optional
            Smoothing window, by default "hann"
        """
        import scipy.signal as signal

        if length % 2 == 0:
            raise ValueError("Smoothing length needs to be odd")
        self._length: int = length
        self._window: str = window
        # get the window weights
        if self._window == "flat":
            self._window_weights = np.ones(self._length, "d")
        else:
            self._window_weights = eval("signal." + window + f"({self._length})")
        self._convolve_weights = self._window_weights / self._window_weights.sum()
        # calculate offset for recovering data
        self._smoothed_offset = self._length + intdiv((self._length - 1), 2)

    def smooth(self, data: np.ndarray) -> np.ndarray:
        """Smooth a 1-D array

        Parameters
        ----------
        data : np.ndarray
            Array to smooth

        Returns
        -------
        np.ndarray
            Smoothed array
        """
        if data.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")
        if data.size < self._length:
            raise ValueError("Input vector needs to be bigger than window size.")
        if self._length < 3:
            return data

        padded = np.pad(data, (self._length, self._length), mode="edge")
        smoothed = np.convolve(padded, self._convolve_weights, mode="full")
        return smoothed[self._smoothed_offset : self._smoothed_offset + data.size]
