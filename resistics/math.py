from typing import Union, List, Optional
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
    for ii in range(3, int(math.sqrt(n)) + 1, 2):
        while (n % ii) == 0:
            prime_list.append(ii)
            n = n // ii
    if n > 2:
        prime_list.append(n)
    return prime_list


def pad2(n_samples: int) -> int:
    """Calculate the amount of padding to next power of 2

    Parameters
    ----------
    n_samples : int
        Size of array to be padded

    Returns
    -------
    int
        Amout of padding samples required to increase to next power of 2

    Examples
    --------
    >>> from resistics.math import pad2
    >>> pad2(14)
    2
    """
    import math

    next_power = math.ceil(math.log(n_samples, 2))
    next_size = math.pow(2, int(next_power))
    return int(next_size) - n_samples


def fft(data: np.ndarray, norm: bool = True):
    """Forward real fourier transform

    Parameters
    ----------
    data : np.ndarray
        Time array to be transformed
    norm : bool, optional
        Normalization mode. Default is None, meaning no normalization on the
        forward transforms and scaling by 1/n on the ifft. For norm="ortho",
        both directions are scaled by 1/sqrt(n).

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
        Normalization mode. Default is None, meaning no normalization on the
        forward transforms and scaling by 1/n on the ifft. For norm="ortho",
        both directions are scaled by 1/sqrt(n).

    Returns
    -------
    np.ndarray
        Inverse fourier transformed data
    """
    import numpy.fft as fft

    if not norm:
        return fft.irfft(data, n=nsamples)
    return fft.irfft(data, n=nsamples, norm="ortho")


def get_freqs(fs: float, n_samples: int) -> np.ndarray:
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

    Examples
    --------
    >>> from resistics.math import get_freqs
    >>> get_freqs(10, 6)
    array([0., 1., 2., 3., 4., 5.])
    """
    return np.linspace(0, fs / 2, n_samples)


def get_evaluation_freqs_min(fs: float, f_min: float) -> np.ndarray:
    """
    Calculate evaluation frequencies with mimum allowable frequency

    Highest frequency is nyquist / 4

    Parameters
    ----------
    fs : float
        Sampling frequency
    f_min : float
        Minimum allowable frequency

    Returns
    -------
    np.ndarray
        Array of evaluation frequencies

    Raises
    ------
    ValueError
        If f_min <= 0

    Examples
    --------

    >>> from resistics.math import get_evaluation_freqs_min
    >>> fs = 256
    >>> get_evaluation_freqs_min(fs, 30)
    array([64.      , 45.254834, 32.      ])
    >>> get_evaluation_freqs_min(fs, 128)
    Traceback (most recent call last):
    ...
    ValueError: Minimum frequency 128 must be > 64.0
    """
    f0 = fs / 4

    if f_min <= 0:
        raise ValueError(f"Minimimum frequency {f_min} not > 0")
    if f_min > f0:
        raise ValueError(f"Minimum frequency {f_min} must be > {f0}")

    ii = 1
    evaluation_freqs = []
    while True:
        freq = f0 / np.power(2, (ii - 1.0) / 2.0)
        if freq < f_min:
            break
        evaluation_freqs.append(freq)
        ii += 1
    return np.array(evaluation_freqs)


def get_evaluation_freqs_size(fs: float, n_freq: int) -> np.ndarray:
    """
    Calculate evaluation frequencies with maximum size

    Highest frequency is nyquist/4

    Parameters
    ----------
    fs : float
        Sampling frequency
    n_freq : int
        Number of evaluation frequencies

    Returns
    -------
    np.ndarray
        Array of evaluation frequencies

    Examples
    --------
    >>> from resistics.math import get_evaluation_freqs_size
    >>> fs = 256
    >>> n_freq = 3
    >>> get_evaluation_freqs_size(fs, n_freq)
    array([64.      , 45.254834, 32.      ])
    """
    f0 = fs / 4
    return f0 / np.power(2, (np.arange(1, n_freq + 1) - 1) / 2)


def get_evaluation_freqs(
    fs: float, f_min: Optional[float] = None, n_freq: Optional[int] = None
) -> np.ndarray:
    """
    Get evaluation frequencies either based on size or a minimum frequency

    Parameters
    ----------
    fs : float
        Sampling frequency Hz
    f_min : Optional[float], optional
        Minimum cutoff for evaluation frequencies, by default None
    n_freq : Optional[int], optional
        Number of evaluation frequencies, by default None

    Returns
    -------
    np.ndarray
        Evaluation frequencies array

    Raises
    ------
    ValueError
        ValueError if both f_min and n_freq are None

    Examples
    --------
    >>> from resistics.math import get_evaluation_freqs
    >>> get_evaluation_freqs(256, f_min=30)
    array([64.      , 45.254834, 32.      ])
    >>> get_evaluation_freqs(256, n_freq=3)
    array([64.      , 45.254834, 32.      ])
    """
    if f_min is None and n_freq is None:
        raise ValueError("One of min_freq and n_freq must be passed")
    elif f_min is not None:
        return get_evaluation_freqs_min(fs, f_min)
    else:
        return get_evaluation_freqs_size(fs, n_freq)


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
        """
        Smooth data

        Parameters
        ----------
        length : int
            Smoothing length
        window : str, optional
            Smoothing window to use, by default "hann"

        Raises
        ------
        ValueError
            If the length is even
        """
        import scipy.signal as signal

        if length % 2 == 0:
            raise ValueError("Smoothing length needs to be odd")
        self._length: int = length
        self._window: str = window
        # get the window weights
        self._window_weights = signal.get_window(window, self._length)
        self._convolve_weights = self._window_weights / self._window_weights.sum()
        # calculate offset for recovering data
        self._smoothed_offset = self._length + intdiv((self._length - 1), 2)

    def smooth(self, data: np.ndarray) -> np.ndarray:
        """
        Smooth a 1-D array

        Parameters
        ----------
        data : np.ndarray
            Array to smooth

        Returns
        -------
        np.ndarray
            Smoothed array

        Raises
        ------
        ValueError
            If not a 1-D array
        ValueError
            If window size > array size
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
