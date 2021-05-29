from typing import Optional
import numpy as np


def get_eval_freqs_min(fs: float, f_min: float) -> np.ndarray:
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

    >>> from resistics.math import get_eval_freqs_min
    >>> fs = 256
    >>> get_eval_freqs_min(fs, 30)
    array([64.      , 45.254834, 32.      ])
    >>> get_eval_freqs_min(fs, 128)
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
    eval_freqs = []
    while True:
        freq = f0 / np.power(2, (ii - 1.0) / 2.0)
        if freq < f_min:
            break
        eval_freqs.append(freq)
        ii += 1
    return np.array(eval_freqs)


def get_eval_freqs_size(fs: float, n_freqs: int) -> np.ndarray:
    """
    Calculate evaluation frequencies with maximum size

    Highest frequency is nyquist/4

    Parameters
    ----------
    fs : float
        Sampling frequency
    n_freqs : int
        Number of evaluation frequencies

    Returns
    -------
    np.ndarray
        Array of evaluation frequencies

    Examples
    --------
    >>> from resistics.math import get_eval_freqs_size
    >>> fs = 256
    >>> n_freqs = 3
    >>> get_eval_freqs_size(fs, n_freqs)
    array([64.      , 45.254834, 32.      ])
    """
    f0 = fs / 4
    return f0 / np.power(2, (np.arange(1, n_freqs + 1) - 1) / 2)


def get_eval_freqs(
    fs: float, f_min: Optional[float] = None, n_freqs: Optional[int] = None
) -> np.ndarray:
    """
    Get evaluation frequencies either based on size or a minimum frequency

    Parameters
    ----------
    fs : float
        Sampling frequency Hz
    f_min : Optional[float], optional
        Minimum cutoff for evaluation frequencies, by default None
    n_freqs : Optional[int], optional
        Number of evaluation frequencies, by default None

    Returns
    -------
    np.ndarray
        Evaluation frequencies array

    Raises
    ------
    ValueError
        ValueError if both f_min and n_freqs are None

    Examples
    --------
    >>> from resistics.math import get_eval_freqs
    >>> get_eval_freqs(256, f_min=30)
    array([64.      , 45.254834, 32.      ])
    >>> get_eval_freqs(256, n_freqs=3)
    array([64.      , 45.254834, 32.      ])
    """
    if f_min is None and n_freqs is None:
        raise ValueError("One of f_min and n_freqs must be passed")
    elif f_min is not None:
        return get_eval_freqs_min(fs, f_min)
    else:
        return get_eval_freqs_size(fs, n_freqs)
