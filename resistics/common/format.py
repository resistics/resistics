"""
Frequent formatting functions used across the resistics source code
"""
from typing import Union, List, Set, Dict, Any
from datetime import datetime
import numpy as np


def datetime_format(ns: bool = False) -> str:
    """Get the datetime format format for datetime strptime and strftime

    Returns
    -------
    str
        The datetime str format
    """
    if ns:
        return "%Y-%m-%d %H:%M:%S.%f"
    return "%Y-%m-%d %H:%M:%S"


def strformat_fs(fs: float) -> str:
    """Convert sampling frequency into a string for filenames

    Parameters
    ----------
    fs : float
        The sampling frequency

    Returns
    -------
    str
        Sample frequency converted to string for the purposes of a filename

    Examples
    --------
    >>> from resistics.common.format import strformat_fs
    >>> strformat_fs(512.0)
    '512_000000'
    """
    return (f"{fs:.6f}").replace(".", "_")


def array_to_string(
    data: np.ndarray, sep: str = ", ", precision: int = 8, scientific: bool = False
) -> str:
    """Convert an array to a string for logging or printing

    Parameters
    ----------
    data : np.ndarray
        The array
    sep : str, optional
        The separator to use, by default ", "
    precision : int, optional
        Number of decimal places, by default 8. Ignored for integers.
    scientific : bool, optional
        Flag for formatting floats as scientific, by default False

    Returns
    -------
    str
        String representation of array

    Examples
    --------
    >>> import numpy as np
    >>> from resistics.common.format import array_to_string
    >>> data = np.array([1,2,3,4,5])
    >>> array_to_string(data)
    '1, 2, 3, 4, 5'
    >>> data = np.array([1,2,3,4,5], dtype=np.float32)
    >>> array_to_string(data)
    '1.00000000, 2.00000000, 3.00000000, 4.00000000, 5.00000000'
    >>> array_to_string(data, precision=3, scientific=True)
    '1.000e+00, 2.000e+00, 3.000e+00, 4.000e+00, 5.000e+00'
    """
    style: str = "e" if scientific else "f"
    float_formatter = lambda x: f"{x:.{precision}{style}}"
    output_str = np.array2string(
        data, separator=sep, formatter={"float_kind": float_formatter}
    )
    return output_str.lstrip("[").rstrip("]")


def list_to_string(lst: List[Any]) -> str:
    """Convert a list to a comma separated string

    Parameters
    ----------
    lst : List[Any]
        Input list to convert to a string

    Returns
    -------
    str
        Output string

    Examples
    --------
    >>> from resistics.common.format import list_to_string
    >>> list_to_string(["a", "b", "c"])
    'a, b, c'
    >>> list_to_string([1,2,3])
    '1, 2, 3'
    """
    output_str = ""
    for val in lst:
        output_str += f"{val}, "
    return output_str.strip().rstrip(",")


def list_to_ranges(data: Union[List, Set]) -> str:
    """Convert a list of numbers to a list of ranges

    Parameters
    ----------
    data : Union[List, Set]
        List or set of integers

    Returns
    -------
    str
        Formatted output string

    Examples
    --------
    >>> from resistics.common.format import list_to_ranges
    >>> data = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 26, 35, 40, 45]
    >>> list_to_ranges(data)
    '1-4:1,6-12:2,15-24:3,26,35-45:5'
    """
    lst = list(data) if isinstance(data, set) else data
    lst = sorted(lst)
    n = len(lst)
    formatter = lambda start, stop, step: f"{start}-{stop}:{step}"

    result = []
    scan = 0
    while n - scan > 2:
        step = lst[scan + 1] - lst[scan]
        if lst[scan + 2] - lst[scan + 1] != step:
            result.append(str(lst[scan]))
            scan += 1
            continue

        for jj in range(scan + 2, n - 1):
            if lst[jj + 1] - lst[jj] != step:
                result.append(formatter(lst[scan], lst[jj], step))
                scan = jj + 1
                break
        else:
            result.append(formatter(lst[scan], lst[-1], step))
            return ",".join(result)

    if n - scan == 1:
        result.append(str(lst[scan]))
    elif n - scan == 2:
        result.append(",".join(map(str, lst[scan:])))

    return ",".join(result)