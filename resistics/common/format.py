from typing import Union, List, Set, Dict
from logging import getLogger
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
    """
    style: str = "e" if scientific else "f"
    float_formatter = lambda x: f"{x:.{precision}{style}}"
    output_str = np.array2string(
        data, separator=sep, formatter={"float_kind": float_formatter}
    )
    return output_str.lstrip("[").rstrip("]")


def list_to_string(lst: List) -> str:
    """Convert a list to a comma separated string

    Parameters
    ----------
    lst : List
        Input list to convert to a string

    Returns
    -------
    str
        Output string
    """
    output_str = ""
    for val in lst:
        output_str += f"{val}, "
    return output_str.strip().rstrip(",")


def list_to_ranges(data: Union[List, Set]) -> str:
    """Convert a list of numbers to a list of ranges

    For example, the list [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 26, 35, 40, 45]
    becomes "1-4:1,6-12:2,15-24:3,26,35-45:5"

    Parameters
    ----------
    data : Union[List, Set]
        List or set of integers

    Returns
    -------
    str
        Formatted output string
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