from datetime import datetime
import numpy as np
from typing import Union, List, Set


def generalPrint(pre: str, info: str) -> None:
    """Print to terminal

    Parameters
    ----------
    pre : str
        String describing where the information is being output from 
    info : str
        Information string
    """
    print("{} {}: {}".format(datetime.now().strftime("%H:%M:%S"), pre, info))


def breakPrint() -> None:
    """Prints a break in the termainal to make things easier to read"""
    print("---------------------------------------------------")
    print("")
    print("---------------------------------------------------")


def blockPrint(pre: str, text: Union[List, str]) -> None:
    """Prints a block of information to the terminal with separators

    Parameters
    ----------
    pre : str
        String describing where the information is being output from 
    text : str, List
        Information string or a list of strings to print out in the black
    """
    generalPrint(pre, "####################")
    generalPrint(pre, "{} INFO BEGIN".format(pre.upper()))
    generalPrint(pre, "####################")
    if isinstance(text, str):
        generalPrint(pre, text)
    else:
        for t in text:
            generalPrint(pre, t)
    generalPrint(pre, "####################")
    generalPrint(pre, "{} INFO END".format(pre.upper()))
    generalPrint(pre, "####################")


def warningPrint(pre: str, info: str) -> None:
    """Print a warning to the terminal

    Parameters
    ----------
    pre : str
        String describing where the information is being output from 
    info : str
        Information string
    """
    print("{} {}: {}".format(datetime.now().strftime("%H:%M:%S"), pre.upper(), info))


def errorPrint(pre: str, info: str, quitRun: bool = False):
    """Print an error

    Parameters
    ----------
    pre : str
        String describing where the information is being output from 
    info : str
        Information string
    quitRun : bool, optional
        Bool flag for quitting execution of the code due to the error
    """
    print(
        "ERROR: {} {}: {}".format(
            datetime.now().strftime("%H:%M:%S"), pre.upper(), info
        )
    )
    if quitRun:
        print("Exiting...")
        exit()


def breakComment() -> str:
    """Returns a string to show a break in data comments
    
    Returns
    -------
    str
        A separator string
    """
    return "---------------------------------------------------"


def arrayToString(data: np.ndarray, tabs: bool = False, decimals: int = 8) -> str:
    """Convert an array to a string separated by commas

    Parameters
    ----------
    data : np.ndarray
        Data array
    tabs : bool, optional
        Bool flag for using tabs instead of commas 
    decimals : int, optional
        Number of decimal places to print

    Returns
    -------
    out : str
        Output string
    """
    outputStr: str = ""
    sep: str = ", "
    if tabs:
        sep = "\t"

    for d in data:
        outputStr = outputStr + "{num:.{dec}f}{sep}".format(
            num=d, dec=decimals, sep=sep
        )
    outputStr = outputStr.strip()
    outputStr = outputStr.rstrip(",")
    return outputStr


def arrayToStringSci(data: np.ndarray) -> str:
    """Convert an array to a string of scientific numbers separated by commas

    Parameters
    ----------
    data : np.ndarray
        Data array

    Returns
    -------
    out : str
        Output string
    """
    outputStr = ""
    for d in data:
        outputStr = outputStr + "{:.6e}, ".format(d)
    outputStr = outputStr.strip()
    return outputStr.rstrip(",")


def arrayToStringInt(data: np.ndarray) -> str:
    """Convert an array to a string of integers numbers separated by commas

    Parameters
    ----------
    data : np.ndarray
        Data array

    Returns
    -------
    out : str
        Output string
    """
    outputStr = ""
    for d in data:
        if isinstance(d, float):
            d = int(d)
        outputStr = outputStr + "{:d}, ".format(d)
    outputStr = outputStr.strip()
    return outputStr.rstrip(",")


def listToString(lst: List, floatlim: bool = True) -> str:
    """Convert a list to a comma separated string

    Parameters
    ----------
    lst : list
        List

    Returns
    -------
    out : str
        Output string
    """
    outputStr = ""
    for val in lst:
        outputStr = outputStr + "{}, ".format(val)
    outputStr = outputStr.strip()
    return outputStr.rstrip(",")


def list2rangesFormatter(
    start: Union[int, str], end: Union[int, str], step: Union[int, str]
) -> str:
    """A string for showing a range

    Parameters
    ----------
    start : int
        Start point
    end : int
        End point
    step : int
        Step

    Returns
    -------
    out : str
        Output string
    """
    return "{}-{}:{}".format(start, end, step)


def list2ranges(data: Union[List, Set]):
    """Convert a list of numbers to a list of ranges

    Parameters
    ----------
    data : list, set
        List or set of numbers

    Returns
    -------
    out : str
        Output string
    """

    lst = data
    if type(data) is set:
        lst = list(data)
    lst = sorted(lst)
    n = len(lst)
    result = []
    resultVals = []
    scan = 0
    while n - scan > 2:
        step = lst[scan + 1] - lst[scan]
        if lst[scan + 2] - lst[scan + 1] != step:
            result.append(str(lst[scan]))
            resultVals.append([lst[scan], lst[scan], 0])
            scan += 1
            continue

        for j in range(scan + 2, n - 1):
            if lst[j + 1] - lst[j] != step:
                result.append(list2rangesFormatter(lst[scan], lst[j], step))
                resultVals.append([lst[scan], lst[j], step])
                scan = j + 1
                break
        else:
            result.append(list2rangesFormatter(lst[scan], lst[-1], step))
            resultVals.append([lst[scan], lst[-1], step])
            return ",".join(result)

    if n - scan == 1:
        result.append(str(lst[scan]))
        resultVals.append([lst[scan], lst[scan], 0])
    elif n - scan == 2:
        result.append(",".join(map(str, lst[scan:])))
        resultVals.append([lst[scan], lst[scan], 0])
        resultVals.append([lst[scan + 1], lst[scan + 1], 0])

    return ",".join(result)
