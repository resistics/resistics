import inspect
from datetime import datetime
from typing import Dict, List, Union, Any

from resistics.project.utils import generalPrint, warningPrint, errorPrint, blockPrint


def projectText(infoStr: str) -> None:
    """General print to terminal

    Parameters
    ----------
    infoStr : str
        The string to print to the console
    """
    generalPrint("{} Info".format(inspect.stack()[1][3]), infoStr)


def projectBlock(textLst: List[str]) -> None:
    """Class information as a list of strings

    Parameters
    ----------
    textLst : list[str]
        List of strings with information
    """
    blockPrint(inspect.stack()[1][3], textLst)


def projectWarning(warnStr: str) -> None:
    """Warning print to terminal

    Parameters
    ----------
    warnStr : str
        The string to print to the console
    """
    warningPrint("{} Warning".format(inspect.stack()[1][3]), warnStr)


def projectError(errorStr: str, quitRun: bool = False) -> None:
    """Error print to terminal and possibly quit

    Parameters
    ----------
    errorStr : str
        The string to print to the console
    quitRun : bool, optional (False)
        If True, the code will exit
    """
    errorPrint("{} Warning".format(inspect.stack()[1][3]), errorStr, quitRun=quitRun)


def checkDateOptions(options: Dict, timeStart: datetime, timeStop: datetime) -> bool:
    """Check to see if data contributes to user specified date range

    Parameters
    ----------
    options : Dict
        Options dictionary with start and stop options specified by user (if specified at all)
    timeStart : datetime
        Start time of data
    timeStop : datetime
        Stop time of data

    Returns
    -------
    bool
        True if data contributes to the date range
    """
    # now check the user provided dates
    if options["start"] and options["start"] > timeStop:
        # this data has nothing to contribute in the optional date range
        return False
    if options["stop"] and options["stop"] < timeStart:
        # this data has nothing to contribute in the optional date range
        return False
    return True
