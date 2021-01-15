"""
Functions for doing common checks used in resistics
"""
from logging import getLogger
import inspect
from typing import List, Dict, Any

logger = getLogger(__name__)


def parse_keywords(
    default: Dict[str, Any], keywords: Dict[str, Any], printkw: bool = True
):
    """General print to terminal

    Parameters
    ----------
    default : Dict[str, Any]
        Dictionary of default parameters
    keywords : Dict[str, Any]
        Dictionary of optional keywords
    printkw : bool
        Print out the keywords

    Returns
    -------
    Dict[str, Any]
        The dictionary with the appropriate defaults overwritten by keyword arguments
    """
    for kw in default:
        if kw in keywords:
            default[kw] = keywords[kw]
    if printkw:
        logger.debug(f"{default}")
    return default


def electric_chans() -> List[str]:
    """List of acceptable electric channels

    Returns
    -------
    List[str]
        List of acceptable electric channels
    """
    return ["Ex", "Ey", "E1", "E2", "E3", "E4"]


def is_electric(chan: str) -> bool:
    """Check if a channel is electric

    Parameters
    ----------
    chan : str
        Channel name

    Returns
    -------
    bool
        True if channel is electric

    Examples
    --------
    >>> from resistics.common.checks import is_electric
    >>> is_electric("Ex")
    True
    >>> is_electric("Hx")
    False
    """
    if chan in electric_chans():
        return True
    return False


def magnetic_chans() -> List[str]:
    """List of acceptable magnetic channels

    Returns
    -------
    List[str]
        List of acceptable magnetic channels
    """
    return ["Hx", "Hy", "Hz", "Bx", "By", "Bz"]


def is_magnetic(chan: str) -> bool:
    """Check if channel is magnetic

    Parameters
    ----------
    chan : str
        Channel name

    Returns
    -------
    bool
        True if channel is magnetic

    Examples
    --------
    >>> from resistics.common.checks import is_electric
    >>> is_electric("Ex")
    False
    >>> is_electric("Hx")
    True
    """
    if chan in magnetic_chans():
        return True
    return False


def to_resistics_chan(chan: str) -> str:
    """Convert channels to ensure consistency

    Parameters
    ----------
    chan : str
        Channel name

    Returns
    -------
    str
        Converted channel name

    Examples
    --------
    >>> from resistics.common.checks import to_resistics_chan
    >>> to_resistics_chan("Bx")
    'Hx'
    >>> to_resistics_chan("Ex")
    'Ex'
    """
    standard_chans = ["Hx", "Hy", "Hz", "Ex", "Ey"]
    if chan in standard_chans:
        return chan
    if chan == "Bx":
        return "Hx"
    if chan == "By":
        return "Hy"
    if chan == "Bz":
        return "Hz"
    return chan
