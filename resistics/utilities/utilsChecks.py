import inspect
from typing import List, Dict, Any

# import from package
from resistics.utilities.utilsPrint import generalPrint


def parseKeywords(
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
    """

    for w in default:
        if w in keywords:
            default[w] = keywords[w]
    if printkw:
        generalPrint(
            "{}::utilsCheck::parseKeywords".format(inspect.stack()[1][3]), str(default)
        )
    return default


def elecChannelsList() -> List:
    """List of acceptable electric channels

    Returns
    -------
    out : List
        List of acceptable electric channels 
    """

    return ["Ex", "Ey"]


def isElectric(chan: str) -> bool:
    """Check if channel is electric
    
    Parameters
    ----------
    chan : str
        Channel name

    Returns
    -------
    out : bool
        True if channel is electric 
    """

    if chan in elecChannelsList():
        return True
    return False


def magChannelsList() -> List:
    """List of acceptable magnetic channels

    Returns
    -------
    out : List
        List of acceptable magnetic channels 
    """

    return ["Hx", "Hy", "Hz", "Bx", "By", "Bz"]


def isMagnetic(chan: str) -> bool:
    """Check if channel is magnetic

    Parameters
    ----------
    chan : str
        Channel name

    Returns
    -------
    out : bool
        True if channel is magnetic 
    """

    if chan in magChannelsList():
        return True
    return False


def consistentChans(chan: str) -> str:
    """Convert channels to ensure consistency

    Parameters
    ----------
    chan : str
        Channel name

    Returns
    -------
    out : str
        Converted channel name 
    """

    standardChans = ["Hx", "Hy", "Hz", "Ex", "Ey"]
    if chan in standardChans:
        return chan
    if chan == "Bx":
        return "Hx"
    if chan == "By":
        return "Hy"
    if chan == "Bz":
        return "Hz"
    # otherwise return chan
    return chan
