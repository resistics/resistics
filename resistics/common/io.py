import os
import sys
import itertools
import glob
from datetime import datetime
from typing import List, Tuple

from resistics.common.print import generalPrint, warningPrint, errorPrint


def getDataDirectoryFormats() -> List[str]:
    """Get list of data directory formats   

    Returns
    -------
    out : List[str]
        A list of allowable data directory formats
    """
    return ["meas", "run", "phnx", "lemi"]


def getDirectoryContents(path: str) -> Tuple[List, List]:
    """Get contents of directory

    Includes both files and directories
    
    Parameters
    ----------
    path : str
        Parent directory path

    Returns
    -------
    dirs : list
        List of directories
    files : list
        List of files excluding hidden files
    """
    if not checkDirExistence(path):
        # return empty lists if directory does not exist
        return [], []
    dirList = os.listdir(path)
    dirs = []
    files = []
    for d in dirList:
        if os.path.isdir(os.path.join(path, d)):
            dirs.append(d)
        else:
            files.append(d)
    return dirs, removeHiddenFiles(files)


def getFilesInDirectory(path: str) -> List:
    """Get files in directory

    Excludes hidden files
    
    Parameters
    ----------
    path : str
        Parent directory path

    Returns
    -------
    files : list
        List of files excluding hidden files
    """
    _, files = getDirectoryContents(path)
    return files


def getDirsInDirectory(path: str):
    """Get subdirectories in directory

    Excludes hidden files
    
    Parameters
    ----------
    path : str
        Parent directory path

    Returns
    -------
    dirs : list
        List of subdirectories
    """
    dirs, _ = getDirectoryContents(path)
    return dirs


def removeHiddenFiles(files: List) -> List:
    """Remove hidden files from list of files

    Hidden files are those which begin with a .
    
    Parameters
    ----------
    files : list
        List of files

    Returns
    -------
    files : list
        List of files with hidden files removed
    """
    filesNew = []
    for f in files:
        if f[0] != ".":
            filesNew.append(f)
    return filesNew


def getDataDirsInDirectory(path: str) -> List:
    """Get subdirectories in directory

    This uses known data formats as defined in getDataDirectoryFormats
    
    Parameters
    ----------
    path : str
        Parent directory path

    Returns
    -------
    dirs : list
        List of directories containing time data
    """
    dirs = getDirsInDirectory(path)
    dirsData = []
    formats = getDataDirectoryFormats()
    for d in dirs:
        for f in formats:
            if f in d:
                dirsData.append(d)
    return dirsData


def checkDirExistence(path: str) -> bool:
    """Check if directory exists

    ..todo:: 
        
        Should check that it is actually a directory
    
    Parameters
    ----------
    path : str
        Path to check

    Returns
    -------
    out : bool
        True if directory exists
    """
    if not os.path.exists(path):
        return False
    return True


def makeDir(path: str) -> None:
    """Make directory
    
    Parameters
    ----------
    path : str
        Directory path to make
    """
    os.makedirs(path)


def checkAndMakeDir(path: str) -> None:
    """Check if directory exists and make if not
    
    Parameters
    ----------
    path : str
        Directory path to make
    """
    if not checkDirExistence(path):
        makeDir(path)


def checkFilepath(path: str) -> bool:
    """Check if file exists
    
    TODO: Should check that it is actually a file

    Parameters
    ----------
    path : str
        Filepath to check

    Returns
    -------
    out : bool
        True if file exists
    """
    if not os.path.exists(path):
        generalPrint(
            "utilsio::checkFilepath", "File path {} could not be found.".format(path)
        )
        return False
    return True


def fileFormatSampleFreq(sampleFreq: float) -> str:
    """Provide a consistent way to represent floating numbers in filenames

    Parameters
    ----------
    sampleFreq : float
        The sampling frequency
    
    Returns
    -------
    str
        Float converted for string for the purposes of a filename
    """
    sampleFreqStr: str = "{:.3f}".format(sampleFreq)
    return sampleFreqStr.replace(".", "_")


def lineToKeyAndValue(line: str, delim="=") -> Tuple[str, str]:
    """Helper function to read headers

    Parameters
    ----------
    line : str
        A string representing a header line
    delim : str
        The delimeter separating key and value
    
    Returns
    -------
    key : str
        The header key
    val : str
        The header value
    """
    split = line.split(delim)
    key = split[0].strip()
    val = split[1].strip()
    return key, val
