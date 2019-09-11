import numpy as np
import math
import xml.etree.ElementTree as ET
from typing import List, Tuple


def getKnownCalibrationFormats() -> Tuple[List, List]:
    """Return list of supported calibration formats and their extensions

    Returns
    -------
    extensions : List
        List of calibration file extensions
    formats : List
        List of calibration file formats 
    """

    calExt = ["TXT", "TXT", "RSP", "RSPX"]
    calFormats = ["induction", "metronix", "rsp", "rspx"]
    return calExt, calFormats


def getCalName(format, ext: str, sensor: str, serial: int, chopper) -> str:
    """Get the calibration file name
    
    Parameters
    ----------
    format : str
        Calibration format
    ext : str
        Calibration file extension
    sensor : str
        Sensor name
    serial : int
        The sensor serial number
    chopper : bool
        Boolean flag for chopper on or off

    Returns
    -------
    out : str
        Name of calibration file
    """

    if format == "induction":
        return inductionName(ext, sensor, serial, chopper)
    elif format == "metronix":
        return metronixName(ext, sensor, serial, chopper)
    elif format == "rsp":
        return rspName(ext, sensor, serial, chopper)
    elif format == "rspx":
        return rspxName(ext, sensor, serial, chopper)
    else:
        return metronixName(ext, sensor, serial, chopper)


def inductionName(ext: str, sensor: str, serial: int, chopper: bool) -> str:
    """Get internal format induction coil calibration file name
    
    Parameters
    ----------
    ext : str
        Calibration file extension
    sensor : str
        Sensor name
    serial : int
        The sensor serial number
    chopper : bool
        Boolean flag for chopper on or off

    Returns
    -------
    out : str
        Name of calibration file
    """

    return "IC_{}.{}".format(serial, ext)


def metronixName(ext: str, sensor: str, serial: int, chopper: bool) -> str:
    """Get Metronix calibration file name
    
    Parameters
    ----------
    ext : str
        Calibration file extension
    sensor : str
        Sensor name
    serial : int
        The sensor serial number
    chopper : bool
        Boolean flag for chopper on or off

    Returns
    -------
    out : str
        Name of calibration file
    """

    if sensor == "" and not serial > 0:
        return None

    return "{}{}.{}".format(sensor, serial, ext)


def rspName(ext: str, sensor: str, serial: int, chopper: bool) -> List[str]:
    """Get RSP calibration file name
    
    Parameters
    ----------
    ext : str
        Calibration file extension
    sensor : str
        Sensor name
    serial : int
        The sensor serial number
    chopper : bool
        Boolean flag for chopper on or off

    Returns
    -------
    out : List[str]
        Name of calibration files
    """

    if len(sensor) < 5:
        # not possible to get a sensor number
        return None

    board = "HF"
    if chopper:
        board = "LF"
    sensorNum = int(sensor[3:])
    names = []
    names.append("TYPE-{:03d}_{}-ID-{:06d}.{}".format(sensorNum, board, serial, ext))
    names.append("TYPE-{:03d}_BB-ID-{:06d}.{}".format(sensorNum, serial, ext))
    return names


def rspxName(ext: str, sensor: str, serial: int, chopper: bool) -> List[str]:
    """Get RSPX calibration file name
    
    Parameters
    ----------
    ext : str
        Calibration file extension
    sensor : str
        Sensor name
    serial : int
        The sensor serial number
    chopper : bool
        Boolean flag for chopper on or off

    Returns
    -------
    out : List[str]
        Name of calibration files
    """

    if len(sensor) < 6:
        # not possible to get a sensor number
        return None

    board = "HF"
    if chopper:
        board = "LF"
    sensorNum = int(sensor[3:5])
    names = []
    names.append("TYPE-{:03d}_{}-ID-{:06d}.{}".format(sensorNum, board, serial, ext))
    names.append("TYPE-{:03d}_BB-ID-{:06d}.{}".format(sensorNum, serial, ext))
    return names


def defaultCalibration():
    """Default calibration data

    Returns
    -------
    data : np.ndarray
        Data lines converted to a float array
    staticGain : float
        Static gain    
    """

    return [1] * 10, 1
