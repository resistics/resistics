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


def getCalName(format, ext: str, sensor: str, serial, chopper) -> str:
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


def inductionName(ext: str, sensor: str, serial, chopper: bool) -> str:
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


def metronixName(ext: str, sensor: str, serial, chopper: bool) -> str:
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

    return "{}{}.{}".format(sensor, serial, ext)


def rspName(ext: str, sensor: str, serial, chopper: bool) -> str:
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
    out : str
        Name of calibration file
    """

    board = "HF"
    if chopper:
        board = "LF"
    sensorNum = int(sensor[3:5])
    return "Metronix_Coil-----TYPE-{:03d}_{}-ID-{:06d}.{}".format(
        sensorNum, board, serial, ext
    )


def rspxName(ext: str, sensor: str, serial, chopper: bool) -> str:
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
    out : str
        Name of calibration file
    """

    board = "HF"
    if chopper:
        board = "LF"
    sensorNum = int(sensor[3:5])
    return "Metronix_Coil-----TYPE-{:03d}_{}-ID-{:06d}.{}".format(
        sensorNum, board, serial, ext
    )


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
