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

    calExt = ["TXT", "RSP", "RSPX"]
    calFormats = ["metronix", "rsp", "rspx"]
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
    serial : 
    chopper :    

    Returns
    -------
    out : str
        Name of calibration file
    """

    if format == "metronix":
        return metronixName(ext, sensor, serial, chopper)
    elif format == "rsp":
        return rspName(ext, sensor, serial, chopper)
    elif format == "rspx":
        return rspxName(ext, sensor, serial, chopper)
    else:
        return metronixName(ext, sensor, serial, chopper)


def metronixName(ext: str, sensor: str, serial, chopper) -> str:
    """Get Metronix calibration file name
    
    Parameters
    ----------
    ext : str
        Calibration file extension
    sensor : str
        Sensor name
    serial : 
    chopper:    

    Returns
    -------
    out : str
        Name of calibration file
    """

    return "{}{}.{}".format(sensor, serial, ext)


def rspName(ext: str, sensor: str, serial, chopper) -> str:
    """Get RSP calibration file name
    
    Parameters
    ----------
    ext : str
        Calibration file extension
    sensor : str
        Sensor name
    serial : 
    chopper :    

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


def rspxName(ext: str, sensor: str, serial, chopper) -> str:
    """Get RSPX calibration file name
    
    Parameters
    ----------
    ext : str
        Calibration file extension
    sensor : str
        Sensor name
    serial : 
    chopper :    

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


# these are functions for reading calibration files
def readCalFile(
    format: str, filepath: str, sensor: str, serial, chopper
) -> Tuple[np.ndarray, float]:
    """Read data from calibration file
    
    Data is returned with units: F [Hz], Magnitude [mV/nT], Phase [radians]

    Parameters
    ----------
    format : str
        Calibration file extension
    filepath : str
    sensor : str
        Sensor name
    serial : 
    chopper :    

    Returns
    -------
    out : str
        Name of calibration file
    """

    if format == "metronix":
        return metronixData(filepath, chopper)
    elif format == "rsp":
        return rspData(filepath)
    elif format == "rspx":
        return rspxData(filepath)
    else:
        # return a unit response
        return defaultCalibration()


def metronixData(filepath: str, chopper) -> Tuple[np.ndarray, float]:
    """Read data from calibration file
    
    Metronix data is in units: F [Hz], Magnitude [V/nT*Hz], Phase [deg] for both chopper on
    Data is returned with units: F [Hz], Magnitude [mV/nT], Phase [radians]

    Parameters
    ----------
    format : str
        Calibration file extension
    filepath : str
    sensor : str
        Sensor name
    serial : 
    chopper :    

    Returns
    -------
    data : np.ndarray
        Calibration data
    staticGain : float
        Static gain
    """

    # no static gain - already included
    staticGain = 1
    # open file
    f = open(filepath, "r")
    lines = f.readlines()
    numLines = len(lines)
    f.close()
    # variables to save line numbers
    chopperOn = 0
    chopperOff = 0
    # find locations for chopperOn and chopperOff
    for il in range(0, numLines):
        # remove whitespace and new line characters
        lines[il] = lines[il].strip()
        if "Chopper On" in lines[il]:
            chopperOn = il
        if "Chopper Off" in lines[il]:
            chopperOff = il

    # get the part of the file required depending on chopper on or off
    dataLines = []
    dataOn = chopperOff
    if chopper:
        dataOn = chopperOn
    # get the data - starting from the next line
    il = dataOn + 1
    while il < numLines and lines[il] != "":
        # save line then increment
        dataLines.append(lines[il])
        il = il + 1

    # get the data as an array
    data = linesToArray(dataLines)
    # sort and extend
    data = sortCalData(data)
    data = extendCalData(data)
    # unit manipulation
    # change V/(nT*Hz) to mV/nT
    data[:, 1] = data[:, 1] * data[:, 0] * 1000
    # change phase to radians
    data[:, 2] = data[:, 2] * (math.pi / 180)
    return data, staticGain


def rspData(filepath: str) -> Tuple[np.ndarray, float]:
    """Read data from calibration file
    
    RSP data is in units: F [Hz], Magnitude [mv/nT], Phase [deg]
    Data is returned with units: F [Hz], Magnitude [mV/nT], Phase [radians]

    Parameters
    ----------
    filepath : str
        Filepath to calibration file    

    Returns
    -------
    data : np.ndarray
        Calibration data
    staticGain : float
        Static gain
    """

    f = open(filepath, "r")
    lines = f.readlines()
    numLines = len(lines)
    f.close()

    staticGain = 1
    dataOn = 0
    for il in range(0, numLines):
        # remove whitespace and new line characters
        lines[il] = lines[il].strip()
        # find static gain value
        if "StaticGain" in lines[il]:
            staticGain = float(lines[il].split()[1])
        if "FREQUENCY" in lines[il]:
            dataOn = il
    dataLines = []
    il = dataOn + 2
    # get part of file desired
    while il < numLines and lines[il] != "":
        # save line then increment
        dataLines.append(lines[il])
        il = il + 1

    # get the data as an array
    data = linesToArray(dataLines)
    # change phase to radians and apply static gain
    data[:, 1] = data[:, 1] * staticGain
    data[:, 2] = data[:, 2] * (math.pi / 180)
    # sort and extend
    data = sortCalData(data)
    data = extendCalData(data)
    return data, staticGain


def rspxData(filepath: str) -> Tuple[np.ndarray, float]:
    """Read data from calibration file
    
    RSP data is in units: F [Hz], Magnitude [mv/nT], Phase [deg]
    Data is returned with units: F [Hz], Magnitude [mV/nT], Phase [radians]

    Parameters
    ----------
    filepath : str
        Filepath to calibration file   

    Returns
    -------
    data : np.ndarray
        Calibration data
    staticGain : float
        Static gain
    """

    # this is xml format - use EL tree
    tree = ET.parse(filepath)
    root = tree.getroot()
    # static gain
    staticGain = 1
    if root.find("StaticGain") is not None:
        staticGain = float(root.find("StaticGain").text)
    # get the calibration data
    dataList = []
    for resp in root.findall("ResponseData"):
        dataList.append(
            [
                float(resp.get("Frequency")),
                float(resp.get("Magnitude")),
                float(resp.get("Phase")),
            ]
        )
    # now create array
    data = np.array(dataList)
    # change phase to radians and apply static gain
    data[:, 1] = data[:, 1] * staticGain
    data[:, 2] = data[:, 2] * (math.pi / 180)
    # sort and extend
    data = sortCalData(data)
    data = extendCalData(data)
    return data, staticGain


# sort the calData
def sortCalData(data: np.ndarray) -> np.ndarray:
    """Sort calibration data by frequency ascending (low to high)

    Parameters
    ----------
    data : np.ndarray
        Unsorted calibration data   

    Returns
    -------
    data : np.ndarray
        Sorted calibration data
    """

    return data[data[:, 0].argsort()]


# extend the calData - the data should already be sorted
def extendCalData(data: np.ndarray) -> np.ndarray:
    """Extend calibration data by frequency

    Add extra points at the start and end of the calibration data to ensure complete coverage with the time data 

    Parameters
    ----------
    data : np.ndarray
        Calibration data   

    Returns
    -------
    data : np.ndarray
        Extended calibration data
    """

    # add a line at the top (low frequency) extending the calibration information
    data = np.vstack((np.array([0.0000001, data[0, 1], data[0, 2]]), data))
    # add a line at the top (high frequency) extending the calibration information
    data = np.vstack((data, np.array([100000000, data[-1, 1], data[-1, 2]])))
    return data


def linesToArray(dataLines: List) -> np.ndarray:
    """Convert data lines from a file to an array

    Parameters
    ----------
    dataLines : list
        Data lines read in from a file   

    Returns
    -------
    data : np.ndarray
        Data lines converted to a float array
    """

    # data to columns
    numData = len(dataLines)
    for il in range(0, numData):
        dataLines[il] = dataLines[il].split()
    return np.array(dataLines, dtype=float)


# default calibration if none found
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


def unitCalibration() -> np.ndarray:
    """Unit calibration data

    Returns
    -------
    data : np.ndarray
        Unit calibration data  
    """

    unitCal = [[-100000000, 1, 0], [0, 1, 0], [100000000, 1, 0]]
    return np.array(unitCal)
