import os
import glob
import numpy as np
import scipy.interpolate as interp
from typing import Dict, List

# import from package
from resistics.calculators.calculator import Calculator
from resistics.dataObjects.timeData import TimeData
from resistics.utilities.utilsChecks import isMagnetic
from resistics.utilities.utilsConfig import loadConfig
from resistics.utilities.utilsCalibrate import (
    getKnownCalibrationFormats,
    readCalFile,
    getCalName,
    unitCalibration,
)
from resistics.utilities.utilsFreq import (
    forwardFFT,
    inverseFFT,
    getFrequencyArray,
    padNextPower2,
)
from resistics.utilities.utilsPrint import generalPrint, warningPrint, blockPrint


class Calibrator(Calculator):
    """Class for calibrating a time dataset

    Attributes
    ----------
    calExt : list
        Accepted calibration extensions
    calFormats : list
        The calibration file formats of the respective extensions
    calDir : str
        Directory path of calibration files
    calFiles : list
        List of calibration files found in calibration directory
    calTypes :
        Types of the calibration files found in calibration directory
    numCalFiles : int
        Number of found calibration files in calibration directory
    useTheoretical : bool
        Flag to use theoretical calibration function

    Methods
    -------
    __init__(calDirectory)
        Initialise with calibration directory
    setCalDir(calDirectory)
        Set the calibration directory path
    findCalFiles()
        Find calibration files in calibration directory
    calibrate(timeData, sensor, serial, chopper)
        Calibrate time data
    getCalFile(sensor, serial, chopper)
        Get index of matching calibration flile in calFiles list
    calibrateChan(data, fs, calData)
        Calibrate an individual channel
    interpolateCalData(calData, f, spline)
        Interpolate calibration data to same frequencies as channel data
    getTheoreticalCalData(sensor)
        Get theoretical calibration data
    printList()
        Class status returned as list of strings
    """

    def __init__(self, calDirectory: str) -> None:
        """Set the calibration directory and initialise

        Calibrator will automatically find calibration files in the provided directory
    
        Parameters
        ----------
        calDirectory : str
            Path of directory containing calibration files
        """

        self.calExt, self.calFormats = getKnownCalibrationFormats()
        self.calFiles: List = []
        self.calTypes: List = []
        self.calDir: str = calDirectory
        self.findCalFiles()
        # set whether to use theoretical calibration functions
        conf = loadConfig()
        self.useTheoretical: bool = conf["Calibration"]["usetheoretical"]

    def setCalDir(self, calDirectory: str) -> None:
        """Set the calibration directory and find files

        Calibrator will automatically find calibration files in the provided directory
    
        Parameters
        ----------
        calDirectory : str
            Path of directory containing calibration files
        """

        self.calDir = calDirectory
        self.findCalFiles()

    def findCalFiles(self) -> None:
        """Find calibration files in calibration directory"""

        self.calFiles.clear()
        self.calTypes.clear()
        for cE, cF in zip(self.calExt, self.calFormats):
            # get all files of format cE
            tmp1 = glob.glob(os.path.join(self.calDir, "*.{}".format(cE)))
            tmp2 = [cF] * len(tmp1)
            self.calFiles = self.calFiles + tmp1
            self.calTypes = self.calTypes + tmp2
        self.numCalFiles = len(self.calFiles)

    def calibrate(
        self,
        timeData: TimeData,
        sensor: Dict[str, str],
        serial: Dict[str, str],
        chopper: Dict[str, str],
    ) -> TimeData:
        """Calibrate time data

        For each channel in timeData, searches for a matching calibration file based on sensor type, serial number and chopper. If a calibration file is found, the channel is calibrated using the data in the file. If useTheoretical is False and no file is found, the data is not calibrated

        todo:
        If no calibration file is found and the channel is a magnetic data channel, a theoretical function can be used
    
        Parameters
        ----------
        timeData : TimeData
            TimeData object
        sensor : Dict
            Dictionary of sensor information with channels as the key and sensor as the value
        serial :
            Dictionary of serial information with channels as the key and sensor as the value
        chopper :
            Dictionary of chopper information with channels as the key and sensor as the value

        Returns
        -------
        timeData : TimeData
            Calibration TimeData object
        """

        # iterate over data
        for chan in timeData.chans:
            # output some info
            self.printText("Calibrating channel {}".format(chan))
            # try and find calibration file
            calIndex = self.getCalFile(sensor[chan], serial[chan], chopper[chan])
            if calIndex < 0:
                # no file found
                if self.useTheoretical and isMagnetic(chan):
                    # use theoretical
                    calData = self.getTheoreticalCalData(sensor[chan])
                    timeData.data[chan] = self.calibrateChan(
                        timeData.data[chan], timeData.sampleFreq, calData
                    )
                    timeData.addComment(
                        "Channel {} calibrated with theoretical calibration function".format(
                            chan
                        )
                    )
                    continue
                else:
                    self.printText(
                        "No Calibration data found - channel will not be calibrated"
                    )
                    timeData.addComment("Channel {} not calibrated".format(chan))
                    continue  # nothing to do
            # else file found
            # no need to separately apply static gain, already included in cal data
            calData, staticGain = readCalFile(
                self.calTypes[calIndex],
                self.calFiles[calIndex],
                sensor[chan],
                serial[chan],
                chopper[chan],
            )
            self.printText(
                "Calibration file found for sensor {}, serial number {}, chopper {}: {}".format(
                    sensor[chan], serial[chan], chopper[chan], self.calFiles[calIndex]
                )
            )
            self.printText(
                "Static gain correction of {} applied to calibration data".format(
                    staticGain
                )
            )
            self.printText("Format: {}".format(self.calTypes[calIndex]))
            timeData.data[chan] = self.calibrateChan(
                timeData.data[chan], timeData.sampleFreq, calData
            )
            timeData.addComment(
                "Channel {} calibrated with calibration data from file {}".format(
                    chan, self.calFiles[calIndex]
                )
            )
        # return timeData at the end
        return timeData

    def getCalFile(self, sensor, serial, chopper) -> int:
        """Get calibration file for a sensor, serial and chopper combination
    
        Parameters
        ----------
        sensor : serial
            Channel data
        serial : float
            Sampling frequency Hz
        chopper : np.ndarray
            Calibration data

        Returns
        -------
        out : int
            Index of file in calFiles list or -1 if file not found
        """

        index = -1
        for calE, calF in zip(self.calExt, self.calFormats):
            # get the name for this format
            calName = getCalName(calF, calE, sensor, serial, chopper)
            # search to find a calibration file with that name
            # take the first encountered
            for iF in range(0, self.numCalFiles):
                if calName in self.calFiles[iF]:
                    index = iF
            # break out of calfile found
            if index != -1:
                break
        # return the index to the calibration file
        return index

    def calibrateChan(
        self, data: np.ndarray, sampleFreq: float, calData: np.ndarray
    ) -> np.ndarray:
        """Calibrate a channel

        Perform fourier transform of channel data, deconvolve (division) sensor impulse response and inverse fourier transform.
    
        Parameters
        ----------
        data : np.ndarray
            Channel data
        sampleFreq : float
            Sampling frequency Hz
        calData : np.ndarray
            Calibration data

        Returns
        -------
        out : np.ndarray
            Calibrated channel data
        """

        # do the forward transform
        dataSize = data.size
        # pad end of array
        data = np.pad(data, (0, padNextPower2(dataSize)), "constant")
        fftData = forwardFFT(data)
        f = getFrequencyArray(sampleFreq, fftData.size)
        # do calibration in frequency domain
        # interpolate calibration info to frequencies in data
        transferFunc = self.interpolateCalData(calData, f)
        # recall, zero element excluded, so start from 1
        # fft zero element should really be zero because average is removed from data
        fftData[1:] = fftData[1:] / transferFunc
        # return the non padded part of the array
        return inverseFFT(fftData, data.size)[:dataSize]

    def interpolateCalData(
        self, calData: np.ndarray, f: np.ndarray, spline: bool = True
    ) -> np.ndarray:
        """Interpolation calibration data on to frequency points
    
        Parameters
        ----------
        calData : np.ndarray
            Calibration data
        f : np.ndarray
            Frequency array where calibration data is defined
        spline : bool (True)
            Boolean flag to use spline interpolation

        Returns
        -------
        out : np.ndarray
            Calibration data interpolated to the frequency array f
        """

        if spline:
            # spline interpolation
            interpFuncMag = interp.InterpolatedUnivariateSpline(
                calData[:, 0], calData[:, 1]
            )
            interpFuncPhase = interp.InterpolatedUnivariateSpline(
                calData[:, 0], calData[:, 2]
            )
        else:
            # linear interpolate
            interpFuncMag = interp.interp1d(calData[:, 0], calData[:, 1])
            interpFuncPhase = interp.interp1d(calData[:, 0], calData[:, 2])

        return interpFuncMag(f[1:]) * np.exp(1j * interpFuncPhase(f[1:]))

    def getTheoreticalCalData(self, sensor: str) -> np.ndarray:
        """Get theoretical calibration data for magnetic channels
    
        Parameters
        ----------
        sensor : str
            Sensor type

        Returns
        -------
        out : np.ndarray
            Theoretical calibration information
        """

        # should use the sensor to figure out what calibration func
        if "mfs06" in sensor or "MFS06" in sensor:
            return unitCalibration()

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """

        textLst = []
        textLst.append("Known extensions and calibration formats")
        textLst.append("Extensions:")
        textLst.append(", ".join(self.calExt))
        textLst.append("Associated formats")
        textLst.append(", ".join(self.calFormats))
        # now print the actual calibration files
        textLst.append("Calibration files found:")
        if len(self.calFiles) == 0:
            textLst.append("\t\tNo calibration files found")
        else:
            for cE, cT in zip(self.calFiles, self.calTypes):
                textLst.append("{}\t\t{}".format(cT, cE))
        return textLst
