import numpy as np
import scipy.signal as signal
import pyfftw
from typing import List, Dict

# import from package
from resistics.calculators.calculator import Calculator
from resistics.dataObjects.spectrumData import SpectrumData
from resistics.dataObjects.timeData import TimeData
from resistics.utilities.utilsConfig import loadConfig


class SpectrumCalculator(Calculator):
    """Class for calculating spectra from time data windows

    The class requires the pyfftw and FFTW3 libraries, which allow for fast fourier transforms for spectra calculations

    Attributes
    ----------
    sampleFreq : float
        Sample frequency of the data
    numSamples : int
        The number of samples in each window
    window : bool (true)
        Flag for applying a window function before fourier transform
    windowFunc : np.ndarray
        Window function to apply to time data before fourier transform
    dataArray : np.ndarray
        numpy array to copy data into before performing fourier transform
    ffotObj : pyfftw.FFTW
        fourier transform object

    Methods
    -------
    __init__(sampleFreq, winSamples)
        Initialise with time data sampling frequency and the number of samples in a window
    calcFourierTransform(timeData)
        Perform fourier transform for timeData and return specData object
    printList()
        Class status returned as list of strings        
    """

    def __init__(self, sampleFreq: float, winSamples: int) -> None:
        """Initialise 
    
        Parameters
        ----------
        sampleFreq : float
            Sampling frequency of time data
        winSamples : int
            Number of samples in a window 
        """

        self.sampleFreq: float = sampleFreq
        self.numSamples: int = winSamples
        config = loadConfig()
        self.window: bool = config["Spectra"]["applywindow"]
        self.windowFunc: np.ndarray = signal.get_window(
            config["Spectra"]["windowfunc"], winSamples
        )
        # create an pyfftw plan
        self.dataArray: np.ndarray = pyfftw.empty_aligned(
            self.numSamples, dtype="float64"
        )
        self.fftObj: pyfftw.FFTW = pyfftw.builders.rfft(self.dataArray)

    def calcFourierCoeff(self, timeData: TimeData) -> SpectrumData:
        """Fast fourier transform of timeData

        Parameters
        ----------
        timeData : TimeData
            A TimeData object

        Returns
        -------
        specData : SpectrumData
            A SpectrumData object
        """

        fftData: Dict = {}
        for c in timeData.chans:
            # first copy data into dataArray
            self.dataArray[:] = timeData.data[c][:]
            # no need to pad, these are usually multiples of two
            # remove mean and window if set
            # detrend
            self.dataArray[:] = signal.detrend(self.dataArray, type="linear")
            if self.window:
                self.dataArray[:] = self.dataArray * self.windowFunc
            # use pytfftw here
            fftData[c] = np.array(self.fftObj())
        # calculate frequency array
        dataSize = fftData[timeData.chans[0]].size
        return SpectrumData(
            windowSize=self.numSamples,
            dataSize=dataSize,
            sampleFreq=timeData.sampleFreq,
            startTime=timeData.startTime,
            stopTime=timeData.stopTime,
            data=fftData,
            comments=timeData.comments
            + ["Fourier transform performed using a hamming window"],
        )

    def printList(self) -> List:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """

        textLst = ["Sample frequency = {:f} [Hz]".format(self.sampleFreq)]
        return textLst
