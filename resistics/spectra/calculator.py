import numpy as np
import scipy.signal as signal
import pyfftw
from typing import List, Dict, Union

from resistics.common.base import ResisticsBase
from resistics.spectra.data import SpectrumData, PowerData
from resistics.time.data import TimeData
from resistics.config.io import loadConfig


class SpectrumCalculator(ResisticsBase):
    """Class for calculating spectra from time data windows

    The class requires the pyfftw and FFTW3 libraries, which allow for fast fourier transforms for spectra calculations

    Attributes
    ----------
    sampleFreq : float
        Sample frequency of the data
    numSamples : int
        The number of samples in each window
    detrend : bool
        Flag for detrending the timedata
    applywindow : bool
        Flag for applying a window to the time data
    window : str
        The name of the window to apply
    windowFunc : np.ndarray
        Window function to apply to time data before fourier transform
    dataArray : np.ndarray
        numpy array to copy data into before performing fourier transform
    ffotObj : pyfftw.FFTW
        fourier transform object

    Methods
    -------
    __init__(sampleFreq, winSamples, config=None)
        Initialise with time data sampling frequency and the number of samples in a window
    calcFourierTransform(timeData)
        Perform fourier transform for timeData and return specData object
    printList()
        Class status returned as list of strings        
    """

    def __init__(
        self, sampleFreq: float, winSamples: int, config: Union[Dict, None] = None
    ) -> None:
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
        if config is None:
            config = loadConfig()
        self.detrend: bool = True
        self.applywindow: bool = config["Spectra"]["applywindow"]
        self.window: str = config["Spectra"]["windowfunc"]
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
            # copy data into dataArray
            self.dataArray[:] = timeData.data[c][:]
            # no need to pad, these are usually multiples of two
            # detrend and apply window function if set
            if self.detrend:
                self.dataArray[:] = signal.detrend(self.dataArray, type="linear")
            if self.applywindow:
                self.dataArray[:] = self.dataArray * self.windowFunc
            # use pytfftw here
            fftData[c] = np.array(self.fftObj())
        # calculate frequency array
        dataSize = fftData[timeData.chans[0]].size

        # comments
        specComments = []
        if self.applywindow:
            specComments.append(
                "Time data multiplied by {} window function".format(self.window)
            )
        specComments.append(
            "Fourier transform performed, time data size = {}, spectra data size = {}".format(
                self.numSamples, dataSize
            )
        )

        return SpectrumData(
            windowSize=self.numSamples,
            dataSize=dataSize,
            sampleFreq=timeData.sampleFreq,
            startTime=timeData.startTime,
            stopTime=timeData.stopTime,
            data=fftData,
            comments=timeData.comments + specComments,
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


def autopower(specData: SpectrumData):
    """Calculate the autopower of spectrum data
    
    Parameters
    ----------
    specData: Spectrum Data
        The SpectrumData object for which to calculate the autopower

    Returns
    -------
    PowerData
        Spectral power data
    """
    chans = specData.chans
    autpowerData = {}
    for chan1 in chans:
        for chan2 in chans:
            crossChan = "{}-{}".format(chan1, chan2)
            conjugateChan = "{}-{}".format(chan2, chan1)
            if (chan1 != chan2) and conjugateChan in autpowerData:
                # can use conjugate symmetry for the autopowers
                autpowerData[crossChan] = np.conjugate(autpowerData[conjugateChan])
            else:
                autpowerData[crossChan] = specData.data[chan1] * np.conjugate(
                    specData.data[chan2]
                )

    return PowerData("auto", specData.sampleFreq, autpowerData)


def crosspower(specData1: SpectrumData, specData2: SpectrumData):
    """Calculate the crosspower between two spectrum data
    
    Parameters
    ----------
    specData1: Spectrum Data
        The first SpectrumData object in the crosspower
    specData2: Spectrum Data
        The second SpectrumData object in the crosspower

    Returns
    -------
    PowerData
        Spectral power data
    """
    from resistics.common.print import errorPrint

    chans1 = specData1.chans
    chans2 = specData2.chans

    if (
        specData1.sampleFreq != specData2.sampleFreq
        or specData1.windowSize != specData2.windowSize
        or specData1.dataSize != specData2.dataSize
    ):
        errorPrint(
            "crosspower",
            "Sample frequency, window size and data size for both SpectrumData need to be matching",
            quitrun=True,
        )

    crosspowerData = {}
    # now need to go through the chans - unable to use conjugate symmetry here
    for chan1 in chans1:
        for chan2 in chans2:
            crossChan = "{}-{}".format(chan1, chan2)
            crosspowerData[crossChan] = specData1.data[chan1] * np.conjugate(
                specData2.data[chan2]
            )

    return PowerData("cross", specData1.sampleFreq, crosspowerData)
