"""
Module for calculating Fourier spectra of time data (using pyFFTW for speed) and calculation of cross spectral powers
"""

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
            self.dataArray[:] = timeData[c][:]
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


def crosspowers(
    specData: SpectrumData,
    primary: Union[List[str], None] = None,
    secondary: Union[List[str], None] = None,
) -> PowerData:
    """Calculate the crosspower between two spectrum data
    
    Parameters
    ----------
    specData: Spectrum Data
        Crosspowers will be calculated between the channels of this SpectrumData
    primary : List[str], optional
        The channels for SpectrumData that come first in the crosspowers (not conjugated)
    secondary : List[str], optional
        The channels for SpectrumData that come second in the crosspowers (conjugated)

    Returns
    -------
    PowerData
        Cross power data
    """
    if primary is None:
        primary = specData.chans
    if secondary is None:
        secondary = specData.chans

    arrayPrimary, chansPrimary = specData.toArray(primary)
    arraySecondary, chansSecondary = specData.toArray(secondary)
    # conjugate the secondary array
    arraySecondary = np.conjugate(arraySecondary)
    # broadcast to loop in C rather than Python
    crosspowers = arrayPrimary[:, np.newaxis, :] * arraySecondary[np.newaxis, :, :]
    return PowerData(chansPrimary, chansSecondary, crosspowers, specData.sampleFreq)
