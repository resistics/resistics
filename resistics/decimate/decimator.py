import numpy as np
import scipy.signal as signal
from datetime import timedelta
from typing import List

from resistics.common.base import ResisticsBase
from resistics.common.math import intdiv
from resistics.common.print import (
    generalPrint,
    warningPrint,
    blockPrint,
    arrayToStringInt,
)
from resistics.config.io import loadConfig
from resistics.decimate.parameters import DecimationParameters
from resistics.time.data import TimeData
from resistics.time.filter import downsample


class Decimator(ResisticsBase):
    """Decimate time data

    Decimates time data by factors until the minimum number of required samples is reached. When a downsample factor is too large, downsampling is performed in multiple steps to maintain accuracy of result.

    Attributes
    ----------
    timeData : TimeData
        timeData object to decimate
    sampleFreq : float
        Sampling frequency of time data in Hz
    chans : List[str]
        Channels in time data
    numSamples : int
        Number of samples in timeData
    decParams : DecimationParams
        A DecimationParams object holding decimation information
    minSamples : int
        Minimum required samples to decimate
    level : int
        Current decimation level
    maxDownSampleFactor : int
        Max allowable downsampling in one go. Downsampling becomes less accurate at large downsample factors

    Methods
    -------
    __init__(timeData, decParams)
        Initialise Decimator with a TimeData object and DecimationParams object
    incrementLevel()
        Downsample the timeData to the next decimation level
    downsample(downsampleFactor)
        Do the downsampling
    printList()
        Class status returned as list of strings  
    """

    def __init__(self, timeData: TimeData, decParams: DecimationParameters) -> None:
        """Initialise with timeData and decimation parameters
    
        Parameters
        ----------
        timeData : TimeData
            The time data to decimate
        decParams : DecimationParams
            Decimation parameters for performing the decimation
        """
        self.timeData: TimeData = timeData
        self.sampleFreq: float = timeData.sampleFreq * 1.0
        self.chans: List = timeData.chans
        self.numSamples: int = timeData.numSamples
        self.decParams: DecimationParameters = decParams
        config = loadConfig()
        self.minSamples: int = config["Decimation"]["minsamples"]
        self.level: int = -1
        self.maxDownsampleFactor: int = 8

    def incrementLevel(self) -> bool:
        """Downsample to the next decimation level

        Returns
        -------
        out : bool
            True if downsampling completed successfully. False otherwise

        Notes
        -----
        When the downsampling factor is too large, downsampling is performed in multiple steps. Downsampling will become increasingly inaccurate using the scipy routine when factor is too large        
        """
        # increment level, 0 is the first level
        self.level = self.level + 1
        downsampleFactor = self.decParams.getIncrementalFactor(self.level)

        # if downsample factor is greater than maxDownsampleFactor, downsample in multiple steps
        numDownsamples = 1
        downsampleList = [downsampleFactor]
        if downsampleFactor > self.maxDownsampleFactor:
            # this should give an integer
            numDownsamples = intdiv(downsampleFactor, self.maxDownsampleFactor)
            downsampleList = [self.maxDownsampleFactor, numDownsamples]
            # print info
            self.printText(
                "Downsample factor of {:d} greater than max decimation factor {:d}.".format(
                    downsampleFactor, self.maxDownsampleFactor
                )
            )
            self.printText(
                "Downsampling in multiple decimations given by factors: {}".format(
                    arrayToStringInt(downsampleList)
                )
            )

        for iDS in range(0, numDownsamples):
            check = self.downsample(downsampleList[iDS])
            # check outcome of decimation
            if not check:
                return False
        return True

    def downsample(self, downsampleFactor: int) -> bool:
        """Downsample time data

        Parameters
        ----------
        downsampleFactor : int
            Downsampling factor        

        Returns
        -------
        bool
            True if downsampling completed successfully. False otherwise

        Notes
        -----
        When the downsampling causes number of samples to fall below minSamples, downsampling is not performed. The function returns False in this situation      
        """
        # check to see not at max level
        if self.level >= self.decParams.numLevels:
            self.printWarning(
                "Error, number of decimation levels exceeded, returning no data"
            )
            return False
        # if downsample factor is 1, nothing to do
        if downsampleFactor == 1:
            return True

        # downsampling reduces the number of samples by downsample factor
        # if new number of samples is too small, return False
        if self.numSamples / downsampleFactor < self.minSamples:
            self.printWarning(
                "Next decimation level has less than {} samples. Decimation is exiting.\nSet minimum of samples required using decimator.setMinSamples().".format(
                    self.minSamples
                )
            )
            return False

        # do the downsampling and update class vars
        self.timeData = downsample(self.timeData, downsampleFactor)
        self.sampleFreq = self.timeData.sampleFreq
        self.numSamples = self.timeData.numSamples
        return True

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """
        textLst = []
        textLst.append("Current level = {:d}".format(self.level))
        if self.level == -1:
            textLst.append("This is the initial level - no decimation has occured")
        textLst.append("Current sample freq. [Hz] = {:.6f}".format(self.sampleFreq))
        textLst.append("Current sample rate [s] = {:.6f}".format(1.0 / self.sampleFreq))
        textLst.append("Current number of samples = {:d}".format(self.numSamples))
        return textLst
