import numpy as np
import math
from datetime import datetime, timedelta
from typing import List

from resistics.common.base import ResisticsBase
from resistics.common.print import blockPrint
from resistics.time.data import TimeData
from resistics.config.io import loadConfig


class Windower(ResisticsBase):
    """Class for windowing time data with overlaps 

    Given time data for a decimation level and a reference time, the windower calculates the number of windows and returns them.

    Attributes
    ----------
    timeData : TimeData
        TimeData object to window
    numSamples : int
        The number of samples in each window
    sampleFreq : float
        Sampling frequency of data in Hz
    winSize : int
        Number of samples in a window
    winDuration : float
        Duration of a window in seconds
    winOlap : np.ndarray
        Number of overlap samples
    chans : List[str]
        List of chans in timeData
    refTime : datetime
        Project reference time
    dataTime : datetime
        Time of timeData first sample
    minWindows : int
        Minimum number of windows required
    winOffset : int
        Offset between local and global window indexing
    firstWindowTime: datetime
        Time of the first window in the TimeData
    winSamples : np.ndarray[int]
        Two dimensional, the sample start and end for each window
    winTimes : 
    winActive : np.ndarray[bool]
        Boolean flagging whether window is active or not

    Methods
    -------
    __init__(refTime, timeData, winSize, winOlap)
        Initialise windower
    getWindowActive(iW)

    getGlobalIndex(iW)
        Return the global index (relative to reference time) given local window index
    getData(iWindow)
        Get timeData for local window iWindow
    getDataGlobal(iGlobal)
        Get timeData for glonal index iGlobal
    initialiseWindows()
        Calculate number of windows, local to global index mapping and local window offset from global
    printList()
        Class status returned as list of strings  
    printWindowTimes()
        Print window times  
    printWindowTimeList()
        Window time information returned as List of strings               
    """

    def __init__(
        self, refTime: datetime, timeData: TimeData, winSize: int, winOlap: int
    ) -> None:
        """Initialise the windower
        
        Parameters
        ----------
        refTime : datetime
            The reference time
        timeData : TimeData
            The time series data to window
        winSize : int
            The window size in samples
        winOlap : int
            The overlap size between windows in samples
        """
        self.timeData: TimeData = timeData
        self.numSamples: int = timeData.numSamples
        self.sampleFreq: float = timeData.sampleFreq
        self.winSize: int = winSize
        self.winDuration: float = (winSize - 1) / timeData.sampleFreq
        self.winOlap: int = winOlap
        self.chans: List[str] = timeData.chans
        # refTime and dataTime are already datetime objects
        self.refTime: datetime = refTime
        self.dataTime: datetime = timeData.startTime
        # min window warning setting
        config = loadConfig()
        self.minWindows: int = config["Window"]["minwindows"]
        # initialise
        self.initialiseWindows()
        self.calcWindowTimes()

    def getWindowActive(self, iWindow: int) -> bool:
        """Returns active status of local index

        Parameters
        ----------
        iWindow : int
            Local index of window

        Returns
        -------
        active : bool
            True if active, false if not
        """
        return self.winActive[iWindow]

    def getGlobalIndex(self, iWindow: int) -> int:
        """Returns global index for local index

        Parameters
        ----------
        iWindow : int
            Local index of window

        Returns
        -------
        iGlobal : int
            Global index
        """
        return iWindow + self.winOffset

    def getData(self, iWindow: int) -> TimeData:
        """Returns time window data for local index

        Parameters
        ----------
        iWindow : int
            Local index of window

        Returns
        -------
        windowData : TimeData
            TimeData object with the window data
        """
        winSamples = self.winSamples[iWindow]
        winData = {}
        for c in self.chans:
            winData[c] = self.timeData.data[c][
                winSamples[0] : winSamples[1] + 1
            ]  # add 1 because numpy indexing like this is not inclusive
        globalWindow = self.winTimes[iWindow][0]
        winStartTime = self.winTimes[iWindow][1]
        winStopTime = self.winTimes[iWindow][2]
        return TimeData(
            sampleFreq=self.sampleFreq,
            startTime=winStartTime,
            stopTime=winStopTime,
            data=winData,
            comments=self.timeData.comments
            + ["Local window iW, global window {}".format(globalWindow)],
        )

    def getDataGlobal(self, iGlobal: int) -> TimeData:
        """Returns time window data for global index

        Parameters
        ----------
        iGlobal : int
            Global index of window

        Returns
        -------
        windowData : TimeData
            TimeData object with the window data
        """
        iWindow = iGlobal - self.winOffset
        return self.getData(iWindow)

    def initialiseWindows(self):
        """Calculate all the window information

        For timeData and referenceTime, initialiseWindows calculates the number of windows (after the reference time) and the global indices of the windows relative to the reference time
        Stores the offset between local indices and the global indices
        """
        # have a reference time
        # the first window starts there
        deltaRefStart = self.dataTime - self.refTime
        if deltaRefStart.total_seconds() < 0:
            self.printWarning(
                "Reference time is after start of recording. Stuff may go wrong!"
            )
        # increment of window start times
        # -1 because inclusive of sample at start
        winStartIncrement = 1.0 * (self.winSize - self.winOlap) / self.sampleFreq
        # calculate number of windows started before reference time
        # and then by taking the ceiling, find the global index of the first window in the data
        self.winOffset = int(
            math.ceil(deltaRefStart.total_seconds() / winStartIncrement)
        )
        # calculate start time of first global window
        offsetSeconds = self.winOffset * winStartIncrement
        # calculate the first window time
        self.firstWindowTime = self.refTime + timedelta(seconds=offsetSeconds)
        # calculate first sample
        deltaStart = self.firstWindowTime - self.dataTime
        sampleStart = deltaStart.total_seconds() * self.sampleFreq
        # next calculate number of windows
        # sample start is the first sample
        # window size is window size inclusive of first sample
        winStart = sampleStart
        winEnd = sampleStart + self.winSize - 1
        winStartOff = self.winSize - self.winOlap
        winSamples = []
        while winEnd < self.numSamples:
            winSamples.append([winStart, winEnd])
            winStart = winStart + winStartOff
            winEnd = winStart + self.winSize - 1
        self.numWindows = len(winSamples)
        # warn if number of windows is small
        if self.numWindows < self.minWindows:
            self.printWarning(
                "Number of windows in data is small - consider stopping decimation"
            )
        # save winSamples as numpy list in class
        self.winSamples = np.array(winSamples, dtype=int)
        # set all windows initially to active
        self.winActive = np.ones(shape=(self.numWindows), dtype=bool)

    def calcWindowTimes(self) -> None:
        """Calculate start and stop times for each window"""
        self.winTimes = []
        iW = 0
        for samples in self.winSamples:
            start = samples[0]
            stop = samples[1]
            win = []
            # global index
            win.append(self.winOffset + iW)
            # start time
            deltaStart = timedelta(seconds=start / self.sampleFreq)
            timeStart = self.dataTime + deltaStart
            deltaEnd = timedelta(seconds=stop / self.sampleFreq)
            timeEnd = self.dataTime + deltaEnd
            # samples2end = self.winSize - 1 # need to remove initial sample
            # timeEnd = timeStart + timedelta(seconds=samples2end/self.sampleFreq)
            win.append(timeStart)
            win.append(timeEnd)
            self.winTimes.append(win)
            iW = iW + 1

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """
        textLst = []
        textLst.append("Sample freq. [Hz] = {:f}".format(self.sampleFreq))
        textLst.append("Window size = {:d}".format(self.winSize))
        textLst.append("Overlap size = {:d}".format(self.winOlap))
        textLst.append("Window duration [s] = {:.3f}".format(self.winDuration))
        textLst.append("Reference time {}".format(self.refTime))
        textLst.append("Data start time {}".format(self.dataTime))
        textLst.append(
            "Number of complete windows in data = {:d}".format(self.numWindows)
        )
        if self.numWindows < self.minWindows:
            textLst.append(
                "Number of windows in data is small - consider stopping decimation"
            )
        if self.numWindows > 0:
            textLst.append(
                "Global index of first window from reference time = {}".format(
                    self.winOffset
                )
            )
            textLst.append(
                "First window starts at time {}, sample {:d}".format(
                    self.firstWindowTime, self.winSamples[0, 0]
                )
            )
        return textLst

    def printWindowTimes(self) -> None:
        """Print the times of the windows"""
        blockPrint("Windower::window times", self.printWindowTimeList())

    def printWindowTimeList(self) -> List:
        """Window time information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """
        winTimes = self.winTimes
        winSamples = self.winSamples
        textLst = []
        textLst.append(
            "NOTE: Sample ranges are inclusive, to get number of samples, use: sample end - sample start + 1"
        )
        for win, winS in zip(winTimes, winSamples):
            textLst.append(
                "Global index = {:d}, start time = {}, end time = {}, start sample = {:d}, end sample = {:d}".format(
                    win[0], win[1], win[2], winS[0], winS[1]
                )
            )
        return textLst