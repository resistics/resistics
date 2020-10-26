"""
Classes for storing time data
"""
from datetime import datetime, timedelta
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Union, Dict

from resistics.common.base import ResisticsBase
from resistics.common.checks import isElectric
from resistics.common.plot import getViewFonts


class TimeData(ResisticsBase):
    """Class for holding time data

    Attributes
    ----------
    numSamples : int
        The number of samples in the data
    sampleFreq : float
        The sampling frequency
    period : float
        The sampling period
    nyquist : float
        The nyquist frequency
    startTime : datetime.datetime
        The time of the first sample
    stopTime : datetime.datetime
        The time of the last sample 
    chans : List[str]
        The channels in the data
    numChans : int
        The number of channels
    data : Dict
        The time data with channels as keys and arrays as values
    comments : List[str]
        Information about the time data as a list of strings

    Methods
    -------
    __init__(sampleFreq, startTime, stopTime, data)
        Initialise the time data
    setData(windowSize, dataSize, sampleFreq, startTime, stopTime, data)
        Set data with parameters
    __getitem__(chan)
        Get time data for channel
    getChannel(chan)
        Get time data for a channel
    __setitem__(chan)
        Set time data for a channel
    setChannel(chan, chanData)
        Set time data for a channel
    getComments()
        Get a deepcopy of the comments
    addComment(comment)
        Add a comment to the dataset
    copy()
        Get a copy of the timeseries data
    view(kwargs)
        View the time data 
    printList()
        Class status returned as list of strings          
    """

    def __init__(
        self,
        sampleFreq: float,
        startTime: Union[datetime, str],
        stopTime: Union[datetime, str],
        data: Dict[str, np.ndarray],
        comments: Union[str, List[str], None] = None,
    ) -> None:
        """Initialise and set object parameters

        Parameters
        ----------
        sampleFreq : float
            The sampling frequency in Hz
        startTime : datetime, str
            The startTime of the window
        stopTime : datetime, str
            The stopTime of the window 
        data : Dict
            The data dictionary with keys as channels and values as spectra data
        comments : str, List[str], None
            Dataset comments
        """
        self.setData(sampleFreq, startTime, stopTime, data)
        self.comments = comments
        if self.comments is None:
            self.comments = []
        if isinstance(self.comments, str):
            self.comments = [self.comments]

    def setData(
        self,
        sampleFreq: float,
        startTime: Union[datetime, str],
        stopTime: Union[datetime, str],
        data,
    ):
        """Set the object parameters

        Parameters
        ----------
        sampleFreq : float
            The sampling frequency in Hz
        startTime : datetime, str
            The startTime of the window
        stopTime : datetime, str
            The stopTime of the window 
        data : Dict
            The data dictionary with keys as channels and values as spectra data
        """
        self.sampleFreq = sampleFreq
        # times
        self.start = startTime
        self.stop = stopTime
        # other properties
        self.chans = sorted(data.keys())
        self.data = data
        self.numSamples = data[self.chans[0]].size

    def __getitem__(self, chan: str) -> np.ndarray:
        """Get channel time data

        Parameters
        ----------
        chan : str
            The channel to get data for
        
        Returns
        -------
        np.ndarray
            The channel data
        """
        return self.getChannel(chan)

    def getChannel(self, chan: str) -> np.ndarray:
        """Get the time data for a channel

        Parameters
        ----------
        chan : str
            The channel for which to get the time data
        
        Returns
        -------
        np.ndarray
            The time data for the channel
        """
        return self.data[chan]

    def __setitem__(self, chan: str, chanData: np.ndarray) -> None:
        """Set channel time data
        
        Parameters
        ----------
        chan : str
            The channel to set the data for
        chanData : np.ndarray
            The new channel data
        """
        self.setChannel(chan, chanData)

    def setChannel(self, chan: str, chanData: np.ndarray) -> None:
        """Set channel time data
        
        Parameters
        ----------
        chan : str
            The channel to set the data for
        chanData : np.ndarray
            The new channel data
        """
        self.data[chan] = chanData
    
    def __iter__(self):
        """Return the channel iterator
        
        Returns
        -------
        list_iterator
            An iterator for the channels
        """
        return iter(self.chans)

    @property
    def startTime(self) -> datetime:
        """Returns the number of channels
        
        Returns
        -------
        datetime
            The start time of the spectra data window
        """
        if isinstance(self.start, str):
            return datetime.strptime(self.start, "%Y-%m-%d %H:%M:%S.%f")
        return self.start

    @property
    def stopTime(self) -> datetime:
        """Returns the number of channels
        
        Returns
        -------
        datetime
            The stop time of the spectra data window
        """
        if isinstance(self.stop, str):
            return datetime.strptime(self.stop, "%Y-%m-%d %H:%M:%S.%f")
        return self.stop

    @property
    def duration(self) -> float:
        """Duration of the time window in seconds
        
        Returns
        -------
        float
            The duration in seconds
        """
        return (self.stopTime - self.startTime.total_seconds()) 

    @property
    def numChans(self) -> int:
        """Returns the number of channels
        
        Returns
        -------
        int
            The number of channels in spectra data
        """
        return len(self.chans)

    @property
    def period(self) -> float:
        """The sampling period in seconds

        Returns
        -------
        period : float
            The sampling period in seconds
        """
        return 1.0 / self.sampleFreq

    @property
    def nyquist(self) -> float:
        """Get the nyquist frequency of the spectra data

        Returns
        -------
        nyquist : float
            The nyquist frequency in Hz
        """
        return self.sampleFreq / 2.0

    @property
    def timeArray(self) -> np.ndarray:
        """Get the datetime array

        Returns
        -------
        np.ndarray
            The datetime array of the time samples
        """
        x = np.empty(shape=(self.numSamples), dtype=datetime)
        for ii in range(0, self.numSamples):
            x[ii] = self.startTime + timedelta(seconds=1.0 * ii / self.sampleFreq)
        return x

    def getComments(self) -> List[str]:
        """Get a deepcopy of the comments
        
        Returns
        -------
        List[str]
            Dataset comments as a list of strings
        """
        return deepcopy(self.comments)

    def addComment(self, comment: str) -> None:
        """Add a new comment

        Parameters
        ----------
        comment : str
            A new comment
        """
        self.comments.append(comment)

    def copy(self):
        """Get a copy of the time data object

        Returns
        -------
        TimeData
            A copy of the time data object
        """
        return TimeData(
            self.sampleFreq,
            self.startTime,
            self.stopTime,
            deepcopy(self.data),
            self.getComments(),
        )

    def view(self, **kwargs) -> Figure:
        """View timeseries data as a line plot

        Parameters
        ----------
        sampleStart : int, optional
            Sample to start plotting from
        sampleStop : int, optional
            Sample to plot to                   
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts
        chans : List[str]
            Channels to plot
        label : str, optional
            Label for the plots
        xlim : List, optional
            Limits for the x axis
        legened : bool
            Boolean flag for adding a legend
        
        Returns
        -------
        plt.figure
            Matplotlib figure object
        """
        # the number of samples to plot
        sampleStart = 0
        sampleStop = 4096
        if "sampleStart" in kwargs:
            sampleStart = kwargs["sampleStart"]
        if "sampleStop" in kwargs:
            sampleStop = kwargs["sampleStop"]
        if sampleStop >= self.numSamples:
            sampleStop = self.numSamples - 1
        # get the x axis ready
        x = self.timeArray
        start = x[sampleStart]
        stop = x[sampleStop]

        # now plot
        fig = (
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(20, 2 * self.numChans))
        )
        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()
        # suptitle
        st = fig.suptitle(
            "Time data from {} to {}, samples {} to {}".format(
                start, stop, sampleStart, sampleStop
            ),
            fontsize=plotfonts["suptitle"],
        )
        st.set_y(0.98)
        # now plot the data
        dataChans = kwargs["chans"] if "chans" in kwargs else self.chans
        numDataChans = len(dataChans)
        for idx, chan in enumerate(dataChans):
            ax = plt.subplot(numDataChans, 1, idx + 1)
            plt.title("Channel {}".format(chan), fontsize=plotfonts["title"])

            # check if channel exists in data and if not, leave empty plot so it's clear
            if chan not in self.data:
                continue

            # label for plot
            lab = (
                kwargs["label"]
                if "label" in kwargs
                else "{}: {} to {}".format(chan, start, stop)
            )
            # plot data
            plt.plot(
                x[sampleStart : sampleStop + 1],
                self.data[chan][sampleStart : sampleStop + 1],
                label=lab,
            )
            # add time label
            if idx == numDataChans - 1:
                plt.xlabel("Time", fontsize=plotfonts["axisLabel"])
            # set the xlim
            xlim = kwargs["xlim"] if "xlim" in kwargs else [start, stop]
            plt.xlim(xlim)
            # y axis options
            if isElectric(chan):
                plt.ylabel("mV/km", fontsize=plotfonts["axisLabel"])
            else:
                plt.ylabel("nT or mV", fontsize=plotfonts["axisLabel"])
            plt.grid(True)
            # set tick sizes
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])
            # legend
            if "legend" in kwargs and kwargs["legend"]:
                plt.legend(loc=4)

        # show if the figure is not in keywords
        if "fig" not in kwargs:
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])
            plt.show()

        return fig

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """
        textLst = []
        textLst.append("Sampling frequency [Hz] = {}".format(self.sampleFreq))
        textLst.append("Sample rate [s] = {}".format(1.0 / self.sampleFreq))
        textLst.append("Number of samples = {}".format(self.numSamples))
        textLst.append("Number of channels = {}".format(self.numChans))
        textLst.append("Channels = {}".format(self.chans))
        textLst.append("Start time = {}".format(self.startTime))
        textLst.append("Stop time = {}".format(self.stopTime))
        if len(self.comments) == 0:
            textLst.append("No comments")
        else:
            textLst.append("Comments")
            for comment in self.comments:
                textLst.append("\t{}".format(comment))
        return textLst
