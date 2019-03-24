import os
from datetime import datetime, timedelta
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union

# import from package
from resistics.dataObjects.dataObject import DataObject
from resistics.utilities.utilsChecks import isElectric
from resistics.utilities.utilsPlotter import getViewFonts


class TimeData(DataObject):
    """Class for holding time data

    Attributes
    ----------
    numSamples : int
        The number of samples in the data
    sampleFreq : float
        The sampling frequency
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
    __init__(kwargs)
        Initialise the time data
    setData(windowSize, dataSize, sampleFreq, startTime, stopTime, data)
        Set data with parameters
    getDateArray() : np.ndarray
        A datetime array of the sample times
    getComments()
        Get a deepcopy of the comments
    addComment(comment)
        Add a comment to the dataset
    copy()
        Get a copy of the timeseries data
    view(kwargs)
        View the spectra data 
    printList()
        Class status returned as list of strings          
    """

    def __init__(
        self,
        sampleFreq: float,
        startTime: Union[datetime, str],
        stopTime: Union[datetime, str],
        data,
        comments: Union[str, List[str]] = [],
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
        comments : List[str]
            Dataset comments
        """

        self.setData(sampleFreq, startTime, stopTime, data)
        self.comments = comments
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
        # start time
        self.startTime = (
            datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S.%f")
            if isinstance(startTime, str)
            else startTime
        )
        # stop time
        self.stopTime = (
            datetime.strptime(stopTime, "%Y-%m-%d %H:%M:%S.%f")
            if isinstance(stopTime, str)
            else stopTime
        )
        # other properties
        self.chans = sorted(data.keys())
        self.numChans = len(self.chans)
        self.data = data
        self.numSamples = data[self.chans[0]].size

    def getDateArray(self) -> np.ndarray:
        """Get the date array

        Returns
        -------
        np.ndarray
            The date array of the time samples
        """

        x = np.empty(shape=(self.numSamples), dtype=datetime)
        for i in range(0, self.numSamples):
            x[i] = self.startTime + timedelta(seconds=1.0 * i / self.sampleFreq)
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

    def view(self, **kwargs) -> plt.figure:
        """Plot statistics for evaluation frequency index

        Plots a simple scatter of each statistic with datetime on the xaxis (datetime of the window start dates). Number of subplots is equal to numStaStatPerWindow.

        Parameters
        ----------
        sampleStart : int, optional
            Sample to start plotting from
        sampleStop : int, optional
            Sample to plot to                   
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotFonts : Dict, optional
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
        x = self.getDateArray()
        start = x[sampleStart]
        stop = x[sampleStop]

        # now plot
        fig = (
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(20, 2 * self.numChans))
        )
        plotFonts = kwargs["plotFonts"] if "plotFonts" in kwargs else getViewFonts()
        # suptitle
        st = fig.suptitle(
            "Time data from {} to {}, samples {} to {}".format(
                start, stop, sampleStart, sampleStop
            ),
            fontsize=plotFonts["suptitle"],
        )
        st.set_y(0.98)
        # now plot the data
        dataChans = kwargs["chans"] if "chans" in kwargs else self.chans
        numDataChans = len(dataChans)
        for idx, chan in enumerate(dataChans):
            ax = plt.subplot(numDataChans, 1, idx + 1)
            # title and label for the plot
            plt.title("Channel {}".format(chan), fontsize=plotFonts["title"])
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
                plt.xlabel("Time", fontsize=plotFonts["axisLabel"])
            # set the xlim
            xlim = kwargs["xlim"] if "xlim" in kwargs else [start, stop]
            plt.xlim(xlim)
            # y axis options
            if isElectric(chan):
                plt.ylabel("mV/km", fontsize=plotFonts["axisLabel"])
            else:
                plt.ylabel("nT or mV", fontsize=plotFonts["axisLabel"])
            plt.grid()
            # set tick sizes
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotFonts["axisTicks"])
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
