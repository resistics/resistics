from copy import deepcopy
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Dict, Union

from resistics.common.base import ResisticsBase
from resistics.common.checks import isElectric
from resistics.common.math import getFrequencyArray
from resistics.common.plot import getViewFonts


class SpectrumData(ResisticsBase):
    """Class for holding spectra data

    Attributes
    ----------
    windowSize : int
        The size of the time window in samples
    dataSize : int
        The number of samples in the frequency spectra
    sampleFreq : float
        The sampling frequency
    startTime : datetime.datetime
        The time of the first sample of the time data which was fourier transformed
    stopTime : datetime.time
        The time of the last sample of the time data which was fourier transformed
    chans : List[str]
        The channels in the data
    numChans : int
        The number of channels
    data : Dict
        The spectra data with channels as keys and arrays as values
    comments : List[str]
        Information about the spectra data as a list of strings

    Methods
    -------
    __init__(kwargs)
        Initialise spectra data
    setData(windowSize, dataSize, sampleFreq, startTime, stopTime, data)
        Set data with parameters
    getComments()
        Get a deepcopy of the comments        
    addComment(comment)
        Add a comment to the dataset
    copy()
        Get a copy of the spectrum data
    view(kwargs)
        View the spectra data 
    printList()
        Class status returned as list of strings          
    """

    def __init__(
        self,
        windowSize: int,
        dataSize: int,
        sampleFreq: float,
        startTime: Union[datetime, str],
        stopTime: Union[datetime, str],
        data,
        comments: Union[List[str], None] = None,
    ) -> None:
        """Initialise and set object parameters

        Parameters
        ----------
        windowSize : int
            The window size in samples of the time data
        dataSize : int
            The spectra size in samples
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
        self.setData(windowSize, dataSize, sampleFreq, startTime, stopTime, data)
        self.comments = comments
        if self.comments is None:
            self.comments = []

    def setData(
        self,
        windowSize: int,
        dataSize: int,
        sampleFreq: float,
        startTime: Union[datetime, str],
        stopTime: Union[datetime, str],
        data,
    ) -> None:
        """Set the object parameters

        Parameters
        ----------
        windowSize : int
            The window size in samples of the original time data
        dataSize : int
            The spectra size in samples
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
        self.windowSize: int = windowSize
        self.sampleFreq: float = sampleFreq
        # start time
        self.startTime: datetime = (
            datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S.%f")
            if isinstance(startTime, str)
            else startTime
        )
        # stop time
        self.stopTime: datetime = (
            datetime.strptime(stopTime, "%Y-%m-%d %H:%M:%S.%f")
            if isinstance(stopTime, str)
            else stopTime
        )
        # other properties
        self.chans: List[str] = sorted(data.keys())
        self.numChans: int = len(self.chans)
        self.data: Dict[str, np.ndarray] = data
        self.dataSize: int = data[self.chans[0]].size

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
    def freqArray(self) -> np.ndarray:
        """Get the frequency array of the spectra data

        Returns
        -------
        freqArray : np.ndarray
            Array of frequencies
        """
        return getFrequencyArray(self.sampleFreq, self.dataSize)

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
        comment : float
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
        return SpectrumData(
            self.windowSize,
            self.dataSize,
            self.sampleFreq,
            self.startTime,
            self.stopTime,
            deepcopy(self.data),
            self.getComments(),
        )

    def view(self, **kwargs) -> Figure:
        """Plot spectra data

        Parameters
        ----------
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts
        chans : List[str], optional
            A list of channels to plot
        label : str, optional
            Label for the plots
        xlim : List, optional
            Limits for the x axis
        color : str, rgba Tuple
            The color for the line plot
        legend : bool
            Boolean flag for adding a legend

        Returns
        -------
        plt.figure
            Matplotlib figure object
        """
        f = self.freqArray
        fig = (
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(20, 2 * self.numChans))
        )
        plotFonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()
        color = kwargs["color"] if "color" in kwargs else None
        # suptitle
        st = fig.suptitle(
            "Spectra data from {} to {}".format(self.startTime, self.stopTime),
            fontsize=plotFonts["suptitle"],
        )
        st.set_y(0.98)
        # now plot the data
        dataChans = kwargs["chans"] if "chans" in kwargs else self.chans
        numPlotChans = len(dataChans)
        for idx, chan in enumerate(dataChans):
            ax = plt.subplot(numPlotChans, 1, idx + 1)
            plt.title("Channel {}".format(chan), fontsize=plotFonts["title"])
            # plot the data
            if "label" in kwargs:
                plt.plot(
                    f, np.absolute(self.data[chan]), color=color, label=kwargs["label"]
                )
            else:
                plt.plot(f, np.absolute(self.data[chan]), color=color)
            # add frequency label
            if idx == numPlotChans - 1:
                plt.xlabel("Frequency [Hz]", fontsize=plotFonts["axisLabel"])
            # x axis options
            xlim = kwargs["xlim"] if "xlim" in kwargs else [f[0], f[-1]]
            plt.xlim(xlim)
            # y axis options
            if isElectric(chan):
                plt.ylabel("[mV/km]", fontsize=plotFonts["axisLabel"])
            else:
                plt.ylabel("[nT]", fontsize=plotFonts["axisLabel"])
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
        textLst.append("Sample rate [s] = {}".format(1.0 / self.sampleFreq))
        textLst.append("Sampling frequency [Hz] = {}".format(self.sampleFreq))
        textLst.append("Nyquist frequency [Hz] = {}".format(self.nyquist))
        textLst.append("Number of time samples = {}".format(self.windowSize))
        textLst.append("Number of frequency samples = {}".format(self.dataSize))
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


class PowerData(ResisticsBase):
    """Class for holding auto/cross power spectra data

    Attributes
    ----------
    windowSize : int
        The size of the time window in samples
    dataSize : int
        The data size of the 
    sampleFreq : float
        The sampling frequency
    powers : List[str]
        The auto / cross powers in the data
    numPowers : int
        The number of auto / cross powers        
    data : Dict
        The cross power data with channels as keys and arrays as values

    Methods
    -------
    __init__(kwargs)
        Initialise power spectra data
    smooth(winLen, winType, inplace)
        Smooth the crosspower data
    interpolate(freqs)
        Interpolate the cross powers to a set of frequencies
    copy()
        Get a copy of the spectrum data
    view(kwargs)
        View the spectra data 
    printList()
        Class status returned as list of strings          
    """

    def __init__(
        self, style: str, sampleFreq: int, data: Dict[str, np.ndarray]
    ) -> None:
        """Initialise power spectra data

        Parameters
        ----------
        style : str
            "auto" or "cross" confirming whether autopower spectra or crosspower spectra
        sampleFreq : float
            The sampling frequency of the original time data
        data : Dict
            Data dictionary with channel pairings (ExHy) as keys and numpy data arrays as values
        """
        self.style: str = style
        self.sampleFreq: float = sampleFreq
        self.data: Dict[str, np.ndarray] = data
        self.powers: List[str] = list(self.data.keys())
        self.numPowers: int = len(self.powers)
        self.dataSize = len(self.data[self.powers[0]])

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
    def freqArray(self) -> np.ndarray:
        """Get the frequency array of the spectra data

        Returns
        -------
        freqArray : np.ndarray
            Array of frequencies
        """
        return getFrequencyArray(self.sampleFreq, self.dataSize)

    def getPower(self, chan1: str, chan2: str, conjugate: bool = False) -> Union[np.ndarray, None]:
        """Get the auto or cross power from the data

        .. note::

            The order matters as getPower(chan1, chan2) will be the complex conjugate of getPower(chan2, chan1)

        Paramters
        ---------
        chan1 : str
            The first channel
        chan2 : str
            The second channel
        conjugate : bool, optional
            Return the complex conjugate of the power data
        
        Returns
        -------
        np.ndarray, None
            Will return the data array if it exists in the data, otherwise None
        """
        power = "{}-{}".format(chan1, chan2)
        if power not in self.powers:
            return None
        if conjugate:
            return np.conjugate(self.data[power])
        return self.data[power]

    def smooth(
        self, winLen: int, winType: str, inplace: bool = False
    ) -> Union[None, ResisticsBase]:
        """Smooth the power data

        Parameters
        ----------
        winLen : int
            The window length to use
        winType : str
            The window type
        inplace : bool, optional
            Whether to do the smoothing in place or return a new PowerData instance

        Returns
        -------
        None, PowerData
            Returns None if inplace is True, otherwise returns a new PowerData instance
        """
        from resistics.common.smooth import smooth1d

        newdata = {}
        for power in self.powers:
            newdata[power] = smooth1d(self.data[power], winLen, winType)
        if inplace:
            self.data = newdata
            return None
        # otherwise return new power data object
        return PowerData(self.style, self.sampleFreq, newdata)

    def interpolate(self, newfreqs: np.ndarray) -> ResisticsBase:
        """Interpolate the power data

        Parameters
        ----------
        newfreqs : np.ndarray
            numpy array of new frequencies to interpolate to

        Returns
        -------
        PowerData
            Returns PowerData with auto / cross power data interpolated to newfreqs
        """
        import scipy.interpolate as interp

        newdata = {}
        freq = self.freqArray
        for power in self.powers:
            interpFunc = interp.interp1d(freq, self.data[power])
            newdata[power] = interpFunc(newfreqs)
        # otherwise return new power data object
        return PowerData(self.style, self.sampleFreq, newdata)
