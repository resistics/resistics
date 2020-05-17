"""
Data storage classes for spectral data including raw Fourier data and cross power data. Methods for combining data classes
"""
from copy import deepcopy
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Dict, Union, Tuple

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
    period : float
        The sampling period
    nyquist : float
        The nyquist frequency
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
    __getitem__(channel)
        The getitem accessor for getting spectral data for a channel
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
        # times
        self.start = startTime
        self.stop = stopTime
        # other properties
        self.chans: List[str] = sorted(data.keys())
        self.data: Dict[str, np.ndarray] = data
        self.dataSize: int = data[self.chans[0]].size

    def __getitem__(self, chan: str):
        """Get channel spectra data
        
        Parameters
        ----------
        str
            The chan
        
        Returns
        -------
        np.ndarray
            Spectral data for a chan
        """
        return self.data[chan]

    def __setitem__(self, chan: str, chanData: np.ndarray) -> None:
        """Set channel spectra data
        
        Parameters
        ----------
        chan : str
            The channel to set the data for
        chanData : np.ndarray
            The new channel data
        """
        self.data[chan] = chanData

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
    def freqArray(self) -> np.ndarray:
        """Get the frequency array of the spectra data

        Returns
        -------
        freqArray : np.ndarray
            Array of frequencies
        """
        return getFrequencyArray(self.sampleFreq, self.dataSize)

    def toArray(self, chans: Union[List[str], None] = None) -> np.ndarray:
        """Convert the dictionary into a numpy array
        
        Each row is a channel, with the order the same as chans order

        Parameters
        ----------
        chans : List[str], optional
            The channels to put in the array. Defaults to all.
        
        Returns
        -------
        dataArray : np.ndarray
            2-D array for data with each row a channel corresponding to the order in class attribute chans.
        chans : List[str]
            The channels in the data array in the appropriate order
        """
        if chans is None:
            chans = self.chans
        dataArray = np.empty(shape=(len(chans), self.dataSize), dtype="complex")
        for idx, chan in enumerate(chans):
            dataArray[idx, :] = self.data[chan]
        return dataArray, chans

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

    def addUnitChannel(self, chan: str) -> None:
        """Add a unit channel, useful for adding intercept when processing
        
        Parameters
        ----------
        chan : str
            The name of the channel. Should not match an existing channel.
        """
        self.data[chan] = np.ones(shape=self.data[self.chans[0]].shape, dtype="complex")
        self.chans.append(chan)

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
        freqArray = self.freqArray
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
                    freqArray,
                    np.absolute(self.data[chan]),
                    color=color,
                    label=kwargs["label"],
                )
            else:
                plt.plot(freqArray, np.absolute(self.data[chan]), color=color)
            # add frequency label
            if idx == numPlotChans - 1:
                plt.xlabel("Frequency [Hz]", fontsize=plotFonts["axisLabel"])
            # x axis options
            xlim = kwargs["xlim"] if "xlim" in kwargs else [freqArray[0], freqArray[-1]]
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


def mergeSpectra(
    specData: Tuple[SpectrumData],
    channels: Union[Tuple[List[str]], None] = None,
    postpend: Union[Tuple[str], None] = None,
) -> SpectrumData:
    """Merge spectra

    Parameters
    ----------
    specData : Tuple[SpectrumData], List[SpectrumData]
        A tuple or list of SpectrumData
    channels : Tuple[List[str]], List[List[str]], optional
        The list of channels to take from each spectrum data. Defaults to all.
    postpend : Tuple[str], List[str], optional
        A string to add to the end of the channels from the different SpectrumData. Default is simply the number in Tuple.
    
    Returns
    -------
    SpectrumData
        Merged SpectrumData with all channels
    """
    from resistics.common.print import errorPrint

    if channels is None:
        channels = [sData.chans for sData in specData]
    if postpend is None:
        postpend = [str(ii) for ii in range(0, len(specData))]

    # merge spectra
    newdata = {}
    for sData, chans, post in zip(specData, channels, postpend):
        for chan in chans:
            newkey = chan + post
            newdata[newkey] = sData.data[chan]

    return SpectrumData(
        specData[0].windowSize,
        specData[0].dataSize,
        specData[0].sampleFreq,
        specData[0].startTime,
        specData[0].stopTime,
        newdata,
    )


class PowerData(ResisticsBase):
    """Class for holding auto/cross power spectra data

    Attributes
    ----------
    primaryChans : List[str]
        The primary chans in the cross power spectra calculations
    secondaryChans : List[str]
        The secondary chans in the cross power spectra calculations (conjugated)
    data : np.ndarray
        3-D data array with dimensions nprimary x nsecondary x datasize
    sampleFreq : float
        The sampling frequency of the original time data
    primaryMap : Dict[str, int]
        Mapping from primary channels to integer index in data array
    secondaryMap : Dict[str, int]
        Mapping from secondary channels to integer index in data array
    dataSize : int
        Get the data size of the cross power data
    powers : List[str]
        A list of cross powers
    numPowers : int
        The number of cross powers
    nyquist : float
        The nyquist frequency
    freqArray : np.ndarray
        The frequency array

    Methods
    -------
    __init__(primaryChans, secondaryChans, data, sampleFreq)
        Initialise the power data
    __getitem__(channel)
        The getitem accessor for getting cross power data
    isFinite()
        Checks to see if all the crosspower data is finite
    getPower(primary, secondary, fIdx)
        Get the auto or cross power from the data
    smooth(smoothLen, smoothFunc, inplace)
        Smooth the cross power data
    interpolate(newfreqs)
        Interpolate the power data to a new set of frequencies 
    """

    def __init__(
        self,
        primaryChans: List[str],
        secondaryChans: List[str],
        data: np.ndarray,
        sampleFreq: float,
    ) -> None:
        """Initialise power spectra data

        Parameters
        ----------
        primaryChans : List[str]
            The primary chans in the cross power spectra calculations
        secondaryChans : List[str]
            The secondary chans in the cross power spectra calculations (conjugated)
        data : np.ndarray
            3-D data array with dimensions nprimary x nsecondary x datasize
        sampleFreq : float
            The sampling frequency of the original time data
        """
        self.primaryChans: List[str] = primaryChans
        self.secondaryChans: List[str] = secondaryChans
        self.data: np.ndarray = data
        self.sampleFreq: float = sampleFreq
        self.primaryMap: Dict[str, int] = {}
        self.secondaryMap: Dict[str, int] = {}
        for idx, chan in enumerate(self.primaryChans):
            self.primaryMap[chan] = idx
        for idx, chan in enumerate(self.secondaryChans):
            self.secondaryMap[chan] = idx

    def __getitem__(self, args):
        """Get cross power between two channels
        
        Parameters
        ----------
        args : Tuple
            Input arguments. Expected to be either length 2 [primary, secondary] or length 3, [primary, secondary, frequency index]
        
        Returns
        -------
        np.ndarray
            Cross power data
        """
        if len(args) == 2:
            return self.getPower(args[0], args[1])
        elif len(args) == 3:
            return self.getPower(args[0], args[1], fIdx=args[2])
        else:
            self.printError(
                "Incorrect number of arguments. Only 2 or 3 arguments supported"
            )

    @property
    def dataSize(self) -> int:
        """Get the data size

        Returns
        -------
        int
            The number of elements in a single cross power
        """
        return self.data.shape[-1]

    @property
    def powers(self) -> List[List[str]]:
        """Get a list of the cross powers

        Returns
        -------
        List[str]
            The available cross powers
        """
        powers = []
        for primary in self.primaryChans:
            for secondary in self.secondaryChans:
                powers.append([primary, secondary])
        return powers

    @property
    def numPowers(self) -> int:
        """Get the number of cross powers

        Returns
        -------
        int
            The number of cross powers
        """
        return len(self.primaryChans) * len(self.secondaryChans)

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

    def isFinite(self) -> bool:
        """Checks to see if all the crosspower data is finite

        Returns
        -------
        bool
            True if all finite, False if there are bad values
        """
        if np.isfinite(self.data).all():
            return True
        return False

    def getPower(
        self, primary: str, secondary: str, fIdx: Union[int, None] = None
    ) -> Union[np.ndarray, None, float]:
        """Get the auto or cross power from the data

        .. note::

            The order matters as getPower(chan1, chan2) will be the complex conjugate of getPower(chan2, chan1)

        Paramters
        ---------
        primary : str
            The first channel
        secondary : str
            The second channel
        fIdx : int, optional
            The frequency index to get
        
        Returns
        -------
        np.ndarray, float
            Will return the data array if no frequency index is specified. Where fIdx is specified, a float is returns.
        """
        if fIdx is None:
            return self.data[
                self.primaryMap[primary], self.secondaryMap[secondary], :
            ].squeeze()
        else:
            return self.data[
                self.primaryMap[primary], self.secondaryMap[secondary], fIdx
            ]

    def smooth(
        self, smoothLen: int, smoothFunc: str, inplace: bool = False
    ) -> Union[None, ResisticsBase]:
        """Smooth the power data

        Parameters
        ----------
        smoothLen : int
            The window length to use
        smoothFunc : str
            The window function
        inplace : bool, optional
            Whether to do the smoothing in place or return a new PowerData instance

        Returns
        -------
        None, PowerData
            Returns None if inplace is True, otherwise returns a new PowerData instance
        """
        from resistics.common.smooth import smooth1d

        newdata = np.empty(shape=self.data.shape, dtype="complex")
        for iPri in range(len(self.primaryChans)):
            for iSec in range(len(self.secondaryChans)):
                newdata[iPri, iSec] = smooth1d(
                    self.data[iPri, iSec], smoothLen, smoothFunc
                )
        if inplace:
            self.data = newdata
            return None
        return PowerData(
            self.primaryChans, self.secondaryChans, newdata, self.sampleFreq
        )

    def interpolate(self, newfreqs: np.ndarray) -> ResisticsBase:
        """Interpolate the power data

        Parameters
        ----------
        newfreqs : np.ndarray
            numpy array of new frequencies to interpolate to

        Returns
        -------
        PowerData
            Returns PowerData with cross power data interpolated to newfreqs
        """
        import scipy.interpolate as interp

        freq = self.freqArray
        shape = self.data.shape
        newdata = np.empty(shape=(shape[0], shape[1], len(newfreqs)), dtype="complex")
        for iPri in range(len(self.primaryChans)):
            for iSec in range(len(self.secondaryChans)):
                interpFunc = interp.interp1d(freq, self.data[iPri, iSec])
                newdata[iPri, iSec] = interpFunc(newfreqs)
        return PowerData(
            self.primaryChans, self.secondaryChans, newdata, self.sampleFreq
        )

    def view(self, **kwargs) -> Figure:
        """Plot spectra data

        Parameters
        ----------
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts
        powers : List[List[str]], optional
            A list of cross powers to plot
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
        freqArray = self.freqArray
        fig = (
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(20, 10))
        )
        plotFonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()
        color = kwargs["color"] if "color" in kwargs else None
        # suptitle
        st = fig.suptitle("Cross power data", fontsize=plotFonts["suptitle"])
        st.set_y(0.98)
        # plot data
        plotPowers = kwargs["powers"] if "powers" in kwargs else self.powers
        for power in plotPowers:
            plt.plot(
                freqArray, np.absolute(self.getPower(power[0], power[1])), color=color,
            )
            # x axis options
            plt.xlabel("Frequency [Hz]", fontsize=plotFonts["axisLabel"])
            xlim = kwargs["xlim"] if "xlim" in kwargs else [freqArray[0], freqArray[-1]]
            plt.xlim(xlim)
            # y axis options
            plt.ylabel("Cross power amplitude", fontsize=plotFonts["axisLabel"])
            plt.grid()
            # set tick sizes
            ax = plt.gca()
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
        textLst.append("Nyquist frequency [Hz] = {}".format(self.nyquist))
        textLst.append("Number of powers = {}".format(self.numPowers))
        textLst.append("Primary channels = {}".format(self.primaryChans))
        textLst.append("Secondary channels = {}".format(self.secondaryChans))
        textLst.append("Available crosspowers = {}".format(self.powers))
        return textLst
