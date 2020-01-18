from copy import deepcopy
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.dates import (
    DateFormatter,
    DayLocator,
    AutoDateLocator,
    AutoDateFormatter,
)
from matplotlib.figure import Figure
from matplotlib.collections import PathCollection
from typing import List, Dict, Tuple, Union, Any

# import from package
from resistics.common.base import ResisticsBase
from resistics.common.print import arrayToString
from resistics.window.utils import gArray2datetime
from resistics.common.plot import (
    getPlotRowsAndCols,
    getViewFonts,
    colorbar2dOther,
    colorbar2dTime,
)


class StatisticData(ResisticsBase):
    """Class for holding information about a statistic

    Attributes
    ----------
    statName : str
        The name of the statistic
    refTime : datetime.datetime
        The reference time of the project
    sampleFreq : float
        The sampling frequency of the data
    winSize : int
        Window size in samples
    winOlap : int
        Window overlap in samples
    numWindows : int
        Number of windows
    winStats : List[str]
        Values calculated for the statistic
    numStatsPerWindow : int
        The number of statistics per window (length of winStats)
    stats : np.ndarray
        The statistic data of size number of windows * number evaluation frequencies * number of window stats        
    evalFreq : List, np.ndarray
        The evaluation frequencies for the statistic data
    freq2index : Dict
        Mapping from evaluation frequency to index
    globalIndices : List, np.ndarray
        Array of global indices. Allows local -> global conversion
    global2localMap : Dict
        Dictionary which maps global indices to local. Allows global -> local conversion
    comments : List[str]
        Statistic comments
    dtype : str (default "float")
        The data type of the statistic values
    maxcols : int (default 4)
        The number of columns in plots
    globalDatesStart : List, np.ndarray
        Global window start datetimes
    globalDatesStop : List, np.ndarray
        Global window stop datetimes

    Methods
    -------
    __init__(sampleFreq, numLevels, evalFreq, kwargs)
        Initialise maskData
    getStats(includewindows, maskwindows)
        Get the statistics array and choose to either include some windows or mask some windows
    getMaskedWindows(maskwindows)
        Return the local indices to use given a set of global indices to exclude
    getStatLocal(localIndex)
        Get statistic values for a local window index
    getStatGlobal(globalIndex)
        Get statistic values for a global window index
    getGlobalIndex(localIndex)
        Get the global window index for a local window index
    getGlobalDates()
        Get the global index dates for all the windows
    setStatParams(numWindows, winStats, evalFreq)
        Set the statistic parameters and prepare the data arrays
    addStat(localIndex, globalIndex, stat)
        Add a statistic, providing the local index, the corresponding global index and the statististic values
    getComments()
        Get a deepcopy of the comments    
    addComment(comment)
        Add a comment to the comments
    view(eFreqI, **kwargs) 
        View statistic values for a evaluation frequency (time on x axis, statistic values on y axis)
    histogram(eFreqI, **kwargs)
        View a histogram of the statistic values for each statistic component for a single evaluation frequency
    crossplot(eFreqI, **kwargs)   
        View a crossplot of the data
    addColourbar(plot, cax, title, plotfonts)
        Add a colourbar to a plot    
    addColourbarDates(plot, cax, title, plotfonts)
        Add a coloubar the represents dates to the plot   
    dateTicks(gIndices, dates, timeNum)
        Deal with the dateTicks of the plot
    calcColourData(plotData, val, eFreqI, keywords)
        Calculate colours 
    getRowsCols(maxcols, numStats)
        Get the number of rows and columns for a plot
    printList()
        Class status returned as list of strings   
    """

    def __init__(
        self,
        statName: str,
        refTime: datetime,
        sampleFreq: float,
        winSize: int,
        winOlap: int,
        **kwargs
    ):
        """Initialise statistic data 
    
        Parameters
        ----------
        statName : str
            Name of the statistic
        refTime : datetime.datetime
            Reference time for the project
        sampleFreq : float
            Sample frequency of the data in Hz 
        winSize : int
            The window size in samples
        winOlap : int
            The window overlap in samples
        kwargs : Dict
            Two optional arguments: "name" for statistic name and "stats" for the statistics to use
        """
        self.statName: str = statName
        self.refTime = refTime
        if isinstance(self.refTime, str):
            self.refTime = datetime.strptime(self.refTime, "%Y-%m-%d %H:%M:%S.%f")
        self.sampleFreq: float = sampleFreq
        self.winSize: int = winSize
        self.winOlap: int = winOlap
        # defaults from keywords
        self.numWindows: int = 0
        self.winStats: List[str] = []
        self.numStatsPerWindow: int = 0
        self.evalFreq: Union[List, np.ndarray] = []
        self.freq2index: Dict = {}
        self.globalIndices: Union[List, np.ndarray] = []
        self.global2localMap: Dict = {}
        self.comments: List[str] = []
        self.dtype: str = "float"
        # global dates
        self.globalDatesStarts: Union[List, np.ndarray] = []
        self.globalDatesStops: Union[List, np.ndarray] = []
        # plot params
        self.maxcols: int = 4
        # parse keywords
        self.initialiseFromKeywords(kwargs)

    def initialiseFromKeywords(self, keywords):
        """Initialise object properties using the keywords"""
        if "numWindows" in keywords:
            self.numWindows = keywords["numWindows"]
        if "winStats" in keywords:
            self.winStats = keywords["winStats"]
        self.numStatsPerWindow = len(self.winStats)
        if "evalFreq" in keywords:
            self.evalFreq = keywords["evalFreq"]
        for idx, eFreq in enumerate(self.evalFreq):
            self.freq2index[eFreq] = idx
        if "stats" in keywords:
            self.stats = keywords["stats"]
        if "globalIndices" in keywords:
            self.globalIndices = keywords["globalIndices"]
            # make the global 2 local map
            for ii in np.arange(0, len(self.globalIndices)):
                self.global2localMap[self.globalIndices[ii]] = ii
        if "comment" in keywords:
            self.comments = keywords["comment"]
        if "dtype" in keywords:
            self.dtype = keywords["dtype"]

    def getStats(
        self,
        includewindows: Union[np.ndarray, List] = [],
        maskwindows: Union[np.ndarray, List] = [],
    ) -> np.ndarray:
        """Get statistics when either selecting a set of indices or wanting to mask a set of indices

        Parameters
        ----------
        includewindows : List, np.ndarray, optional
            Windows to get
        maskwindows : List, np.ndarray, optional
            Windows to exclude
        
        Returns
        -------
        np.ndarray
            Statistics limited by the window selection options
        """
        if len(includewindows) > 0:
            return self.stats[includewindows, :]
        elif len(maskwindows) > 0:
            includewindows = self.getMaskedIndices(maskwindows)
            # if this is false, then want to return all the statistics
            if includewindows:
                return self.stats[includewindows, :]
        return self.stats

    def getMaskedIndices(
        self, maskWindows: Union[List, np.ndarray]
    ) -> Union[List, bool]:
        """Given a list of global windows to mask, returns the set of indices to include in the plot

        Parameters
        ----------
        maskWindows : List, np.ndarray
            Global indices of windows to mask
        
        Returns
        -------
        List
            List of indices to include
        """
        includeWindows = set(self.globalIndices) - set(maskWindows)
        if len(includeWindows) == self.numWindows:
            # nothing to mask
            return False

        indices = []
        for win in includeWindows:
            indices.append(self.global2localMap[win])
        return indices

    def getStatLocal(self, localIndex: int) -> np.ndarray:
        """Get statistics for a local window index

        Parameters
        ----------
        localIndex : int
            Local index of the window

        Returns
        -------
        out : np.ndarray
            Statistics for the local window
        """
        return self.stats[localIndex, :]

    def getStatGlobal(self, globalIndex: int) -> np.ndarray:
        """Get statistics for a local window index

        Parameters
        ----------
        globalIndex : int
            Global index of the window

        Returns
        -------
        out : np.ndarray
            Statistics for the local window
        """
        if globalIndex not in self.global2localMap:
            self.printError(
                "There are two statistics with the same global window index"
            )
            self.printError("This should never happen")
            self.printError("Exiting", quitRun=True)
        localIndex = self.global2localMap[globalIndex]
        return self.stats[localIndex]

    def getGlobalIndex(self, localIndex):
        """Get global index for local index

        Parameters
        ----------
        localIndex : int
            Local index of the window

        Returns
        -------
        globalIndex : int
            Global index for local index
        """
        return self.globalIndices[localIndex]

    def getGlobalDates(self):
        """Get the global start dates and end dates for each window
        
        Returns
        -------
        np.ndarray
            The global window start times
        """
        # want to get the start date for each window here
        if len(self.globalDatesStarts) != self.numWindows:
            self.globalDatesStarts, self.globalDatesStops = gArray2datetime(
                self.globalIndices,
                self.refTime,
                self.sampleFreq,
                self.winSize,
                self.winOlap,
            )
        return self.globalDatesStarts

    def setStatParams(
        self, numWindows: int, winStats: List[str], evalFreq, dtype: Any = None
    ):
        """Set the statistic parameters

        Parameters
        ----------
        numWindows : int
            Number of windows for which the statistic has been or will be calculated
        winStats : int
            The parameters in the statistic
        evalFreq : List
            A list of evaluation frequencies
        dtype : str, optional
            The datatype of the statistic values
        """
        self.numWindows = numWindows
        # details about the statistics
        self.winStats = winStats
        self.numStatsPerWindow = len(winStats)
        # details about the evaluation frequencies
        self.evalFreq = evalFreq
        for idx, eFreq in enumerate(evalFreq):
            self.freq2index[eFreq] = idx
        # data type
        if not (dtype is None):
            self.dtype = dtype
        self.stats = np.empty(
            shape=(self.numWindows, self.evalFreq.size, self.numStatsPerWindow),
            dtype=self.dtype,
        )
        # an array to hold global indices and a dictionary to map them back to local indices
        self.globalIndices = np.empty(shape=(self.numWindows), dtype=int)
        self.global2localMap = {}

    def addStat(self, localIndex: int, globalIndex: int, stat) -> None:
        """Add statistic values for a window for all evaluation frequencies

        Parameters
        ----------
        localIndex : int
            Local index of the window
        globalIndex : int
            Global index of the window
        stat : Dict
            An dictionary of dictionaries. First set of key, values are evaluationFrequencies and dictionaries, second set of key, values are statistic names in winStats and their corresponding value for the window
        """
        # add the data - the data is in the format [eFreq][key][val]
        for idx, eFreq in enumerate(self.evalFreq):
            # do it in this order because sorted
            for ist, st in enumerate(self.winStats):
                self.stats[localIndex, idx, ist] = stat[eFreq][st]
        # then add the global index
        self.globalIndices[localIndex] = globalIndex
        self.global2localMap[globalIndex] = localIndex

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

    def view(self, eFreqI: int, **kwargs) -> Figure:
        """Plot statistics for evaluation frequency index

        Plots a simple scatter of each statistic with datetime on the xaxis (datetime of the window start dates). Number of subplots is equal to numStaStatPerWindow.

        Parameters
        ----------
        eFreqI : int
            Evaluation frequency index
        maskwindows : List, np.ndarray
            Global windows to exclude
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts
        label : str, optional
            Label for the plots
        clim : List, optional
            Limits for colourbar axis
        xlim : List, optional
            Limits for the x axis
        ylim : List, optional
            Limits for the y axis
        colortitle : str, optional
            Title for the colourbar
        legened : bool
            Boolean flag for adding a legend
        
        Returns
        -------
        plt.figure
            Matplotlib figure object        
        """
        # get windows to plot and global dates
        maskWindows = kwargs["maskwindows"] if "maskwindows" in kwargs else []
        plotIndices = self.getMaskedIndices(maskWindows)
        globalDates = self.getGlobalDates()
        eFreq = self.evalFreq[eFreqI]

        # plot params
        nrows, ncols = self.getRowsCols(self.maxcols)
        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()

        fig: plt.figure = (
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(4 * ncols, 5 * nrows))
        )
        st = fig.suptitle(
            "{} data for evaluation frequency: {}".format(self.statName, eFreq),
            fontsize=plotfonts["suptitle"],
        )
        st.set_y(0.98)

        # plot the data
        for idx, val in enumerate(self.winStats):
            ax = plt.subplot(nrows, ncols, idx + 1)
            plt.title("Value {}".format(val), fontsize=plotfonts["title"])
            label = kwargs["label"] if "label" in kwargs else eFreq

            # limit the data by plotIndices if not False
            plotData = np.squeeze(self.stats[:, eFreqI, idx])
            plotDates = globalDates
            if plotIndices:
                plotData = plotData[plotIndices]
                plotDates = plotDates[plotIndices]

            # the colourdata
            colourbool, colourdata, cmap = self.calcColourData(
                plotData, val, eFreqI, kwargs
            )
            # scatter plot
            if not colourbool:
                scat = plt.scatter(
                    plotDates,
                    plotData,
                    edgecolors="none",
                    marker="o",
                    s=12,
                    label=label,
                )
            else:
                scat = plt.scatter(
                    plotDates,
                    plotData,
                    c=colourdata,
                    edgecolors="none",
                    marker="o",
                    s=12,
                    cmap=cmap,
                    label=label,
                )
                clim = (
                    kwargs["clim"]
                    if ("clim" in kwargs and len(kwargs["clim"]) > 0)
                    else [colourdata.min(), colourdata.max()]
                )
                scat.set_clim(clim)
            # x axis options
            plt.xlabel("Time", fontsize=plotfonts["axisLabel"])
            xlim = (
                kwargs["xlim"]
                if ("xlim" in kwargs and len(kwargs["xlim"]) > 0)
                else [globalDates[0], globalDates[-1]]
            )
            plt.xlim(xlim)
            ax.format_xdata = DateFormatter("%H-%M-%S")
            fig.autofmt_xdate()
            # y axis options
            if "ylim" in kwargs and len(kwargs["ylim"]) > 0:
                plt.ylim(kwargs["ylim"])
            plt.ylabel("Value {}".format(val), fontsize=plotfonts["axisLabel"])
            # set tick sizes
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])
            plt.grid(True, ls="--")
            # legend
            if "legend" in kwargs and kwargs["legend"]:
                plt.legend(loc=4)

        # show if the figure is not in keywords
        if "fig" not in kwargs:
            if colourbool:
                fig.tight_layout(rect=[0.02, 0.02, 0.85, 0.92])
                cax = fig.add_axes([0.88, 0.10, 0.03, 0.80])
                colourtitle = (
                    kwargs["colortitle"] if "colortitle" in kwargs else "Value"
                )
                self.addColourbar(scat, cax, colourtitle, plotfonts)
            else:
                fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
            plt.show()

        return fig

    def histogram(self, eFreqI: int, **kwargs) -> Figure:
        """Plot statistics for evaluation frequency index

        Plots a histogram of each statistic with bins on the xaxis and count on the yaxis. Ideal for exploring the distribution of statistic values over the windows.

        Parameters
        ----------
        eFreqI : int
            Evaluation frequency index
        maskwindows : List, np.ndarray
            Global windows to exclude            
        numbins : int
            The number of bins for the histogram data binning
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts
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
        # deal with maskwindows, which are global indices
        maskWindows = kwargs["maskwindows"] if "maskwindows" in kwargs else []
        plotIndices = self.getMaskedIndices(maskWindows)
        eFreq = self.evalFreq[eFreqI]

        # plot options
        numbins = kwargs["numbins"] if "numbins" in kwargs else 40
        nrows, ncols = self.getRowsCols(self.maxcols)

        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()
        fig = (
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(4 * ncols, 4 * nrows))
        )
        st = fig.suptitle(
            "{} data for evaluation frequency: {}".format(self.statName, eFreq),
            fontsize=plotfonts["suptitle"],
        )
        st.set_y(0.98)

        # plot the data
        for idx, val in enumerate(self.winStats):
            ax = plt.subplot(nrows, ncols, idx + 1)
            plt.title("Value {}".format(val), fontsize=plotfonts["title"])
            label = kwargs["label"] if "label" in kwargs else eFreq

            # data
            plotData = np.squeeze(self.stats[:, eFreqI, idx])
            if plotIndices:
                plotData = plotData[plotIndices]
            # remove infinities and nans
            plotData = plotData[np.isfinite(plotData)]

            # x axis options
            xlim = (
                kwargs["xlim"]
                if "xlim" in kwargs
                else [np.min(plotData), np.max(plotData)]
            )
            plt.xlim(xlim)
            plt.xlabel("Value", fontsize=plotfonts["axisLabel"])
            # now plot with xlim in mind
            plt.hist(plotData, numbins, range=xlim, facecolor="red", alpha=0.75)
            # y axis options
            plt.ylabel("Count", fontsize=plotfonts["axisLabel"])
            # set tick sizes
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])
            plt.grid(True, ls="--")
            # legend
            if "legend" in kwargs and kwargs["legend"]:
                plt.legend(loc=4)

        # show if the figure is not in keywords
        if "fig" not in kwargs:
            fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
            plt.show()

        return fig

    def crossplot(self, eFreqI: int, **kwargs) -> Figure:
        """Plots crossplots of statistic components for evaluation frequency index

        Notes
        -----
        By default, the crossplots plotted are
        statistic component 1 vs statistic component 2
        statistic component 3 vs statistic component 4
        etc

        But crossplots can be explicity set by using the crossplots keyword. They should be specified as a list of a list of strings
        e.g.
        crossplots = [[component2, component3], [component1, component4]]

        Parameters
        ----------
        eFreqI : int
            Evaluation frequency index
        maskwindows : List, np.ndarray
            Global windows to exclude              
        crossplots : List[List[str]], optional
            The parameters to crossplot
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts
        label : str, optional
            Label for the plots
        clim : List, optional
            Limits for colourbar axis
        xlim : List, optional
            Limits for the x axis
        ylim : List, optional
            Limits for the y axis
        colortitle : str, optional
            Title for the colourbar
        legened : bool
            Boolean flag for adding a legend
        
        Returns
        -------
        plt.figure
            Matplotlib figure object            
        """
        # deal with maskwindows, which are global indices
        maskWindows = kwargs["maskwindows"] if "maskwindows" in kwargs else []
        plotIndices = self.getMaskedIndices(maskWindows)

        # figure out the crossplots
        if "crossplots" in kwargs:
            crossplots = kwargs["crossplots"]
        else:
            crossplots = list(zip(self.winStats[::2], self.winStats[1::2]))

        # plot parameters
        nrows, ncols = self.getRowsCols(self.maxcols, numStats=len(crossplots))
        eFreq = self.evalFreq[eFreqI]

        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()
        fig = (
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(4 * ncols, 5 * nrows))
        )
        st = fig.suptitle(
            "{} crossplots for evaluation frequency: {:.3f} Hz".format(
                self.statName, eFreq
            ),
            fontsize=plotfonts["suptitle"],
        )
        st.set_y(0.98)

        # now plot the data
        for idx, cplot in enumerate(crossplots):
            ax = plt.subplot(nrows, ncols, idx + 1)
            plt.title(
                "{} vs. {}".format(cplot[0], cplot[1]), fontsize=plotfonts["title"]
            )
            label = kwargs["label"] if "label" in kwargs else eFreq

            # the colourdata
            colourbool, colourdata, cmap = self.calcColourData(
                self.globalIndices, cplot[0], eFreqI, kwargs
            )

            # get plot data
            plotI1 = self.winStats.index(cplot[0])
            plotData1 = np.squeeze(self.stats[:, eFreqI, plotI1])
            plotI2 = self.winStats.index(cplot[1])
            plotData2 = np.squeeze(self.stats[:, eFreqI, plotI2])
            if plotIndices:
                plotData1 = plotData1[plotIndices]
                plotData2 = plotData2[plotIndices]
                colourdata = colourdata[plotIndices]

            # scatter plot
            scat = plt.scatter(
                plotData1,
                plotData2,
                c=colourdata,
                edgecolors="none",
                marker="o",
                s=12,
                cmap=cmap,
                label=label,
            )
            clim = (
                kwargs["clim"]
                if "clim" in kwargs
                else [colourdata.min(), colourdata.max()]
            )
            scat.set_clim(clim)
            # x axis options
            plt.xlabel("Value {}".format(cplot[0]), fontsize=plotfonts["axisLabel"])
            if "xlim" in kwargs:
                plt.xlim(kwargs["xlim"])
            # y axis options
            plt.ylabel("Value {}".format(cplot[1]), fontsize=plotfonts["axisLabel"])
            if "ylim" in kwargs:
                plt.ylim(kwargs["ylim"])
            # set tick sizes
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])
            plt.grid(True, ls="--")
            # legend
            if "legend" in kwargs and kwargs["legend"]:
                plt.legend(loc=4)

        # show if the figure is not in keywords
        if "fig" not in kwargs:
            fig.tight_layout(rect=[0.02, 0.02, 0.85, 0.92])
            cax = fig.add_axes([0.88, 0.10, 0.03, 0.80])
            if not colourbool:
                self.addColourbarDates(scat, cax, "Time", plotfonts)
            else:
                colourtitle = (
                    kwargs["colortitle"] if "colortitle" in kwargs else "Value"
                )
                self.addColourbar(scat, cax, colourtitle, plotfonts)
            plt.show()

        return fig

    def densityplot(self, eFreqI: int, **kwargs) -> Figure:
        """Plots density plots of statistic components for evaluation frequency index

        Notes
        -----
        By default, the density plots plotted are
        statistic component 1 vs statistic component 2
        statistic component 3 vs statistic component 4
        etc

        But density plots can be explicity set by using the crossplots keyword. They should be specified as a list of a list of strings
        e.g.
        crossplots = [[component2, component3], [component1, component4]]

        Parameters
        ----------
        eFreqI : int
            Evaluation frequency index
        maskwindows : List, np.ndarray
            Global windows to exclude              
        crossplots : List[List[str]], optional
            The parameters to crossplot
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts
        label : str, optional
            Label for the plots
        xlim : List, optional
            Limits for the x axis
        ylim : List, optional
            Limits for the y axis
        colortitle : str, optional
            Title for the colourbar
        legened : bool
            Boolean flag for adding a legend
        
        Returns
        -------
        plt.figure
            Matplotlib figure object            
        """
        # deal with maskwindows, which are global indices
        maskWindows = kwargs["maskwindows"] if "maskwindows" in kwargs else []
        plotIndices = self.getMaskedIndices(maskWindows)

        # figure out the crossplots
        if "crossplots" in kwargs:
            crossplots = kwargs["crossplots"]
        else:
            crossplots = list(zip(self.winStats[::2], self.winStats[1::2]))

        # plot parameters
        nrows, ncols = self.getRowsCols(self.maxcols, numStats=len(crossplots))
        eFreq = self.evalFreq[eFreqI]

        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()
        fig = (
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(4 * ncols, 5 * nrows))
        )
        st = fig.suptitle(
            "{} density plots for evaluation frequency: {:.3f} Hz".format(
                self.statName, eFreq
            ),
            fontsize=plotfonts["suptitle"],
        )
        st.set_y(0.98)

        # now plot the data
        for idx, cplot in enumerate(crossplots):
            ax = plt.subplot(nrows, ncols, idx + 1)
            plt.title(
                "{} vs. {}".format(cplot[0], cplot[1]), fontsize=plotfonts["title"]
            )
            label = kwargs["label"] if "label" in kwargs else eFreq

            # get plot data
            plotI1 = self.winStats.index(cplot[0])
            plotData1 = np.squeeze(self.stats[:, eFreqI, plotI1])
            plotI2 = self.winStats.index(cplot[1])
            plotData2 = np.squeeze(self.stats[:, eFreqI, plotI2])
            if plotIndices:
                plotData1 = plotData1[plotIndices]
                plotData2 = plotData2[plotIndices]

            nbins = 200
            if "xlim" in kwargs:
                plt.xlim(kwargs["xlim"])
                rangex = kwargs["xlim"]
            else:
                minx = np.percentile(plotData1, 2)
                maxx = np.percentile(plotData1, 98)
                plt.xlim(minx, maxx)
                rangex = [minx, maxx]

            if "ylim" in kwargs:
                plt.ylim(kwargs["ylim"])
                rangey = kwargs["ylim"]
            else:
                miny = np.percentile(plotData2, 2)
                maxy = np.percentile(plotData2, 98)
                plt.ylim(miny, maxy)
                rangey = [miny, maxy]

            plt.hist2d(
                plotData1,
                plotData2,
                bins=(nbins, nbins),
                range=[rangex, rangey],
                cmap=plt.cm.inferno,
            )

            # axis options
            plt.xlabel("Value {}".format(cplot[0]), fontsize=plotfonts["axisLabel"])
            plt.ylabel("Value {}".format(cplot[1]), fontsize=plotfonts["axisLabel"])
            # set tick sizes
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])
            plt.grid(True, ls="--")
            # legend
            if "legend" in kwargs and kwargs["legend"]:
                plt.legend(loc=4)

        # show if the figure is not in keywords
        if "fig" not in kwargs:
            fig.tight_layout(rect=[0.02, 0.02, 0.85, 0.92])
            plt.show()

        return fig

    def addColourbar(
        self, plot: PathCollection, cax, title: str, plotfonts: Dict
    ) -> None:
        """Add a colourbar to a plot

        Parameters
        ----------
        plot : matplotlib.collections.PathCollection
            A plot e.g. a scatter graph
        cax : 
            Colourbar axis
        title : str
            The tital for the colourbar
        plotfonts : Dict
            A dictionary with font types as keys and sizes as values
        """
        plt.colorbar(plot, cax=cax)
        cax.set_title(title, y=1.02, fontsize=plotfonts["title"])

    def addColourbarDates(self, plot: PathCollection, cax, title: str, plotfonts: Dict):
        """Make the colourbar show dates for identifying window times where there is not date axis

        Parameters
        ----------
        plot : matplotlib.collections.PathCollection
            A plot e.g. a scatter graph
        cax : 
            Colourbar axis
        title : str
            The tital for the colourbar
        plotfonts : Dict
            A dictionary with font types as keys and sizes as values
        """
        ticks, tickLabels = self.dateTicks(self.globalIndices, self.getGlobalDates(), 5)
        cb = plt.colorbar(plot, cax=cax)
        cb.set_ticks(ticks)
        cb.set_ticklabels(tickLabels)
        cax.set_title(title, y=1.02, fontsize=plotfonts["title"])

    def dateTicks(self, gIndices, dates, timeNum):
        """Format dateticks
        
        .. todo::

            Write more complete documentation
        """
        numVals = len(gIndices)
        if timeNum >= numVals:
            timeNum = numVals
        plotIndices = []
        for i in range(0, timeNum):
            plotIndices.append(int(i * numVals * 1.0 / (timeNum - 1)))
        plotIndices[-1] = numVals - 1
        ticks = []
        tickLabels = []
        for i in plotIndices:
            ticks.append(gIndices[i])
            tickLabels.append(dates[i].strftime("%m-%d %H:%M:%S"))
        return ticks, tickLabels

    def calcColourData(self, plotData, val, eFreqI, keywords):
        """Calculate the colour data
        
        .. todo::

            Write more complete documentation
        """
        if "colorstat" in keywords and "colormap" in keywords:
            colourVal = keywords["colormap"][val]
            colourIndex = keywords["colorstat"].winStats.index(colourVal)
            return (
                True,
                np.squeeze(keywords["colorstat"].stats[:, eFreqI, colourIndex]),
                colorbar2dOther(),
            )
        return False, plotData, colorbar2dTime()

    def getRowsCols(self, maxcols: int, numStats: int = 0) -> Tuple[int, int]:
        """Get the numbers of rows and columns for the plots

        Parameters
        ----------
        maxcols : int
            The maximum number of columns
        numStats : int
            The number of statistics to plot (in case this is not all of the winStats)
        """
        if numStats < 1:
            numStats = self.numStatsPerWindow
        return getPlotRowsAndCols(maxcols, numStats)

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """
        textLst: List[str] = []
        textLst.append("Statistic Name = {}".format(self.statName))
        textLst.append(
            "Reference time = {}".format(self.refTime.strftime("%Y-%m-%d %H:%M:%S.%f"))
        )
        textLst.append("Sample freq [Hz] = {}".format(self.sampleFreq))
        textLst.append("Window size = {}".format(self.winSize))
        textLst.append("Window overlap = {}".format(self.winOlap))
        textLst.append("Number of windows = {}".format(self.numWindows))
        textLst.append("Statistics per Window = {}".format(", ".join(self.winStats)))
        textLst.append(
            "Evalutation frequencies = {}".format(arrayToString(self.evalFreq))
        )
        # comments
        textLst.append("Comments...")
        if len(self.comments) == 0:
            textLst.append("No comments")
        else:
            for comment in self.comments:
                textLst.append("\t{}".format(comment))
        return textLst

