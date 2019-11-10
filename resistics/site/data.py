import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime, timedelta
from typing import List, Dict, Union

from resistics.common.base import ResisticsBase
from resistics.common.io import getDataDirsInDirectory
from resistics.common.print import arrayToString
from resistics.common.plot import getViewFonts
from resistics.time.utils import getTimeReader


class SiteData(ResisticsBase):
    """Class for holding site information

    Attributes
    ----------
    timePath : int
        Path to site time data
    specPath : float
        Path to site spectra data
    statPath : datetime.datetime
        Path to site stat data
    transFuncPath : datetime.datetime
        Path to site transfer function data 
    measurements : List[str]
        A list of the measurements in the site
    spectra : List[str]
        A list of spectra data for the site
    statistics : List[str]
        A list of statistics data for the site
    transferFunctions : List[str]
        A list of transfer function data for the site
    siteStart : datetime.datetime
        The time of the first sample in the site
    siteEnd : datetime.datetime
        The time of the last sample in the site
    readers : Dict
        Data readers for each measurement
    starts : Dict

    Methods
    -------
    __init__(siteName, timePath, specPath, statPath, transFuncPath)
        Initialise the site data
    refresh()
        Refresh site data
    getSampleFreqs()
        Get list of unique sampling frequencies in site
    getMeasurements(sampleFreq=None) 
        Get list of measurements in site, optionally filtered by sampling frequency
    getMeasurement(meas)
        Get the dataReader for the measurement
    getMeasurementSampleFreq(meas)
        Get the sampling frequency of a measurement in Hz        
    getMeasurementTimePath(meas)
        Get the path for a measurement
    getMeasurementSpecPath(meas)
        Get the path for the spectra of a measurement
    getMeasurementStatPath(meas)
        Get the path for the statistics of a measurement
    getMeasurementTransFuncPath(meas)
        Get the path for the transfer function data of a measurement
    getMeasurementStart(meas)
        Get the time of the first sample of a measurement
    getMeasurementEnd(meas)
        Get the time of the last sample of a measurement
    view(kwargs)
        View the site data measurement information 
    printList()
        Class status returned as list of strings          
    """

    def __init__(
        self,
        siteName: str,
        timePath: str,
        specPath: str,
        statPath: str,
        maskPath: str,
        transFuncPath: str,
    ) -> None:
        self.siteName: str = siteName
        self.timePath: str = os.path.join(timePath, siteName)
        self.specPath: str = os.path.join(specPath, siteName)
        self.statPath: str = os.path.join(statPath, siteName)
        self.maskPath: str = os.path.join(maskPath, siteName)
        self.transFuncPath: str = os.path.join(transFuncPath, siteName)
        # measurement information
        self.measurements: List[str] = []
        self.spectra: List[str] = []
        self.statistics: List[str] = []
        self.transferFunctions: List[str] = []
        self.siteStart = datetime.now()
        self.siteEnd = datetime.now()
        self.readers: Dict = {}
        self.starts: Dict[str, datetime] = {}
        self.ends: Dict[str, datetime] = {}
        self.fs: Dict[str, float] = {}
        # refresh site data
        self.refresh()

    def refresh(self):
        """ Refresh site information"""
        self.measurements = getDataDirsInDirectory(self.timePath)
        for meas in self.measurements:
            # initialise datapath and the various readers
            datapath = os.path.join(self.timePath, meas)
            reader = getTimeReader(datapath)
            if not reader:  # then no data found in this directory
                self.printWarning(
                    "No data found in measurement directory {} for site {}. Please remove this directory".format(
                        meas, self.siteName
                    )
                )
                continue
            # otherwise get the reader and information
            self.readers[meas] = reader
            self.starts[meas] = reader.getStartDatetime()
            self.ends[meas] = reader.getStopDatetime()
            self.fs[meas] = reader.getSampleFreq()

        # check to see if no data directories were found
        if len(self.fs) == 0:
            self.printWarning(
                "No recognised measurement formats in site {}".format(self.siteName)
            )
            return

        # start and end time of project
        if len(self.measurements) != 0:
            self.siteStart = min(self.starts.values())
            self.siteEnd = max(self.ends.values())

        # spectra, statistic and transfer function data
        self.spectra = getDataDirsInDirectory(self.specPath)
        self.statistics = getDataDirsInDirectory(self.statPath)
        self.transferFunctions = getDataDirsInDirectory(self.transFuncPath)

    def getSampleFreqs(self) -> List[float]:
        """Get a list of all the distinct sampling frequencies in the site

        Returns
        -------
        List[float]
            List of the unique sampling frequencies in a site
        """
        sampleFreqs = set(self.fs.values())
        return sorted(list(sampleFreqs))

    def getMeasurements(self, sampleFreq: Union[None, float, int] = None) -> List[str]:
        """Get a list of measurements for a site

        Parameters
        ----------
        sampleFreq : float, int, optional
            Sampling frequency

        Returns
        -------
        List[str]
            List of site time files. If fs is supplied, then only time files with sampling frequency fs.
        """
        if sampleFreq is None:
            return self.measurements
        # fs is supplied
        filteredMeasurements = []
        for meas in self.measurements:
            if self.fs[meas] == sampleFreq:
                filteredMeasurements.append(meas)
        return filteredMeasurements

    def getMeasurement(self, meas: str):
        """Get the data reader of a measurement at a site

        Parameters
        ----------
        meas : str
            The name of the measurement

        Returns
        -------
        DataReader, bool
            Returns the data reader for the measurement or if the measurement is not found, False
        """
        if self.checkMeasurement(meas):
            return self.readers[meas]
        self.printWarning(
            "Meausrement directory {} for site {} not found".format(self.siteName, meas)
        )
        return False

    def getMeasurementSampleFreq(self, meas: str) -> float:
        """Get the sample frequency in Hz of a particular measurement at a site

        Parameters
        ----------
        site : str
            The name of the site
        meas : str
            The name of the measurement

        Returns
        -------
        float
            Sampling frequency of site
        """
        return self.fs[meas]

    def getMeasurementTimePath(self, meas: str) -> str:
        """Get the time data path for a measurement at a site

        Parameters
        ----------
        meas : str
            The name of the measurement

        Returns
        -------
        str
            Path to the time data for a measurement at the site
        """
        return os.path.join(self.timePath, meas)

    def getMeasurementSpecPath(self, meas: str) -> str:
        """Get the spectra data path for a measurement at a site

        Parameters
        ----------
        meas : str
            The name of the measurement

        Returns
        -------
        str
            Path to the spectra data for a measurement at the site
        """
        return os.path.join(self.specPath, meas)

    def getMeasurementStatPath(self, meas: str) -> str:
        """Get the statistic data path for a measurement at a site

        Parameters
        ----------
        meas : str
            The name of the measurement

        Returns
        -------
        str
            Path to the statistic data for a measurement at the site
        """
        return os.path.join(self.statPath, meas)

    def getSpecdirMaskPath(self, specdir: str) -> str:
        """Get the mask path for a particular set of spectra calculations

        Masks are calculated with relation to sampling frequencies over a whole set of spectra directories.

        Parameters
        ----------
        specdir : str
            The spectra set

        Returns
        -------
        str
            Path to mask data for the spectra set
        """
        return os.path.join(self.maskPath, specdir)

    def getMeasurementTransFuncPath(self, meas: str) -> str:
        """Get the transfer function data path for a measurement at a site

        Parameters
        ----------
        meas : str
            The name of the measurement

        Returns
        -------
        str
            Path to the transfer function data for a measurement at the site
        """
        return os.path.join(self.transFuncPath, meas)

    def getMeasurementStart(self, meas: str) -> datetime:
        """Get the start time of a particular measurement at a site

        This returns the time of the first sample for the measurement

        Parameters
        ----------
        meas : str
            The name of the measurement

        Returns
        -------
        datetime.datetime
            Start time of a measurement at a site
        """
        return self.starts[meas]

    def getMeasurementEnd(self, meas: str) -> datetime:
        """Get the end time of a particular measurement at a site

        This returns the time of the last sample for the measurement

        Parameters
        ----------
        meas : str
            The name of the measurement

        Returns
        -------
        datetime.datetime
            End time of a measurement at a site
        """
        return self.ends[meas]

    def checkMeasurement(self, meas: str) -> bool:
        """Check if site and measurement are part of the project
        
        Parameters
        ----------
        meas : str
            Name of measurement

        Returns
        -------
        bool 
            True if measurement exists
        """
        if meas in self.measurements:
            return True
        else:
            return False

    def view(self, **kwargs) -> Figure:
        """Plot a timeline of the measurements in the site
        
        Parameters
        ----------
        figsize : Tuple (width, height), optional
            The figure size
        plotfonts : Dict, optional
            Fonts to use for plotting
        show : bool, optional
            Boolean flag for showing
        """
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
        import numpy as np
        from resistics.common.plot import getPlotFonts

        figsize = kwargs["figsize"] if "figsize" in kwargs else (15, 8)
        plotFonts = kwargs["plotFonts"] if "plotFonts" in kwargs else getPlotFonts()
        show = kwargs["show"] if "show" in kwargs else True

        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
        # get xlimits
        xStart = self.siteStart - timedelta(days=2)
        xEnd = self.siteEnd + timedelta(days=2)

        numMeas = len(self.measurements)
        height = 0.4
        for idx, meas in enumerate(self.measurements):
            idx = numMeas - idx
            measStart = mdates.date2num(self.starts[meas])
            measEnd = mdates.date2num(self.ends[meas])
            measWidth = measEnd - measStart
            yOffset = 0.5 + height
            rect = Rectangle((measStart, idx - yOffset), measWidth, 2 * height)
            ax.add_patch(rect)
        # x axis formatting
        days = mdates.DayLocator()
        daysFmt = mdates.DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(daysFmt)
        plt.xlim([xStart, xEnd])
        ax.xaxis.grid(True, which="major", ls="--", color="gainsboro")
        plt.xlabel("Date", fontsize=plotFonts["axisLabel"])
        fig.autofmt_xdate(rotation=30)
        for label in ax.get_xticklabels():
            label.set_fontsize(plotFonts["axisTicks"])
        # yaxis formatting
        plt.ylim([0, numMeas])
        yticks = np.arange(1, numMeas + 1)
        ax.set_yticks(yticks)
        ax.set_yticklabels([])
        ax.set_yticks(yticks - 0.5, minor=True)
        ytickLabels = []
        for meas in self.measurements:
            ytickLabels.append("{}\n{:.1f} Hz".format(meas, self.fs[meas]))
        ax.set_yticklabels(
            ytickLabels[::-1], minor=True, fontsize=plotFonts["axisTicks"]
        )
        ax.yaxis.grid(True, which="major", ls="--", color="gainsboro")
        plt.ylabel("Measurements", fontsize=plotFonts["axisLabel"], labelpad=20)
        # title
        plt.title("Site timeline", fontsize=plotFonts["title"])

        fig.tight_layout()
        if show:
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
        textLst.append("Site = {}".format(self.siteName))
        textLst.append("Time data path = {}".format(self.timePath))
        textLst.append("Spectra data path = {}".format(self.specPath))
        textLst.append("Statistics data path = {}".format(self.statPath))
        textLst.append("TransFunc data path = {}".format(self.transFuncPath))
        textLst.append("Site start time = {}".format(self.siteStart))
        textLst.append("Site stop time = {}".format(self.siteEnd))
        sampleFreqText = (
            "No measurement files"
            if len(self.measurements) == 0
            else arrayToString(self.getSampleFreqs())
        )
        textLst.append("Sampling frequencies recorded = {}".format(sampleFreqText))

        # print any measurement directors
        textLst.append(
            "Number of measurement files = {}".format(len(self.measurements))
        )
        if len(self.measurements) > 0:
            textLst.append(
                "Measurement\t\tSample Frequency (Hz)\t\tStart Time\t\tEnd Time"
            )
            for meas in self.measurements:
                textLst.append(
                    "{}\t\t{}\t\t{}\t\t{}".format(
                        meas, self.fs[meas], self.starts[meas], self.ends[meas]
                    )
                )
        else:
            textLst.append("No time data files (measurement files) found")
        return textLst
