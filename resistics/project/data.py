import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Dict, Union, ClassVar

from resistics.common.base import ResisticsBase
from resistics.common.print import arrayToString, listToString
from resistics.common.io import (
    checkAndMakeDir,
    checkDirExistence,
    getDirsInDirectory,
    getDataDirsInDirectory,
    getFilesInDirectory,
)
from resistics.config.data import ConfigData
from resistics.site.data import SiteData


class ProjectData(ResisticsBase):
    """Project class for storing project attributes

    Attributes
    ----------
    projectFile : str
        Path to project file
    timePath : str
        Path to time series data
    specPath : str
        Path to spectra data
    statPath : str
        Path to statistics
    transPath : str
        Path to transfer function data
    calPath : str
        Path to calibration data
    imagePath : str
        Path to project images
    refTime : datetime.datetime
        The project reference time
    projStart : datetime.datetime
        The first time sample in the data
    projEnd : datetime.datetime
        The last time sample in the data
    sites : List[str]
        List of sites in the project
    siteData : Dict[str, SiteData]
        Site data for each site        
    calFiles : List
        A list of calibration files
    config : ConfigData
        Configuration data

    Methods
    -------
    __init__()
        Initialise project with default values
    refresh()
        Refresh the project
    checkDirectories()
        Check the project directories to make sure they exist
    getCalInfo()
        Get calibration file information
    getSiteInfo()   
        Get information about all the sites and store site data information
    getSites(fs=None)
        Get a list of all sites, or those with data at a sampling frequency of fs Hz 
    getNumSites()
        Get the total number of sites
    getSampleFreqs()
        Get a list of all the unique sample frequencies in the project
    getSiteConcurrent(site)
        Get a list of sites that were recording concurrently to site
    getSiteData(site)
        Get site data for a single site
    checkSite(site)
        Check a site exists
    view()
        View the project timeline
    printList()
        Class status returned as list of strings   
    """

    def __init__(
        self,
        projectFile: str,
        refTime: datetime,
        calPath: str,
        timePath: str,
        specPath: str,
        statPath: str,
        maskPath: str,
        transFuncPath: str,
        imagePath: str,
        config=None,
    ):
        # project file and reference time
        self.projectFile: str = projectFile
        self.refTime: datetime = refTime
        # paths to various parts of the project
        self.calPath: str = calPath
        self.timePath: str = timePath
        self.specPath: str = specPath
        self.statPath: str = statPath
        self.maskPath: str = maskPath
        self.transFuncPath: str = transFuncPath
        self.imagePath: str = imagePath
        self.projStart: datetime = datetime.now()
        self.projEnd: datetime = datetime.now()
        # site information
        self.sites: List[str] = []
        self.siteData: Dict[str, SiteData] = {}
        # cal files list
        self.calFiles: List = []
        # configuration options
        self.config = config
        if self.config is None:
            self.config = ConfigData()
        # refresh
        self.refresh()

    def refresh(self) -> None:
        """Refresh the project

        If new time, spectra or other data directories have been added to the project, these will not show up in the project information until the project has been refreshed (i.e. all the information read again from the disc)
        """
        self.checkDirectories()
        self.getCalInfo()
        self.getSiteInfo()
        # update start and end dates for the project
        startDates = []
        endDates = []
        for site in self.sites:
            startDates.append(self.getSiteData(site).siteStart)
            endDates.append(self.getSiteData(site).siteEnd)
        if len(self.sites) > 0:
            self.projStart = min(startDates)
            self.projEnd = max(endDates)

    def checkDirectories(self) -> None:
        """Check to see if project directories exist and if not, make them"""
        # data directories
        checkAndMakeDir(self.timePath)
        checkAndMakeDir(self.specPath)
        checkAndMakeDir(self.statPath)
        checkAndMakeDir(self.maskPath)
        checkAndMakeDir(self.transFuncPath)
        # calibration directories
        checkAndMakeDir(self.calPath)
        # image directory
        checkAndMakeDir(self.imagePath)

    def getCalInfo(self):
        """Get a list of all the calibration files in the project"""
        self.calFiles = getFilesInDirectory(self.calPath)

    def getSiteInfo(self) -> None:
        """Get information about sites"""
        self.sites = getDirsInDirectory(self.timePath)
        self.sites = sorted(self.sites)
        for site in self.sites:
            self.siteData[site] = SiteData(
                site,
                self.timePath,
                self.specPath,
                self.statPath,
                self.maskPath,
                self.transFuncPath,
            )

    def getSites(self, sampleFreq: Union[None, float, int] = None):
        """Get list of sites

        Optionally filtered to those with measurements recorded at fs Hz

        Parameters
        ----------
        sampleFreq : float, int, optional
            Sampling frequency Hz

        Returns
        -------
        List[str]
            List of site names
        """
        if sampleFreq is None:
            return self.sites
        # fs is supplied
        filteredSites = []
        for site in self.sites:
            if sampleFreq in self.getSiteData(site).getSampleFreqs():
                filteredSites.append(site)
        return filteredSites

    def getNumSites(self) -> int:
        """Number of sites in the project

        Returns
        -------
        int
            Number of sites
        """
        return len(self.sites)

    def getSampleFreqs(self) -> List[float]:
        """List of all the sampling frequencies found in the project

        Returns
        -------
        List[float]
            List of sampling frequencies in Hz
        """
        sampleFreq = set()
        for site in self.sites:
            sampleFreq.update(self.getSiteData(site).getSampleFreqs())
        return sorted(list(sampleFreq))

    def getSiteConcurrent(self, site: str) -> List[str]:
        """Find sites which were occupied at the same time as a chosen site

        This lists all sites that were occupied at the same time as the user supplied site. It can be useful in cases where multi-site processing will be performed

        Parameters
        ----------
        site : str
            The name of the site

        Returns
        -------
        List[str]
            List of sites which were recording concurrently
        """
        siteStart = self.getSiteData(site).siteStart
        siteEnd = self.getSiteData(site).siteEnd
        concSites = []
        for s in self.sites:
            if s == site:
                continue
            start = self.getSiteData(s).siteStart
            end = self.getSiteData(s).siteEnd
            # check that start time is before end time of site and that end time is after start time of site
            if (start < siteEnd) and (end > siteStart):
                concSites.append(s)
        return concSites

    def getSiteData(self, site) -> Union[SiteData, bool]:
        """Get site data for site

        Parameters
        ----------
        site : str
            Site name

        Returns
        -------
        SiteData
            Site data for site or False if the site is not found
        """
        if site not in self.sites:
            self.printWarning("Site {} does not exist in the project".format(site))
            return False
        return self.siteData[site]

    def checkSite(self, site: str) -> bool:
        """Check if site is part of the project
        
        Parameters
        ----------
        site : str
            Name of the site
        """
        if site in self.sites:
            return True
        else:
            return False

    def createSite(self, site: str) -> bool:
        """Creates a site folder in the time data path

        Parameters
        ----------
        site : str
            Name of the site
        """
        sitePath = os.path.join(self.timePath, site)
        chk = checkDirExistence(sitePath)
        if chk:
            self.printWarning(
                "Site {} already exists in project time data folder".format(site)
            )
            return False
        checkAndMakeDir(sitePath)
        return True

    def view(self, **kwargs) -> Figure:
        """Plot a timeline of the project
        
        Parameters
        ----------
        figsize : Tuple (width, height), optional
            The figure size
        plotfonts : Dict, optional
            Fonts to use for plotting
        show : bool, optional
            Boolean flag fow showing plot. Default is True.
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
        xStart = self.refTime - timedelta(days=2)
        xEnd = self.projEnd + timedelta(days=2)

        numSites = len(self.sites)
        height = 0.4
        for idx, site in enumerate(self.sites):
            idx = numSites - idx
            siteStart = mdates.date2num(self.siteData[site].siteStart)
            siteEnd = mdates.date2num(self.siteData[site].siteEnd)
            siteWidth = siteEnd - siteStart
            yOffset = 0.5 + height
            rect = Rectangle((siteStart, idx - yOffset), siteWidth, 2 * height)
            ax.add_patch(rect)
        # plot the reference line
        plt.plot(
            [self.refTime, self.refTime], [0, numSites], color="lightcoral", ls="-"
        )
        plt.text(
            self.refTime, 0.02, "  Reference Time", fontsize=plotFonts["axisTicks"]
        )
        # x axis formatting
        weeks = mdates.WeekdayLocator()
        weeksFmt = mdates.DateFormatter("%Y-%m-%d")
        days = mdates.DayLocator()
        ax.xaxis.set_major_locator(weeks)
        ax.xaxis.set_major_formatter(weeksFmt)
        ax.xaxis.set_minor_locator(days)
        plt.xlim([xStart, xEnd])
        ax.xaxis.grid(True, which="minor", ls="--", color="gainsboro")
        plt.xlabel("Date", fontsize=plotFonts["axisLabel"])        
        fig.autofmt_xdate(rotation=30)
        for label in ax.get_xticklabels():
            label.set_fontsize(plotFonts["axisTicks"])        
        # y axis formatting
        plt.ylim([0, numSites])
        yticks = np.arange(1, numSites + 1)
        ax.set_yticks(yticks)
        ax.set_yticklabels([])
        ax.set_yticks(yticks - 0.5, minor=True)
        ax.set_yticklabels(
            self.sites[::-1], minor=True, fontsize=plotFonts["axisTicks"]
        )
        ax.yaxis.grid(True, which="major", ls="--", color="gainsboro")
        plt.ylabel("Sites", fontsize=plotFonts["axisLabel"], labelpad=20)
        # title
        plt.title("Project Timeline", fontsize=plotFonts["title"])

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
        textLst: List[str] = []
        textLst.append("Time data path = {}".format(self.timePath))
        textLst.append("Spectra data path = {}".format(self.specPath))
        textLst.append("Statistics data path = {}".format(self.statPath))
        textLst.append("Mask data path = {}".format(self.maskPath))
        textLst.append("TransFunc data path = {}".format(self.transFuncPath))
        textLst.append("Calibration data path = {}".format(self.calPath))
        textLst.append("Images data path = {}".format(self.imagePath))
        textLst.append(
            "Reference time = {}".format(self.refTime.strftime("%Y-%m-%d %H:%M:%S"))
        )
        textLst.append(
            "Project start time = {}".format(
                self.projStart.strftime("%Y-%m-%d %H:%M:%S.%f")
            )
        )
        textLst.append(
            "Project stop time = {}".format(
                self.projEnd.strftime("%Y-%m-%d %H:%M:%S.%f")
            )
        )
        textLst.append("Project found {} sites:".format(self.getNumSites()))
        for site in self.sites:
            textLst.append(
                "{}\t\tstart: {}\tend: {}".format(
                    site,
                    self.getSiteData(site).siteStart,
                    self.getSiteData(site).siteEnd,
                )
            )
        textLst.append(
            "Sampling frequencies found in project (Hz): {}".format(
                listToString(self.getSampleFreqs())
            )
        )
        return textLst