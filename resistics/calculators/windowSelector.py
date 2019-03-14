import os
from datetime import date, time, datetime, timedelta
from typing import List, Dict, Set

# import from package
from resistics.calculators.calculator import Calculator
from resistics.calculators.decimationParameters import DecimationParams
from resistics.calculators.windowParameters import WindowParams
from resistics.dataObjects.projectData import ProjectData
from resistics.dataObjects.siteData import SiteData
from resistics.dataObjects.maskData import MaskData
from resistics.ioHandlers.spectrumReader import SpectrumReader
from resistics.ioHandlers.maskIO import MaskIO
from resistics.utilities.utilsPrint import list2ranges, blockPrint
from resistics.utilities.utilsWindow import datetime2gIndex, gIndex2datetime


class WindowSelector(Calculator):
    """Select windows for further processing

    Finds windows for further processing. Given more than one site, WindowSelector will find the shared windows. For example, when processing with a remote reference, WindowSelector will find the shared global windows (i.e. referenced to the project reference time) of the local site and the remote site.

    Shared windows will accept maskData when constraints are to be used. Further, datetime constraints can be added to the selection process, in the case where, for example, only night time data is to be processed.

    For datetime constraints, the order of priorities are:
    1. datetime constraints
    2. date constraints
    3. time constraints 


    Attributes
    ----------
    projData : ProjectData
        A ProjectData instance
    sampleFreq : float
        Sampling frequency of the data
    decParams : DecimationParameters
        A DecimationParameters instance detailing the decimation schemes
    winParams : WindowParameters
        A WindowParameters instance detailing the windowing information 
    sites : List[str]
        List of sites for which to calculate shared windows
    sharedWindows : Dict
        Dictionary to store shared window information between sites. The keys of the dictionary are the decimation levels.
    siteMasks : Dict
        Masks to use for each site
    siteSpecFolders : Dict
        Spectra folders for each site at sampleFreq
    siteSpecReaders : Dict
        SpectraReaders for each site at sampleFreq
    siteSpecRanges : Dict
        Global window ranges for all the spectra files for the sites at sampleFreq
    siteGlobalIndices : Dict
        Global window sets for all the spectra files for the sites at sampleFreq
    specdir : str
        The spectra data to use for calculating shared windows and the subsequent processing
    prepend : str
        The string prepending the spectra data files. This is usually "spectra" and not expected to change
    datetimes : Dict
        User supplied datetime constraints. These are the highest priority in the time constraints. This is a list of constraints for each decimation level.
    dates : Dict
        User supplied date constraints. These are the second highest priority in the time constraints. This is a list of constraints for each decimation level.
    times : Dict
        User supplied time constraints. These are the last priority in the time constraints. This is a list of constraints for each decimation level.
    datetimeConstraints : Dict
        Final combined date and time constraints. This is a list of constraints for each decimation level

    Methods
    -------
    __init__(decParams)
        Initialise window selector with information about the decimation parameters
    """

    def __init__(
        self,
        projData: ProjectData,
        sampleFreq: float,
        decParams: DecimationParams,
        winParams: WindowParams,
        **kwargs
    ) -> None:
        """Initialise window selector

        Parameters
        ----------
        projData : ProjectData
            A ProjectData instance
        sampleFreq : float
            The sampling frequency of the raw time data
        decParams : DecimationParams
            A decimation parameters instance detailing decimaion scheme
        winParams : WindowParams
            A window parameters instance detailing window schemes
        specdir : str, optional
            The spectra directories to use 
        """

        self.projData: ProjectData = projData
        self.sampleFreq: float = sampleFreq
        self.decParams = decParams
        self.winParams = winParams
        self.sites: List = []
        # shared indices
        self.sharedWindows: Dict = {}
        # the masks to use for each site
        self.siteMasks: Dict = {}
        # the spec files for each site at fs
        self.siteSpecFolders: Dict = {}
        self.siteSpecReaders: Dict = {}
        # global indices (ranges and sets)
        self.siteSpecRanges: Dict = {}
        self.siteGlobalIndices: Dict = {}
        # spectra directory information
        self.specdir = kwargs["specdir"] if "specdir" in kwargs else "spectra"
        self.prepend: str = "spectra"
        # time constraints: priority is datetimes > dates > times
        self.datetimes: Dict = {}
        self.dates: Dict = {}
        self.times: Dict = {}
        # final constraints saved in
        self.datetimeConstraints: Dict = {}
        # add a list for each decimation level
        for iDec in range(0, self.decParams.numLevels):
            self.datetimes[iDec] = []
            self.dates[iDec] = []
            self.times[iDec] = []
            self.datetimeConstraints[iDec] = []

    def getSharedWindowsLevel(self, iDec: int) -> Set:
        """Get the shared windows for a decimation level

        Parameters
        ----------
        iDec : int
            The decimation level (0 is the first level)
        
        Returns
        -------
        set
            The shared windows for the decimation level
        """

        return self.sharedWindows[iDec]

    def getNumSharedWindows(self, iDec: int) -> int:
        """Get the number of shared windows for a decimation level

        Parameters
        ----------
        iDec : int
            The decimation level (0 is the first level)
        
        Returns
        -------
        int
            The number of shared windows for the decimation level
        """

        return len(self.sharedWindows[iDec])

    def getWindowsForFreq(self, iDec: int, eIdx: int) -> Set:
        """Get the number of shared windows for a decimation level and evaluation frequency



        Parameters
        ----------
        iDec : int
            The decimation level (0 is the first level)
        eIdx : int
            The evaluation frequency index
        
        Returns
        -------
        set
            The shared windows for evaluation frequency eIdx at decimation level iDec
        """

        sharedWindows = self.getSharedWindowsLevel(iDec)
        # now mask for the particular frequency - mask for each given site
        for s in self.sites:
            for mask in self.siteMasks[s]:
                # remove the masked windows from shared indices
                sharedWindows = sharedWindows - mask.getMaskWindowsFreq(iDec, eIdx)
        return sharedWindows

    def getUnmaskedWindowsLevel(self, iDec: int) -> Set:
        """Get unmasked windows for a decimation level

        Calculate the number of non masked windows for the decimation level. This should speed up processing when constraints are applied.

        Parameters
        ----------
        iDec : int
            The decimation level

        Returns
        -------
        set
            Unmasked windows for the decimation level
        """

        indices = set()
        evalFreq = self.decParams.getEvalFrequenciesForLevel(iDec)
        for eIdx, eFreq in enumerate(evalFreq):
            indices.update(self.getWindowsForFreq(iDec, eIdx))
        return indices

    def getDatetimeConstraints(self):
        """Get the datetime constraints
        
        Returns
        -------
        
        """

        self.calcDatetimeConstraints()
        return self.datetimeConstraints

    def getLevelDatetimeConstraints(self, iDec: int):
        """Get the datetime constraints
        
        Returns
        -------
        
        """

        self.calcDatetimeConstraints()
        return self.datetimeConstraints[iDec]

    def getMasks(self):
        """Get the datetime constraints
        
        Returns
        -------
        
        """

        return self.siteMasks

    def getSpecReaderForWindow(self, site: str, iDec: int, iWin: int):
        """Get the spectrum reader for a window

        Parameters
        ----------
        site : str
            The name of the site to get the spectrum reader for
        iDec : int
            The decimation level
        iWin : int
            The window index
        
        Returns 
        -------
        specFile : str, bool
            The name of the spectra file or False if the window is not found in any spectra file
        specReader : SpectrumReader, bool
            The spectrum reader or False if the window is not found in any spectra file
        """

        specRanges = self.siteSpecRanges[site][iDec]
        specReaders = self.siteSpecReaders[site][iDec]
        for specFile in specRanges:
            if iWin >= specRanges[specFile][0] and iWin <= specRanges[specFile][1]:
                return specFile, specReaders[specFile]

        # if here, no window found
        self.printWarning(
            "Shared window {}, decimation level {} does not appear in any files given the constraints applied".format(
                iWin, iDec
            )
        )
        return False, False

    def getDataSize(self, iDec: int) -> int:
        """Get the spectrum reader for a window

        Parameters
        ----------
        iDec : str
            The decimation level

        Returns 
        -------
        int 
            The data size (number of points in the spectrum) at the decimation level
        """

        # return data size of first file
        site = self.sites[0]
        specReaders = self.siteSpecReaders[site][iDec]
        for sF in specReaders:
            return specReaders[sF].getDataSize()

    def setSites(self, sites) -> None:
        """Set the sites for which to find the shared windows

        Parameters
        ----------
        sites : List[str]
            List of sites
        """

        # first remove repeated sites
        sitesSet = set(sites)
        sites = list(sitesSet)
        # now continue
        self.sites = sites
        for s in self.sites:
            self.siteMasks[s] = []
            self.siteSpecFolders[s] = []
            self.siteSpecReaders[s] = {}
            self.siteSpecRanges[s] = {}
            # use sets to hold gIndices
            # optimised to find intersections
            self.siteGlobalIndices[s] = {}
        # at the end, calculate global indices
        self.calcGlobalIndices()

    # # this is the spectra director
    # def setSpecDir(self, specdir):
    #     specDir = specdir

    # # this is the prepend for the spectra files
    # def setPrepend(self, prepend):
    #     self.prepend = prepend

    def addDatetimeConstraint(self, start: str, stop: str):
        """Add a datetime constraint

        Parameters
        ----------
        start : str
            Datetime constraint start in format %Y-%m-%d %H:%M:%S
        stop : str
            Datetime constraint end in format %Y-%m-%d %H:%M:%S
        """

        numLevels = self.decParams.numLevels
        for iDec in range(0, numLevels):
            self.addLevelDatetimeConstraint(start, stop, iDec)

    def addLevelDatetimeConstraint(self, start: str, stop: str, iDec: int):
        """Add a datetime constraint

        Parameters
        ----------
        start : str
            Datetime constraint start in format %Y-%m-%d %H:%M:%S
        stop : str
            Datetime constraint end in format %Y-%m-%d %H:%M:%S
        iDec : int
            The decimation level
        """

        datetimeStart = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        datetimeStop = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S")
        self.datetimes[iDec].append([datetimeStart, datetimeStop])

    def addDateConstraint(self, dateC: str):
        """Add a date constraint

        Parameters
        ----------
        dateC : str
            Datetime constraint in format %Y-%m-%d
        """

        numLevels = self.decParams.numLevels
        for iDec in range(0, numLevels):
            self.addLevelDateConstraint(dateC, iDec)

    def addLevelDateConstraint(self, dateC: str, iDec: int):
        """Add a date constraint for a decimation level

        Parameters
        ----------
        dateC : str
            Datetime constraint in format %Y-%m-%d
        iDec : int
            The decimation level
        """

        datetimeC = datetime.strptime(dateC, "%Y-%m-%d").date()
        self.dates[iDec].append(datetimeC)

    def addTimeConstraint(self, start: str, stop: str):
        """Add a time constraint. This will recur on every day of recording.

        Parameters
        ----------
        start : str
            Time constraint start in format %H:%M:%S
        stop : str
            Time constraint end in format %H:%M:%S
        """

        numLevels = self.decParams.numLevels
        for iDec in range(0, numLevels):
            self.addLevelTimeConstraint(start, stop, iDec)

    def addLevelTimeConstraint(self, start, stop, iDec):
        """Add a time constraint for a decimation level. This will recur on every day of recording.

        Parameters
        ----------
        start : str
            Time constraint start in format %H:%M:%S
        stop : str
            Time constraint end in format %H:%M:%S
        iDec : int
            The decimation level
        """

        timeStart = datetime.strptime(start, "%H:%M:%S").time()
        timeStop = datetime.strptime(stop, "%H:%M:%S").time()
        self.times[iDec].append([timeStart, timeStop])

    def addWindowMask(self, site: str, maskName: str) -> None:
        """Add a window mask
        
        This is a mask with values for each evaluation frequency.

        Parameters
        ----------
        site : str
            The site for which to search for a mask
        maskName : str
            The name of the mask
        """

        siteData = self.projData.getSiteData(site)
        maskIO = MaskIO(siteData.getSpecdirMaskPath(self.specdir))
        maskData = maskIO.read(maskName, self.sampleFreq)
        self.siteMasks[site].append(maskData)

    def calcSharedWindows(self):
        """Calculate shared windows between sites

        Calculates the shared windows between sites. No masks are applied in this method. Datetime constraints are applied.
        """

        if len(self.sites) == 0:
            self.printWarning(
                "No sites given to Window Selector. At least one site needs to be given."
            )
            return False

        # calculate datetime constraints
        self.calcDatetimeConstraints()

        # initialise the sharedWindows with a set from one site
        sites = self.sites
        siteInit = sites[0]
        numLevels = self.decParams.numLevels
        for iDec in range(0, numLevels):
            self.sharedWindows[iDec] = self.siteGlobalIndices[siteInit][iDec]

        # now for each decimation level, calculate the shared windows
        for iDec in range(0, numLevels):
            for site in self.sites:
                self.sharedWindows[iDec] = self.sharedWindows[iDec].intersection(
                    self.siteGlobalIndices[site][iDec]
                )

        # apply time constraints
        # time constraints should be formulated as a set
        # and then, find the intersection again
        for iDec in range(0, numLevels):
            constraints = self.getLevelDatetimeConstraints(iDec)
            if len(constraints) != 0:
                datetimeIndices = set()
                for dC in constraints:
                    gIndexStart, firstWindowStart = datetime2gIndex(
                        self.projData.refTime,
                        dC[0],
                        self.decParams.getSampleFreqLevel(iDec),
                        self.winParams.getWindowSize(iDec),
                        self.winParams.getOverlap(iDec),
                    )
                    gIndexEnd, firstWindowEnd = datetime2gIndex(
                        self.projData.refTime,
                        dC[1],
                        self.decParams.getSampleFreqLevel(iDec),
                        self.winParams.getWindowSize(iDec),
                        self.winParams.getOverlap(iDec),
                    )
                    gIndexEnd = (
                        gIndexEnd - 1
                    )  # as the function returns the next window starting after time
                    if gIndexEnd < gIndexStart:
                        gIndexEnd = gIndexStart
                    datetimeIndices.update(list(range(gIndexStart, gIndexEnd)))
                    self.printText(
                        "Decimation level = {}. Applying date constraint {} - {}, global index constraint {} - {}".format(
                            iDec, dC[0], dC[1], gIndexStart, gIndexEnd
                        )
                    )
                self.sharedWindows[iDec] = self.sharedWindows[iDec].intersection(
                    datetimeIndices
                )

    def calcGlobalIndices(self) -> None:
        """Find all the global indices for the sites"""

        # get all the spectra files with the correct sampling frequency
        for site in self.sites:
            siteData = self.projData.getSiteData(site)
            timeFilesFs = siteData.getMeasurements(self.sampleFreq)
            # specFiles = self.proj.getSiteSpectraFiles(s)
            specFolders = siteData.spectra
            specFoldersFs = []
            for specFolder in specFolders:
                if specFolder in timeFilesFs:
                    specFoldersFs.append(specFolder)

            self.siteSpecFolders[site] = specFoldersFs

            # for each decimation level
            # loop through each of the spectra folders
            # and find the global indices ranges for each decimation level
            numLevels = self.decParams.numLevels
            for iDec in range(0, numLevels):
                # get the dictionaries ready
                self.siteSpecReaders[site][iDec] = {}
                self.siteSpecRanges[site][iDec] = {}
                self.siteGlobalIndices[site][iDec] = set()
                # loop through spectra folders and figure out global indices
                for specFolder in self.siteSpecFolders[site]:
                    # here, have to use the specdir option
                    specReader = SpectrumReader(
                        os.path.join(siteData.specPath, specFolder, self.specdir)
                    )
                    # here, use prepend to open the spectra file
                    check = specReader.openBinaryForReading(self.prepend, iDec)
                    # if file does not exist, continue
                    if not check:
                        continue
                    self.siteSpecReaders[site][iDec][specFolder] = specReader
                    globalRange = specReader.getGlobalRange()
                    self.siteSpecRanges[site][iDec][specFolder] = globalRange
                    # and save set of global indices
                    self.siteGlobalIndices[site][iDec].update(
                        list(range(globalRange[0], globalRange[1] + 1))
                    )

    def calcDatetimeConstraints(self) -> None:
        """Calculate overall datetime constraints

        Priority order for datetime constraints is: 
        1. datetime constraints
        2. date constraints
        3. time constraints   
        """

        # calculate site dates if required
        siteDates = self.calcSiteDates()

        # datetime constraints are for each decimation level
        numLevels = self.decParams.numLevels
        for iDec in range(0, numLevels):
            # calculate date and time constraints for each level
            # begin with the datetime constraints - these have highest priority
            self.datetimeConstraints[iDec] = self.datetimes[iDec]

            # check to see whether any date and time constraints
            if len(self.dates[iDec]) == 0 and len(self.times[iDec]) == 0:
                continue

            dateConstraints = []
            if len(self.dates[iDec]) != 0:
                # apply time constraints only on specified days
                dateConstraints = self.dates[iDec]
            else:
                dateConstraints = siteDates

            # finally, add the time constraints to the dates
            # otherwise add the whole day
            dateAndTimeConstraints = []
            if len(self.times[iDec]) == 0:
                # add whole days
                for dC in dateConstraints:
                    start = datetime.combine(dC, time(0, 0, 0))
                    stop = datetime.combine(dC, time(23, 59, 59))
                    dateAndTimeConstraints.append([start, stop])
            else:
                # add each time for each day
                for tC in self.times[iDec]:
                    for dC in dateConstraints:
                        start = datetime.combine(dC, tC[0])
                        stop = datetime.combine(dC, tC[1])
                        # check if this goes over a day
                        if tC[1] < tC[0]:
                            # then need to increment the day
                            dCNext = dC + timedelta(days=1)
                            stop = datetime.combine(dCNext, tC[1])
                        # append to constraints
                        dateAndTimeConstraints.append([start, stop])

            # finally, combine datetimes and dateAndTimeConstraints
            self.datetimeConstraints[iDec] = (
                self.datetimeConstraints[iDec] + dateAndTimeConstraints
            )
            self.datetimeConstraints[iDec] = sorted(self.datetimeConstraints[iDec])

    def calcSiteDates(self) -> List[datetime]:
        starts = []
        stops = []
        for site in self.sites:
            siteData = self.projData.getSiteData(site)
            starts.append(siteData.siteStart)
            stops.append(siteData.siteEnd)
        # need all the dates between
        d1 = max(starts).date()
        d2 = min(stops).date()
        if d1 > d2:
            self.printError(
                "A site passed to the window selector does not overlap with any other sites. There will be no shared windows",
                quitRun=True,
            )
        # now with d2 > d1
        siteDates = []
        delta = d2 - d1
        # + 1 because inclusive of stop and start days
        for i in range(delta.days + 1):
            siteDates.append(d1 + timedelta(days=i))
        return siteDates

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """

        textLst = []
        textLst.append("Sampling frequency [Hz] = {:.6f}".format(self.sampleFreq))
        textLst.append("Spectra directory = {}".format(self.specdir))
        textLst.append("Sites = {}".format(", ".join(self.sites)))
        textLst.append("Site information:")
        for site in self.sites:
            textLst = textLst + self.printSiteInfoList(site)
        return textLst

    def printSiteInfo(self):
        """Print out information about the sites"""

        for site in self.sites:
            blockPrint("WindowSelector::site info", self.printSiteInfoList(site))

    def printSiteInfoList(self, site: str) -> List[str]:
        """Return site window information as a list of strings

        Parameters
        ----------
        site : str
            The site name
        
        Returns
        -------
        List[str]
            Site window information as a list of strings
        """

        textLst = []
        textLst.append("Sampling frequency [Hz] = {:.6f}".format(self.sampleFreq))
        textLst.append("Site = {}".format(site))
        textLst.append("Site global index information")
        numLevels = self.decParams.numLevels
        for iDec in range(0, numLevels):
            textLst.append("Decimation Level = {:d}".format(iDec))
            ranges = self.siteSpecRanges
            for sF in sorted(list(ranges[site][iDec].keys())):
                startTime1, endTime1 = gIndex2datetime(
                    ranges[site][iDec][sF][0],
                    self.projData.refTime,
                    self.sampleFreq / self.decParams.getDecFactor(iDec),
                    self.winParams.getWindowSize(iDec),
                    self.winParams.getOverlap(iDec),
                )
                startTime2, endTime2 = gIndex2datetime(
                    ranges[site][iDec][sF][1],
                    self.projData.refTime,
                    self.sampleFreq / self.decParams.getDecFactor(iDec),
                    self.winParams.getWindowSize(iDec),
                    self.winParams.getOverlap(iDec),
                )
                textLst.append(
                    "Measurement file = {}\ttime range = {} - {}\tGlobal Indices Range = {:d} - {:d}".format(
                        sF,
                        startTime1,
                        endTime2,
                        ranges[site][iDec][sF][0],
                        ranges[site][iDec][sF][1],
                    )
                )
        return textLst

    def printSharedWindows(self) -> None:
        """Print out the shared windows"""

        blockPrint("WindowSelector::shared windows", self.printSharedWindowsList())

    def printSharedWindowsList(self) -> List[str]:
        """Shared window information as a list of strings

        Returns
        -------
        List[str]
            Shared window information as a list of strings
        """

        textLst = []
        numLevels = self.decParams.numLevels
        for iDec in range(0, numLevels):
            textLst.append("Decimation Level = {:d}".format(iDec))
            textLst.append(
                "\tNumber of shared windows = {:d}".format(
                    self.getNumSharedWindows(iDec)
                )
            )
            textLst.append(
                "\tShared Window Indices: {}".format(
                    list2ranges(self.getSharedWindowsLevel(iDec))
                )
            )
        textLst.append(
            "NOTE: These are the shared windows at each decimation level. Windows for each evaluation frequency might vary depending on masks"
        )
        return textLst

    def printDatetimeConstraints(self):
        """Print out the datetime constraints"""

        blockPrint(
            "WindowSelector::datetime constraints", self.printDatetimeConstraintsList()
        )

    def printDatetimeConstraintsList(self) -> List[str]:
        """Datetime constraint information as a list of strings

        Returns
        -------
        List[str]
            Datetime constraint information as a list of strings
        """

        textLst = []
        # calculate datetime constraints
        self.calcDatetimeConstraints()
        # populate textLst
        textLst.append("Datetime constraints")
        numLevels = self.decParams.numLevels
        for iDec in range(0, numLevels):
            textLst.append("Decimation Level = {:d}".format(iDec))
            for d in self.getLevelDatetimeConstraints(iDec):
                textLst.append("\tConstraint {} - {}".format(d[0], d[1]))
        return textLst

    def printWindowMasks(self) -> None:
        """Print mask information"""

        blockPrint("WindowSelector::window masks", self.printWindowMasksList())

    def printWindowMasksList(self) -> List[str]:
        """Window mask information as a list of strings

        Returns
        -------
        List[str]
            Window mask information as a list of strings
        """

        textLst = []
        for s in self.sites:
            textLst.append("Site = {}".format(s))
            if len(self.siteMasks[s]) == 0:
                textLst.append("\tNo masks for this site")
            else:
                for mask in self.siteMasks[s]:
                    textLst.append("\tMask = {}".format(mask.maskName))
        return textLst

    def printWindowsForFrequency(self, listwindows=False):
        """Print information about the windows for each evaluation frequency

        Parameters
        ----------
        listwindows : bool
            Boolean flag to actually write out all the windows. Default is False as this takes up a lot of space in the terminal
        """

        blockPrint(
            "WindowSelector::windows for frequency",
            self.printWindowsForFrequencyList(listwindows),
        )

    def printWindowsForFrequencyList(self, listwindows=False) -> List[str]:
        """Information about windows for each evaluation frequency as a list of strings

        Parameters
        ----------
        listwindows : bool
            Boolean flag to actually write out all the windows. Default is False as this takes up a lot of space in the terminal

        Returns
        -------
        List[str]
            Windows for evaluation frequency information as a list of strings
        """

        textLst = []
        for iDec in range(0, self.decParams.numLevels):
            evalFreq = self.decParams.getEvalFrequenciesForLevel(iDec)
            unmaskedWindows = self.getNumSharedWindows(iDec)
            for eIdx, eFreq in enumerate(evalFreq):
                maskedWindows = self.getWindowsForFreq(iDec, eIdx)
                textLst.append(
                    "Evaluation frequency = {:.6f}, shared windows = {:d}, windows after masking = {:d}".format(
                        eFreq, unmaskedWindows, len(maskedWindows)
                    )
                )
                if listwindows:
                    textLst.append("{}".format(list2ranges(maskedWindows)))
        return textLst
