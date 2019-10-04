import os
from datetime import date, time, datetime, timedelta
from typing import List

from resistics.common.base import ResisticsBase
from resistics.common.print import blockPrint, generalPrint, warningPrint
from resistics.project.data import ProjectData
from resistics.statistics.data import StatisticData
from resistics.statistics.io import StatisticIO
from resistics.mask.data import MaskData


class MaskCalculator(ResisticsBase):
    """Calculate window masking given constraints

    Calculate masks for time windows. The window masker reads in statistics and writes out a file with windows which match the given constraints for each statistic. The masked file will then go into the window WindowSelector, which passes on shared windows to the processor.    

    Attributes
    ----------
    proj : Project
        A Project object
    maskData : MaskData
        A MaskData object
    specDir : str
        The spectra directory

    Methods
    -------
    __init__(proj, maskData)
        Initialise with a Project instance and MaskData instance
    clearMaskWindows()
        Clear calculated masks
    applyConstraints(site)
        Apply masking constraints
    printList()
        Class status returned as list of strings
    """

    def __init__(self, projData: ProjectData, maskData: MaskData, **kwargs) -> None:
        """Initialise with project and maskData
    
        Parameters
        ----------
        proj : ProjectData
            The project to which masking should be applied
        maskData : MaskData
            A MaskData instance 
        """
        self.projData = projData
        # stats is the stat to use
        self.maskData = maskData
        self.sampleFreq = maskData.sampleFreq
        self.specdir: str = "spectra"
        if "specdir" in kwargs:
            self.specdir = kwargs["specdir"]

    def clearMaskWindows(self) -> None:
        """Clear mask windows in maskData object """
        self.maskData.maskWindows = {}
        self.maskData.resetMaskWindows()

    def applyConstraints(self, site: str) -> None:
        """Apply the masking constraints to find windows to be excluded
    
        WindowsSelector will later exclude the masked windows before processing

        Parameters
        ----------
        site : str
            The site in the project to mask
        """
        # run through all the statfiles listed
        siteData = self.projData.getSiteData(site)
        measurements = siteData.getMeasurements(self.sampleFreq)
        evalFreqs = self.maskData.evalFreq
        # create a single statisticIO instance to do the reading
        statIO = StatisticIO()
        # loop over stats
        for stat in self.maskData.stats:
            # loop over decimation levels
            for iDec in range(0, self.maskData.numLevels):
                # loop over measurement folders
                for meas in measurements:
                    # try and open the file
                    statIO.setDatapath(
                        os.path.join(
                            siteData.getMeasurementStatPath(meas), self.specdir
                        )
                    )
                    statData = statIO.read(stat, iDec)
                    if not statData:
                        # to next measurement
                        self.printWarning(
                            "No statistic data found for {}, measurement {} and decimation level {}".format(
                                site, meas, iDec
                            )
                        )
                        continue
                    numWindows = statData.numWindows
                    winStats = statData.winStats
                    for iW in range(0, numWindows):
                        # now check each window
                        winVals = statData.getStatLocal(iW)
                        # now loop over evaluation frequencies
                        for eIdx, eFreq in enumerate(evalFreqs[iDec]):
                            constraintCheck = True
                            freqVal = winVals[eIdx]
                            for component in self.maskData.constraints[eFreq][stat]:
                                index = winStats.index(component)
                                componentVal = freqVal[index]
                                test = (
                                    componentVal
                                    > self.maskData.constraints[eFreq][stat][component][
                                        0
                                    ]
                                    and componentVal
                                    < self.maskData.constraints[eFreq][stat][component][
                                        1
                                    ]
                                )
                                if self.maskData.insideOut[eFreq][stat][component]:
                                    test = not test
                                constraintCheck = constraintCheck and test
                            if not constraintCheck:
                                # if the stat fails, add it - this is a list of windows to exclude
                                self.maskData.maskWindows[eFreq].add(
                                    statData.getGlobalIndex(iW)
                                )

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """
        textLst = []
        textLst.append("Sample Frequency [Hz] = {}".format(self.sampleFreq))
        textLst.append("Spectra directory = {}".format(self.specdir))
        textLst.append("Mask Data information")
        textLst = textLst + self.maskData.printList()
        return textLst
