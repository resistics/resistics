import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime, timedelta
from typing import List, Dict, Any
from configobj import ConfigObj

from resistics.common.base import ResisticsBase
from resistics.common.print import listToString
from resistics.config.io import loadConfig


class ConfigData(ResisticsBase):
    """Class for holding cofiguration options

    Attributes
    ----------
    configFile : str
        Path to the configuration file
    configParams : ConfigObj
        ConfigObj object to parse configuration information

    Methods
    -------
    __init__(configFile)
        Initialise configuration data
    configure()
        Configure all the options
    printList()
        Class information as a list of strings
    """

    def __init__(self, configFile: str = "") -> None:
        """Initialise with default configuration options
        
        Parameters
        ----------
        configFile : str
            Path to the configuration file
        """

        self.configFile: str = configFile
        self.configParams = loadConfig(self.configFile)
        # options dictionary
        self.flags: Dict[str, bool] = {}
        self.configure()

    def configure(self) -> None:
        """Set the configuration parameters
        
        This sets a couple of flags in the configuration to regarding whether a custom set of evaluation frequencies are being used or a custom window size definition
        """

        # frequencies
        self.flags["customfrequencies"] = False
        if len(self.configParams["Frequencies"]["frequencies"]) > 0:
            # custom frequencies have been provided
            self.flags["customfrequencies"] = True
            # check to make sure appropriate number of frequencies provided
            numUserFreqs = len(self.configParams["Frequencies"]["frequencies"])
            numExpectedFreqs = (
                self.configParams["Decimation"]["numlevels"]
                * self.configParams["Frequencies"]["perlevel"]
            )
            if numUserFreqs != numExpectedFreqs:
                self.printError(
                    "Custom frequencies provided. Please ensure that the number of frequencies is equal to numlevels * perlevel",
                    quitrun=True,
                )

        # windows
        self.flags["customwindows"] = False
        numUserWindows = len(self.configParams["Window"]["windowsizes"])
        numUserOverlaps = len(self.configParams["Window"]["overlapsizes"])
        if numUserWindows > 0 and numUserOverlaps > 0:
            self.flags["customwindows"] = True
            numlevels = self.configParams["Decimation"]["numlevels"]
            if numUserWindows != numlevels or numUserOverlaps != numlevels:
                self.printError(
                    "When supplying custom window sizes, the number of window sizes provided should be equal to the number of decimation levels (numlevels)",
                    quitrun=True,
                )

    def getSpectraCores(self) -> int:
        """Returns the number of cores to run specrta calculations on
        
        There is a global ncores parameter and one in the Spectra section. The Spectra one takes precedent when they are both set. 

        Returns
        -------
        ncores : int
            The number of cores to run spectra calculations on
        """
        if self.configParams["Spectra"]["ncores"] != -1:
            return self.configParams["Spectra"]["ncores"]
        if self.configParams["ncores"] != -1:
            return self.configParams["ncores"]
        return 0

    def getStatisticCores(self) -> int:
        """Returns the number of cores to run statistic calculations on
        
        There is a global ncores parameter and one in the Statistics section. The Statistics one takes precedent when they are both set. 

        Returns
        -------
        ncores : int
            The number of cores to run statistic calculations on
        """
        if self.configParams["Statistics"]["ncores"] != -1:
            return self.configParams["Statistics"]["ncores"]
        if self.configParams["ncores"] != -1:
            return self.configParams["ncores"]
        return 0

    def getSolverCores(self) -> int:
        """Returns the number of cores to run solver calculations on
        
        There is a global ncores parameter and one in the Solver section. The Solver one takes precedent when they are both set. 

        Returns
        -------
        ncores : int
            The number of cores to run solver calculations on
        """
        if self.configParams["Solver"]["ncores"] != -1:
            return self.configParams["Solver"]["ncores"]
        if self.configParams["ncores"] != -1:
            return self.configParams["ncores"]
        return 0

    def getConfigComment(self) -> str:
        """Returns a string to add as a comment to data

        Returns
        -------
        str 
            Comment string to add to data
        """

        if self.configFile == "":
            return "Using default configuration"
        else:
            return "Using configuration with name {} in configuration file {}".format(
                self.configParams["name"], self.configFile
            )

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """

        textLst: List[str] = []
        if self.configFile == "":
            textLst.append("Configuration file = Default configuration")
        else:
            textLst.append("Configuration file = {:s}".format(self.configFile))
        textLst.append("Configuration name = {:s}".format(self.configParams["name"]))

        textLst.append("Flags:")
        for key, value in self.flags.items():
            textLst.append("{:s} = {}".format(key, value))

        textLst.append("Configuration Parameters:")
        textLst.append("Name = {:s}".format(self.configParams["name"]))
        textLst.append("ncores = {:d}".format(self.configParams["ncores"]))
        textLst = textLst + self.printListSection("Calibration")
        textLst = textLst + self.printListSection("Decimation")
        textLst = textLst + self.printListSection("Frequencies")
        textLst = textLst + self.printListSection("Window")
        textLst = textLst + self.printListSection("Spectra")
        textLst = textLst + self.printListSection("Statistics")
        textLst = textLst + self.printListSection("Solver")
        return textLst

    def printListSection(self, section: str) -> List[str]:
        """Configuration section information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """

        textLst: List[str] = []
        textLst.append("{:s}:".format(section))
        for key, value in self.configParams[section].items():
            textLst.append("\t{:s} = {}".format(key, value))
        defaultOptions = "No defaults used"
        if len(self.configParams[section].defaults) > 0:
            defaultOptions = listToString(self.configParams[section].defaults)
        textLst.append("\tDefaulted options = {:s}".format(defaultOptions))
        return textLst
