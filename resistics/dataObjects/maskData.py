import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime, timedelta
from typing import List, Dict, Set

# import from package#
from resistics.dataObjects.dataObject import DataObject
from resistics.utilities.utilsPrint import arrayToString, blockPrint
from resistics.utilities.utilsFreq import getFrequencyArray
from resistics.utilities.utilsPlotter import getViewFonts


class MaskData(DataObject):
    """Class for holding information about window masking

    Attributes
    ----------
    maskName : str
        The name of the mask
    sampleFreq : float  
        The sampling frequency of the data in Hz
    numLevels : int
        Number of decimation levels
    evalFreq : List[float]
        The evaluation frequnecies
    constraints : Dict
        Constraints index by evaluation frequency, statistic and component
    insideOut : Dict
        Flag stating whether to mask values inside constraints (False) or outside constraints (True), indexed by evaluation frequency, statistic and component
    maskWindows : Dict
        Dictionary for storing the masked windows for each evaluation frequency
    self.stats : List
        Statistics to use in the masking

    Methods
    -------
    __init__(sampleFreq, numLevels, evalFreq, kwargs)
        Initialise maskData
    setStats(stats)
        Set the statistics used in the MaskData
    addConstraint(stat, constraint, insideOut)
        Add constraints for statistic stat for all evaluation frequencies
    addConstraintLevel(stat, constraint, level, insideOut)
        Add constraints for statistic stat for all evaluation frequencies for a single decimation level  
    addConstraintFreq(stat, constraint, level, eIdx, insideOut)
        Add a constraint for statistic stat for a particular evaluation frequency               
    getConstraintFreq(level, eIdx, stat = "")
        Get the constraints for a evaluation frequency provided by the level and the evaluation frequency index and optional request constraints only for a particular statistic
    getMaskWindowsFreq(level, eIdx)
        Get a set of the masked windows for an evaluation frequency identified by decimation level and evaluation frequency index
    prepareDicts()
        Prepare the storage dictionaries given a new set of statistics
    resetMaskWindows()
        Clear the masked windows
    view(level)
        View the masked windows at a particular decimation level
    printList()
        Class status returned as list of strings
    printConstraints()
        Print constraint information to the console
    printConstraintsList()
        Return constraint information as a list of strings
    """

    def __init__(self, sampleFreq: float, numLevels: int, evalFreq, **kwargs):
        """Initialise mask data 
    
        Parameters
        ----------
        sampleFreq : float
            Data sampling frequency in Hz
        numLevels : int
            Number of decimation levels
        evalFreq : np.ndarray
            2-D Matrix, numlevels * evaluations frequencies per level. Contents are evaluations frequencies
        kwargs : Dict
            Two optional arguments: "name" for statistic name and "stats" for the statistics to use
        """

        self.maskName: str = "mask"
        if "name" in kwargs:
            self.maskName = kwargs["name"]
        self.sampleFreq: float = sampleFreq
        self.numLevels: int = numLevels
        self.evalFreq: List[float] = evalFreq
        # constraints and masked windows
        self.constraints: Dict = {}
        self.insideOut: Dict = {}
        self.maskWindows: Dict = {}
        # stats is the stats to use
        self.stats: List[str] = []
        if "stats" in kwargs:
            self.stats = kwargs["stats"]
            self.prepareDicts()

    def setStats(self, stats):
        """Set the statistics to use for the constraints

        Parameters
        ----------
        stats : List[str]
            A list of statistics for which constraints will be provided
        """

        self.stats = stats
        self.prepareDicts()

    def addConstraint(self, stat: str, constraint, insideOut=[]) -> None:
        """Add a constraint for all levels and evaluation frequencies

        Parameters
        ----------
        stat : str
            The statistic to be constrained
        constraint : 
            The constraint parameter
        insideOut : (optional) 
            Inside out parameters
        """

        for ilevel in range(0, self.numLevels):
            self.addConstraintLevel(stat, constraint, ilevel, insideOut)

    def addConstraintLevel(
        self, stat: str, constraint, level: int, insideOut=[]
    ) -> None:
        """Add a constraint for a whole decimation level

        Parameters
        ----------
        stat : str
            The statistic to be constrained
        constraint : 
            The constraint parameter
        level : int
            The decimation level
        insideOut : (optional) 
            Inside out parameters
        """

        for eIdx, eFreq in enumerate(self.evalFreq[level]):
            self.addConstraintFreq(stat, constraint, level, eIdx, insideOut)

    def addConstraintFreq(
        self, stat: str, constraint, level: int, eIdx: int, insideOut=[]
    ) -> None:
        """Add a constraint for an evaluation frequency

        Parameters
        ----------
        stat : str
            The statistic to be constrained
        constraint : 
            The constraint parameter
        level : int
            The decimation level
        eIdx : int
            Evaluation Frequency index
        insideOut : (optional) 
            Inside out parameters
        """

        eFreq = self.evalFreq[level][eIdx]
        # insideOut = []
        # if "insideOut" in kwargs:
        #     insideOut = kwargs["insideOut"]
        for key in constraint:
            self.constraints[eFreq][stat][key] = constraint[key]
            if key in insideOut:
                self.insideOut[eFreq][stat][key] = True
            else:
                self.insideOut[eFreq][stat][key] = False

    def getConstraintFreq(self, level: int, eIdx: int, stat: str = ""):
        """Get constraints for an evaluation frequency

        Parameters
        ----------
        level : int
            The decimation level
        eIdx : int
            Evaluation Frequency index
        stat : str
            The statistic for which to return the constraints

        Returns
        -------
        out : list
            Constraints for the evaluation frequency
        """

        eFreq = self.evalFreq[level][eIdx]
        if stat != "":
            return self.constraints[eFreq][stat]
        return self.constraints[eFreq]

    # def getConstraintFreqStat(self, level: int, eIdx: int, stat: str):
    #     """Get the constraints for an evaluation frequency and a particular statistic

    #     Parameters
    #     ----------
    #     level : int
    #         The decimation level
    #     eIdx : int
    #         The evaluation frequency index

    #     """

    #     eFreq = self.evalFreq[level][eIdx]
    #     return self.constraints[eFreq][stat]

    def getMaskWindowsFreq(self, level: int, eIdx: int) -> Set[int]:
        """Get a set of the masked windows (used by WindowSelector)

        Parameters
        ----------
        level : int
            The decimation level
        eIdx : int
            Evaluation Frequency index

        Returns
        -------
        out : Set[int]
            A set with the indices of the masked windows
        """

        eFreq = self.evalFreq[level][eIdx]
        return self.maskWindows[eFreq]

    def prepareDicts(self) -> None:
        """Prepare dictionaries for storing masking information"""

        for ilevel in range(0, self.numLevels):
            for eFreq in self.evalFreq[ilevel]:
                # empty set for maskWindows to start off with
                # i.e. remove none
                self.maskWindows[eFreq] = set()
                self.constraints[eFreq] = {}
                self.insideOut[eFreq] = {}
                for stat in self.stats:
                    self.constraints[eFreq][stat] = {}
                    self.insideOut[eFreq][stat] = {}

    def resetMaskWindows(self) -> None:
        """Reset/Clear the masked windows"""

        for ilevel in range(0, self.numLevels):
            for eFreq in self.evalFreq[ilevel]:
                # empty set for maskWindows to start off with
                # i.e. remove none
                self.maskWindows[eFreq] = set()

    def view(self, level: int) -> None:
        """Produces a 2d plot with all the masked windows along the bottom for a decimation level
        
        Parameters
        ----------
        level : int
            The decimation level
        """

        evalFreq = self.evalFreq[level]
        levelMasked = set()
        for eFreq in evalFreq:
            levelMasked.update(self.maskWindows[eFreq])
        levelMasked = np.array(sorted(list(levelMasked)))
        global2localMap = {}
        for i in range(0, levelMasked.size):
            global2localMap[levelMasked[i]] = i
        numFreq = np.arange(0, len(evalFreq))
        data = np.zeros(shape=(levelMasked.size, numFreq.size), dtype=int)
        for idx, eFreq in enumerate(evalFreq):
            for gI in self.maskWindows[eFreq]:
                lI = global2localMap[gI]
                data[lI, idx] = idx + 1

        # now plot
        fig = plt.figure()
        plt.pcolor(levelMasked, numFreq, np.transpose(data), cmap=cm.Dark2_r)
        plt.colorbar()
        plt.show()

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """

        textLst: List[str] = []
        textLst.append("Mask name = {}".format(self.maskName))
        textLst.append("Sampling frequency = {:.6f}".format(self.sampleFreq))
        textLst.append(
            "Statistics to use for constraints = {}".format(", ".join(self.stats))
        )
        textLst.append("Evaluation frequencies [Hz]")
        for il in range(0, self.numLevels):
            freqForLevel = self.evalFreq[il]
            for eFreq in freqForLevel:
                textLst.append(
                    "Decimation level = {:d}, Frequency = {:.6f} [Hz], number of masked windows = {:d}".format(
                        il, eFreq, len(self.maskWindows[eFreq])
                    )
                )
        return textLst

    def printConstraints(self) -> None:
        """Print information about the constraints"""

        blockPrint(
            "{} Constraints".format(self.__class__.__name__),
            self.printConstraintsList(),
        )

    def printConstraintsList(self) -> List[str]:
        """Constraint information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """

        textLst = []
        evalFreq = sorted(list(self.constraints.keys()))
        for eFreq in evalFreq:
            textLst.append("Frequency = {:.6f} [Hz]".format(eFreq))
            for stat in self.stats:
                textLst.append("\tStatistic = {}".format(stat))
                # check to see if any constraints
                if len(list(self.constraints[eFreq][stat].keys())) == 0:
                    textLst.append("\tNo constraints for this statistic")
                    continue
                # if there are, print out
                for component in self.constraints[eFreq][stat]:
                    minVal = self.constraints[eFreq][stat][component][0]
                    maxVal = self.constraints[eFreq][stat][component][1]
                    textLst.append(
                        "\t{}\t{}\t{}\t{}".format(
                            component,
                            minVal,
                            maxVal,
                            self.insideOut[eFreq][stat][component],
                        )
                    )
        return textLst
