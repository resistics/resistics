import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime, timedelta
from typing import List, Dict, Set

# import from package
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
    stats : List
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
        self, stat: str, constraint, declevel: int, insideOut=[]
    ) -> None:
        """Add a constraint for a whole decimation level

        Parameters
        ----------
        stat : str
            The statistic to be constrained
        constraint : 
            The constraint parameter
        declevel : int
            The decimation level
        insideOut : (optional) 
            Inside out parameters
        """

        for eIdx, eFreq in enumerate(self.evalFreq[declevel]):
            self.addConstraintFreq(stat, constraint, declevel, eIdx, insideOut)

    def addConstraintFreq(
        self, stat: str, constraint, declevel: int, eIdx: int, insideOut=[]
    ) -> None:
        """Add a constraint for an evaluation frequency

        Parameters
        ----------
        stat : str
            The statistic to be constrained
        constraint : 
            The constraint parameter
        declevel : int
            The decimation level
        eIdx : int
            Evaluation Frequency index
        insideOut : (optional) 
            Inside out parameters
        """

        eFreq = self.evalFreq[declevel][eIdx]
        # insideOut = []
        # if "insideOut" in kwargs:
        #     insideOut = kwargs["insideOut"]
        for key in constraint:
            self.constraints[eFreq][stat][key] = constraint[key]
            if key in insideOut:
                self.insideOut[eFreq][stat][key] = True
            else:
                self.insideOut[eFreq][stat][key] = False

    def getConstraintFreq(self, declevel: int, eIdx: int, stat: str = ""):
        """Get constraints for an evaluation frequency

        Parameters
        ----------
        declevel : int
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

        eFreq = self.evalFreq[declevel][eIdx]
        if stat != "":
            return self.constraints[eFreq][stat]
        return self.constraints[eFreq]

    def getMaskWindowsFreq(self, declevel: int, eIdx: int) -> Set[int]:
        """Get a set of the masked windows (used by WindowSelector)

        Parameters
        ----------
        declevel : int
            The decimation level
        eIdx : int
            Evaluation Frequency index

        Returns
        -------
        out : Set[int]
            A set with the indices of the masked windows
        """

        eFreq = self.evalFreq[declevel][eIdx]
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

    def view(self, declevel: int = 0, **kwargs) -> plt.figure:
        """Produces a 2-D plot with all the masked windows along the bottom for a decimation level
        
        Parameters
        ----------
        declevel : int, optional
            The decimation level. Default is 0.
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts            
        """

        fig: plt.figure = (
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(10, 10))
        )
        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()

        evalFreq = self.evalFreq[declevel]
        numFreq = len(evalFreq)
        freqIndex = range(0, numFreq)
        levelMasked: Set = set()
        numMasked: List[int] = list()
        freqLabel: List[str] = list()
        for eFreq in evalFreq:
            levelMasked.update(self.maskWindows[eFreq])
            numMasked.append(len(self.maskWindows[eFreq]))
            freqLabel.append("{:.5f}".format(eFreq))
        
        numMaskedTotal = len(levelMasked)
        levelMasked = np.array(sorted(list(levelMasked)))
        global2localMap = {}
        for idx in range(0, levelMasked.size):
            global2localMap[levelMasked[idx]] = idx
        # make array
        data = np.zeros(shape=(levelMasked.size, numFreq), dtype=int)
        for idx, eFreq in enumerate(evalFreq):
            for gI in self.maskWindows[eFreq]:
                lI = global2localMap[gI]
                data[lI, idx] = idx + 1

        st = fig.suptitle(
            "Masked windows for decimation level: {}".format(declevel),
            fontsize=plotfonts["suptitle"],
        )
        st.set_y(0.98)

        # now plot
        ax = plt.subplot(2, 1, 1)
        cmap = cm.get_cmap("Paired", numFreq)
        cmap.set_under("white")
        plt.pcolor(np.transpose(data), cmap=cmap, vmin=1, vmax=numFreq + 1)
        ax.set_yticks(np.array(freqIndex) + 0.5)
        ax.set_yticklabels(freqLabel)
        plt.ylabel(
            "Masked windows for decimation level", fontsize=plotfonts["axisLabel"]
        )
        plt.ylabel("Evaluation Frequency [Hz]", fontsize=plotfonts["axisLabel"])
        plt.title(
            "Relative masked windows for decimation level {}".format(declevel),
            fontsize=plotfonts["title"],
        )
        # set tick sizes
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(plotfonts["axisTicks"])
        # colourbar
        cb = plt.colorbar()
        cb.ax.set_title("Masked\nwindows")
        cb.set_ticks(np.arange(1, numFreq + 1) + 0.5)
        cb.set_ticklabels(freqLabel)

        # bar graph
        ax = plt.subplot(2, 1, 2)
        # add the total number masked
        freqIndex = range(0, numFreq + 1)
        numMasked.append(numMaskedTotal)
        freqLabel.append("Total masked for\n decimation level")
        plt.bar(freqIndex, numMasked)
        ax.set_xticks(freqIndex)
        ax.set_xticklabels(freqLabel)
        plt.xlabel("Evaluation Frequency [Hz]", fontsize=plotfonts["axisLabel"])
        plt.ylabel("Number of masked windows", fontsize=plotfonts["axisLabel"])
        plt.title(
            "Number of masked windows per evaluation frequency",
            fontsize=plotfonts["title"],
        )
        # set tick sizes
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(plotfonts["axisTicks"])

        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
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
