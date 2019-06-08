import numpy as np
from typing import List, Union, Tuple

# import from package
from resistics.calculators.calculator import Calculator
from resistics.utilities.utilsConfig import loadConfig
from resistics.utilities.utilsEvalFreq import getEvaluationFreqSize
from resistics.utilities.utilsPrint import (
    generalPrint,
    warningPrint,
    blockPrint,
    arrayToString,
    errorPrint,
)


class DecimationParams(Calculator):
    """Class to calculate and hold decimation parameters

    Attributes
    ----------
    sampleFreq : float
        Sampling frequency in Hz.
    divFactor : int
        Minimum division factor when working out decimation factors
    numLevels : int
        Number of decimation levels
    freqPerLevel : int
        Number of frequencies per decimation level
    decFactors : np.ndarray
        Incremental decimation factors
    decFrequencies : np.ndarray
        Sampling frequencies at each decimation level
    evalFreq : np.ndarray 
        Array of evaluation frequencies
    evalFreqPerLevel : np.ndarray
        2-D array (numLevels,freqPerLevel) of evaluation frequencies

    Methods
    -------
    __init__(sampleFreq)
        Initialise with sampling frequency
    getSampleFreqLevel(decLevel)
        Get the sample frequency of the level
    getDecFactor(decLevel)
        Get the decimation factor from level 0 to decLevel
    getIncrementalFactor(decLevel)
        Get the decimation factor from decLevel-1 to decLevel
    getEvalFrequenciesPerLevel(decLevel)
        Returns evaluation frequencies for decLevel
    setFrequencyParams(evalFreq, freqPerLevel, maxLevel)
        Set decimation parameters by evaluation frequencies and frequencies per level
    setDecimationParams(numLevels, freqPerLevel)
        Set decimation parameters by number of levels and frequencies per level
    calcFrequencyParams(numLevels, freqPerLevel)
        Calculate decimation parameters based on number of levels and frequencies per level
    calcDecimationParams(evalFreq, maxLevel, freqPerLevel)
        Calculate decimation paratmers based on evaluation frequencies, frequecies per level and max level
    calcNearestFactor(freq)
        Calculate nearest decimation factor
    printList()
        Class status returned as list of strings
    """

    def __init__(self, sampleFreq: float):
        """Initialise decimation parameters with sampling frequency

        Calculates decimation factors and evaluation frequencies based on defaults

        Parameters
        ----------
        sampleFreq : float
            Sampling frequency
        """

        self.sampleFreq = sampleFreq
        config = loadConfig()
        self.divFactor: int = 2
        self.numLevels: int = config["Decimation"]["numlevels"]
        self.freqPerLevel: int = config["Frequencies"]["perlevel"]
        self.decFactors: np.ndarray
        self.decFrequencies: np.ndarray
        self.evalFreq: np.ndarray
        self.evalFreqPerLevel: np.ndarray

        # calculate some initial values decimation parameters based on defaults
        self.calcFrequencyParams(self.numLevels, self.freqPerLevel)

    def getSampleFreqLevel(self, decLevel: int) -> float:
        """Get sampling frequency for decimation level

        Parameters
        ----------
        decLevel : int
            Decimation level

        Returns
        -------
        out : float
            Sample frequency for decimation level decLevel
        """

        self.checkDecimationLevel(decLevel)
        return self.sampleFreq / self.getDecFactor(decLevel)

    def getDecFactor(self, decLevel: int) -> float:
        """Get decimation factor for decimation level

        Returns decimation factor relative to level 0

        Parameters
        ----------
        decLevel : int
            Decimation level

        Returns
        -------
        out : float
            Decimation factor for level decLevel
        """

        self.checkDecimationLevel(decLevel)
        return self.decFactors[decLevel]

    def getIncrementalFactor(self, decLevel: int) -> int:
        """Get decimation factor for decimation level

        Returns decimation factor relative to decLevel - 1

        Parameters
        ----------
        decLevel : int
            Decimation level

        Returns
        -------
        out : int
            Incremental decimation factor from decLevel - 1
        """

        self.checkDecimationLevel(decLevel)
        if decLevel == 0:
            return int(self.decFactors[decLevel])
        else:
            return int(self.decFactors[decLevel] / self.decFactors[decLevel - 1])

    def getEvalFrequenciesForLevel(self, decLevel: int) -> np.ndarray:
        """Get evaluation frequencies for level decLevel

        Parameters
        ----------
        decLevel : int
            Decimation level

        Returns
        -------
        out : np.ndarray
            Array of evaluation frequencies for level decLevel
        """

        self.checkDecimationLevel(decLevel)
        return self.evalFreqPerLevel[decLevel, :]

    def checkDecimationLevel(self, decLevel: int) -> None:
        """Checks user provided decimation levels and quits if out of bounds

        Parameters
        ----------
        decLevel : int
            Decimation level
        """

        if decLevel < 0:
            self.printError("Decimation level must be greater than 0", quitRun=True)
        if decLevel > self.numLevels - 1:
            self.printError(
                "Decimation level must be less than or equal to {}".format(
                    self.numLevels - 1
                ),
                quitRun=True,
            )

    def setFrequencyParams(
        self, evalFreq: List, freqPerLevel: int, maxLevel: int
    ) -> None:
        """Sets frequency parameters and calculates decimation paramters based on those

        Parameters
        ----------
        evalFreq : List
            Evaluation frequencies
        freqPerLevel : int
            Number of evaluation frequencies per level
        maxLevel : int
            Maximum decimation level
        """

        evalFreq[::-1].sort()  # sort in descending order
        self.calcDecimationParams(evalFreq, freqPerLevel, maxLevel)

    def setDecimationParams(self, numLevels: int, freqPerLevel: int) -> None:
        """Sets number of levels and frequencies per levels to calculate decimation parameters

        Parameters
        ----------
        numLevels : int
            Number of decimation levels
        freqPerLevel : int
            Number of evaluation frequencies per level
        """

        self.calcFrequencyParams(numLevels, freqPerLevel)

    def calcFrequencyParams(self, numLevels: int, freqPerLevel: int) -> None:
        """Calculates evaluation frequencies and decimation parameters

        Uses number of levels and frequencies per levels to auto-calculate evaluation frequencies and then the decimation parameters.

        Parameters
        ----------
        numLevels : int
            Number of decimation levels
        freqPerLevel : int
            Number of evaluation frequencies per level
        """

        numFreq = numLevels * freqPerLevel
        evalFreq = getEvaluationFreqSize(self.sampleFreq, numFreq)
        self.calcDecimationParams(evalFreq, numLevels, freqPerLevel)

    def calcDecimationParams(
        self, evalFreq: Union[np.ndarray, List], maxLevel: int, freqPerLevel: int
    ) -> None:
        """Calculate decimation parameters from evaluation frequencies

        Uses evaluation frequencies, number of frequencies per decimation level and max allowable decimation level to calculate decimation parameters.

        Parameters
        ----------
        evalFreq : List, np.ndarray
            Number of decimation levels
        maxLevel : int
            Maximum allowable number of decimation levels
        freqPerLevel :
            Number of frequencies per level
        """

        evalFreq = np.array(evalFreq)
        # calculating decimation parameters from evaluation frequencies
        maxf = self.sampleFreq / 4
        # find the first evaluation frequency less than or equal to maxf
        fHigh = evalFreq[0]
        for ifreq in range(0, evalFreq.size):
            if evalFreq[ifreq] <= maxf:
                fHigh = evalFreq[ifreq]
                break
        iHigh = evalFreq.tolist().index(fHigh)
        evalFreqSub = evalFreq[iHigh:]
        # calculate number of levels
        numLevels = maxLevel
        # check if enough evaluation frequencies
        if len(evalFreqSub) < freqPerLevel * maxLevel:
            # numLevels = int(math.floor(len(evalFreqSub)/freqPerLevel))
            numLevels = int(np.ceil(1.0 * len(evalFreqSub) / freqPerLevel))

        # do another subslice
        evalFreqSub = evalFreqSub[: numLevels * freqPerLevel]

        # now create an array of evalation frequencies per decimation level
        # evalFreqPerLevel = np.ones(shape=(numLevels, freqPerLevel))
        evalFreqPerLevel = np.ones(shape=(numLevels, freqPerLevel)) * -1
        for ilevel in range(0, numLevels):
            for ifreq in range(0, freqPerLevel):
                if ilevel * freqPerLevel + ifreq >= len(evalFreqSub):
                    break
                evalFreqPerLevel[ilevel, ifreq] = evalFreqSub[
                    ilevel * freqPerLevel + ifreq
                ]

        # now calculate decimation factors
        decFactors = np.ones(shape=(numLevels))
        decFrequencies = np.ones(shape=(numLevels))
        for ilevel in range(0, numLevels):
            decFactors[ilevel], decFrequencies[ilevel] = self.calcNearestFactor(
                evalFreqPerLevel[ilevel][0]
            )

        # finally, set all parameters
        self.evalFreq = evalFreqSub
        self.freqPerLevel = freqPerLevel
        self.numLevels = numLevels
        self.evalFreqPerLevel = evalFreqPerLevel
        self.decFactors = decFactors
        self.decFrequencies = decFrequencies

    def calcNearestFactor(self, freq: float):
        """Calculates nearest decimation factor given frequency

        Parameters
        ----------
        freq : float
            Frequency for which to calculate decimation factor

        Returns
        -------
        fac : int
            The decimation factor
        f : float
            Sampling frequency in Hz given the decimation factor (sampleFreq/decimationFactor)
        """

        # want sampling frequency to be 4 times greater than highest freq
        fsMin = freq * 4
        # set to initial sampling frequency
        f = float(self.sampleFreq)
        fac = 1
        while f > fsMin * self.divFactor:
            f = f / self.divFactor
            fac = fac * self.divFactor
        return fac, f

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """

        textLst = []
        textLst.append("Sampling frequency = {:f}".format(self.sampleFreq))
        textLst.append("Number of decimation levels = {:d}".format(self.numLevels))
        # decimation factors
        for il in range(0, self.numLevels):
            textLst.append(
                "Level = {:d}\tsample freq. [Hz] = {:.6f}\tsample rate [s] = {:.6f}\tdec. factor = {:07d}\tinc. factor = {:d}".format(
                    il,
                    self.decFrequencies[il],
                    1.0 / self.decFrequencies[il],
                    int(self.decFactors[il]),
                    self.getIncrementalFactor(il),
                )
            )
        # evaluation frequencies
        textLst.append("Evaluation frequencies [Hz]")
        for il in range(0, self.numLevels):
            freqForLevel = self.getEvalFrequenciesForLevel(il)
            eFreqStr = arrayToString(freqForLevel)
            textLst.append("Level = {:d}: {}".format(il, eFreqStr))
        return textLst
