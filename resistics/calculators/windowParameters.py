import numpy as np
from typing import List

# import from package
from resistics.calculators.calculator import Calculator
from resistics.calculators.decimationParameters import DecimationParams
from resistics.utilities.utilsConfig import loadConfig


class WindowParams(Calculator):
    """WindowParams

    Calculates window sizes for each decimation level based on minimum allowable window size and overlap. Alternatively, users can directly set windowSizes and overlaps. 

    Attributes
    ----------
    decParams : float
        Exposure in seconds.
    minSize : int
        Minimum allowable window size
    minOlap : int
        Minimum allowable overlap
    windowFactor : float
        Window size calculated by sampling frequency / windowFactor to ensure good frequency domain resolution
    overlapFraction : float
        Overlap size as a fraction of window size
    windows : List[int], np.ndarray
        List or array with window sizes for each decimation level
    overlaps : List[int], np.ndarray
        List or array with overlap sizes for each decimation level

    Methods
    -------
    __init__(decParams)
        Initialise with information about the decimation parameters
    getWindowSize(iDec)
        Returns window size for decimation level iDec (starts at 0)
    getOverlap(iDec)
        Returns overlap size for decimation level iDec (starts at 0)
    setWindowParameters(windowSizes, windowOverlaps)
        Directly set window sizes and overlaps
    setMinParams(minSize, minOverlap)
        Set minimum allowable window size and overlap which will be honoured when window sizes by decimation level are automatically calculated
    calcParameters(windowFactor, overlapFraction)
        Calculate window and overlap sizes for each decimation level
    printList()
        Class status returned as list of strings          
    """

    def __init__(self, decParams: DecimationParams) -> None:
        """Initialise with decParams and default window parameters

        Parameters
        ----------
        decParams : int
            Decimation level
        """

        self.decParams = decParams
        config = loadConfig()
        self.minSize = config["Window"]["minwindowsize"]
        self.minOlap = config["Window"]["minoverlapsize"]
        self.windowFactor = config["Window"]["windowfactor"]
        self.overlapFraction = config["Window"]["overlapfraction"]
        self.calcParameters(self.windowFactor, self.overlapFraction)

    def getWindowSize(self, decLevel: int) -> int:
        """Get window size for decimation level

        Parameters
        ----------
        decLevel : int
            Decimation level

        Returns
        -------
        windowSize : int
            Window size for decimation level
        """

        return self.windows[decLevel]

    def getOverlap(self, decLevel: int) -> int:
        """Get window overlap for decimation level

        Parameters
        ----------
        decLevel : int
            Decimation level

        Returns
        -------
        windowOverlap : int
            Window overlap for decimation level
        """

        return self.overlaps[decLevel]

    def setWindowParameters(
        self, windowSizes: List[int], windowOverlaps: List[int]
    ) -> None:
        """Directly set window parameters rather than calculating them out

        If a user is not satisfied with the window size calculation, window parameters can be defined explicitly here

        Parameters
        ----------
        windowSizes : List[int]
            List of window sizes for each decimation level
        windowOverlaps : List[int]
            List of window overlaps for each decimation level
        """

        if (
            len(windowSizes) != self.decParams.numLevels
            or len(windowOverlaps) != self.decParams.numLevels
        ):
            print(
                "Error: not enough window sizes given. Must be equal to number of decimation levels"
            )
            return
        self.windows = windowSizes
        self.overlaps = windowOverlaps

    def setMinParams(self, minSize: int, minOlap: int) -> None:
        """Change default minimum window size and overlap parameters

        These values will be honoured when automatically calculating window sizes in calcParameters

        Parameters
        ----------
        minSize : int
            List of window sizes for each decimation level
        minOlap : int
            List of window overlaps for each decimation level
        """

        self.minSize = minSize
        self.minOlap = minOlap
        self.calcParameters(self.windowFactor, self.overlapFraction)

    def calcParameters(self, windowFactor, overlapFraction) -> None:
        """Calculate window size and overlap for each decimation level based on minimum allowable parameters (minSize, minOlap)

        The window and overlap sizes (number of samples) are calculated based on the following:

        Window size = frequency at decimation level / windowFactor
        Overlap size = Window size * overlapFraction

        The window size is calculated based on the sampling frequency of the decimation level to ensure good frequency domain resolution


        Parameters
        ----------
        windowFactor : float
            Window size is calculated as frequency at decimation level / windowFactor
        overlapFraction : float
            Overlap size as a fraction of the window size
        """

        self.windows = np.ones(shape=(self.decParams.numLevels), dtype=int)
        self.overlaps = np.ones(shape=(self.decParams.numLevels), dtype=int)
        decFreq = self.decParams.decFrequencies
        for il in range(0, self.decParams.numLevels):
            self.windows[il] = int(decFreq[il] / windowFactor)
            if self.windows[il] < self.minSize:
                self.windows[il] = self.minSize
            self.overlaps[il] = int(self.windows[il] * overlapFraction)
            if self.overlaps[il] < self.minOlap:
                self.overlaps[il] = self.minOlap

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """

        textLst = []
        textLst.append(
            "Number of decimation levels = {:d}".format(self.decParams.numLevels)
        )
        decFrequencies = self.decParams.decFrequencies
        for il in range(0, self.decParams.numLevels):
            textLst.append(
                "Level = {:d}, sample freq. [Hz] = {:.6f}, sample rate [s] = {:6f}".format(
                    il, decFrequencies[il], 1.0 / decFrequencies[il]
                )
            )
            textLst.append(
                "Window size = {:d}, window duration [s] = {:f}".format(
                    self.windows[il], (self.windows[il] - 1) / decFrequencies[il]
                )
            )
            textLst.append(
                "Window overlap = {:d}, overlap duration [s] = {:f}".format(
                    self.overlaps[il], (self.overlaps[il] - 1) / decFrequencies[il]
                )
            )
        return textLst
