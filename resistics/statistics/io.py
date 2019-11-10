import os
import numpy as np
import math
from datetime import datetime, timedelta
from typing import List, Tuple, Union

from resistics.common.base import ResisticsBase
from resistics.common.io import checkAndMakeDir
from resistics.common.print import arrayToString, breakComment
from resistics.statistics.data import StatisticData


class StatisticIO(ResisticsBase):
    """Class for reading and writing statistic data

    Statistics are calculated out for measurement directories and are located in:
    project -> statData -> site -> meas -> specdir -> statdir

	Attributes
	----------
	dataPath : str
		Path to mask file directory

	Methods
	-------
	__init__(datapath)
		Initialise the statisticIO handler.
	read(statName, inc)
		Read in statistic statName at decimation level inc
	write(statData: StatisticData, inc: int)
		Write out statistic data at decimation level inc
	getFileNames(datapath, filename, inc)
		Get the filenames of the statistic data, the statistic info file and the statistic comments file in location datapath, with base name filename and decimation level inc
    printList()
        Class status returned as list of strings       
	"""

    def __init__(self, datapath="") -> None:
        """Initialise
        
        Parameters
        ----------
        datapath : str, optional
            The path to the statistic data
        """
        self.datapath: str = datapath

    def setDatapath(self, datapath: str) -> None:
        """Set the datapath

        Parameters
        ----------
        datapath : str
            The path to the statistic data
        """
        self.datapath = datapath

    def read(self, statName: str, inc: int) -> Union[None, StatisticData]:
        """Read a statistic file

        Parameters
        ----------
        statName : str
            The statistic name
        inc : int
            The increment, usually the decimation level
        
        Returns
        -------
        StatisticData
            A statistic data object
        """
        statFile, infoFile, commentFile = self.getFileNames(
            self.datapath, statName, inc
        )
        statFile = statFile + ".npy"
        # want a channel ordering
        if not os.path.exists(infoFile) or not os.path.exists(statFile):
            self.printWarning(
                "Unable to find info file {} or stat file {}".format(infoFile, statFile)
            )
            return None
        f = open(infoFile, "r")
        lines = f.readlines()
        f.close()
        refTime = datetime.strptime(lines[0].strip(), "%Y-%m-%d %H:%M:%S.%f")
        sampleFreq = float(lines[1].strip())
        winSize = int(lines[2].strip())
        winOlap = int(lines[3].strip())
        numWindows = int(lines[4].strip())
        winStats = lines[5].strip().split()
        evalFreq = np.fromstring(lines[6].strip(), dtype=float, sep=",")
        dtype = lines[7].strip()
        # now deal with the global indices
        indexInformation = lines[
            9:
        ]  # this is the information about local to global map
        globalIndicesAll = np.empty(shape=(numWindows), dtype=int)
        localIndices = []
        globalIndices = []
        for inInfo in indexInformation:
            inInfo = inInfo.strip()
            if inInfo == "":
                continue
            # now get the local and global indices
            split = inInfo.split("-")
            localIndices.append(int(split[0].strip()))
            globalIndices.append(int(split[1].strip()))
        # now fill the global indices array
        for idx, localI in enumerate(localIndices):
            if idx == len(localIndices) - 1:
                numWindowsFromLocal = numWindows - localI
                globalIndicesAll[localI:] = np.arange(
                    globalIndices[idx], globalIndices[idx] + numWindowsFromLocal
                )
                break
            numWindowsFromLocal = localIndices[idx + 1] - localI
            globalIndicesAll[localI : localI + numWindowsFromLocal] = np.arange(
                globalIndices[idx], globalIndices[idx] + numWindowsFromLocal
            )

        # load the statistics
        stats = np.load(statFile)

        # finally, read the comments file
        comments = []
        if os.path.exists(commentFile):
            f = open(commentFile, "r")
            comments = f.readlines()
            for idx, c in enumerate(comments):
                comments[idx] = c.rstrip()
            f.close()

        # now return a statisticData object
        return StatisticData(
            statName,
            refTime,
            sampleFreq,
            winSize,
            winOlap,
            numWindows=numWindows,
            winStats=winStats,
            evalFreq=evalFreq,
            stats=stats,
            globalIndices=globalIndicesAll,
            comment=comments,
            dtype=dtype,
        )

    def write(self, statData: StatisticData, inc: int) -> None:
        """Write a statistic file

        Parameters
        ----------
        statData : StatisticData
            The statistic data to write out
        inc : int
            The increment, usually the decimation level
        """
        # write the info file - numWindows and channel ordering
        checkAndMakeDir(os.path.join(self.datapath, "{}".format(statData.statName)))
        statFile, infoFile, commentFile = self.getFileNames(
            self.datapath, statData.statName, inc
        )

        # info file
        # want a channel ordering
        f = open(infoFile, "w")
        f.write("{}\n".format(statData.refTime.strftime("%Y-%m-%d %H:%M:%S.%f")))
        f.write("{}\n".format(statData.sampleFreq))
        f.write("{}\n".format(statData.winSize))
        f.write("{}\n".format(statData.winOlap))
        f.write("{:d}\n".format(statData.numWindows))
        f.write("{}\n".format(" ".join(statData.winStats)))
        # now write out the evaluation frequencies
        f.write("{}\n".format(arrayToString(statData.evalFreq)))
        # finally, write out the datatype
        f.write("{}\n".format(statData.dtype))
        # write out the window information
        # only write out when not consecutive
        f.write("Window map: localIndex - globalIndex\n")
        prevI = -10
        for idx, gIndex in enumerate(statData.globalIndices):
            if gIndex != prevI + 1:
                f.write("{:d} - {:d}\n".format(idx, gIndex))
            prevI = gIndex
        f.close()

        # check to save comments
        # only want this on the first increment - no need to rewrite after the others
        if inc < 1:
            import resistics

            with open(commentFile, "w") as f:
                for c in statData.comments:
                    f.write("{}\n".format(c))
                f.write(
                    "Statistic data for statistic {} written to {} on {} using resistics {}\n".format(
                        statData.statName,
                        self.datapath,
                        datetime.now(),
                        resistics.__version__,
                    )
                )
                f.write(breakComment())

        # save binary stat file
        np.save(statFile, statData.stats)

    def getFileNames(
        self, datapath: str, filename: str, inc: int
    ) -> Tuple[str, str, str]:

        """Get the statistic file name

        Parameters
        ----------
        datapath : str
            The datapath to the statistics data
        filename : str
            The filename of the statistics data
        inc : int
            The increment level, usually the decimation level
        
        Returns
        -------
        statFile : str
            Name of the statistics data file
        infoFile : str
            Name of the statistics info file
        commentFile : str
            Name of the statistics comments file
        """
        # inc is meant to be the decimation parameter
        # exclude extension on statfile because .npy added by numpy
        statFile = ""
        infoFile = ""
        if inc < 0:
            statFile = os.path.join(
                datapath, "{}".format(filename), "{}".format(filename)
            )
            infoFile = os.path.join(
                datapath, "{}".format(filename), "{}.info".format(filename)
            )
        else:
            statFile = os.path.join(
                datapath, "{}".format(filename), "{}{:02d}".format(filename, inc)
            )
            infoFile = os.path.join(
                datapath, "{}".format(filename), "{}{:02d}.info".format(filename, inc)
            )
        # name of the comment file
        commentFile = os.path.join(datapath, "{}".format(filename), "comments.txt")
        return statFile, infoFile, commentFile

    def printList(self) -> List[str]:  #
        """Get class information as a list of strings

        Returns
        -------
        List[str]
            Class information as a list of strings
        """
        textLst = []
        if self.datapath == "":
            textLst.append(
                "No datapath given. Set the datapath using the setDatapath function"
            )
        else:
            textLst.append("Datapath = {}".format(self.datapath))
        return textLst

