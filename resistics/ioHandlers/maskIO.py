import os
import numpy as np
import math
from datetime import datetime, timedelta
from typing import List, Tuple

# import from package
from resistics.ioHandlers.ioHandler import IOHandler
from resistics.dataObjects.maskData import MaskData
from resistics.utilities.utilsIO import checkAndMakeDir, fileFormatSampleFreq
from resistics.utilities.utilsPrint import listToString


class MaskIO(IOHandler):
    """Class for reading and writing maskData

    Masks are referenced to sampling frequencies rather than particular measurements. The idea if that masks are calculated out for all the data and later, the data can be constrained either using date and time constraints or by using window masks based on statistics.

    Mask files are located in:
    project -> maskData -> site -> sample frequency -> specdir

	Attributes
	----------
	datapath : str
		Path to mask file directory

	Methods
	-------
	__init__(datapath)
		Initialise the MaskIO object.
	read(maskName, sampleFreq)
		Read in maskName for sampleFreq from datapath        
	write(maskData)
		Write out maskData object to datapath.
	getFileNames(makeName, sampleFreq)
		The filename for a maskFile given a maskName and sampling frequency
    printList()
        Class status returned as list of strings       
	"""

    def __init__(self, datapath: str = "") -> None:
        """Initialise

        Parameters
        ----------
        datapath : str, optional
            Path to mask file directory
        """

        self.datapath: str = datapath

    def read(self, maskName: str, sampleFreq: float) -> MaskData:
        """Read in maskData from a file defined by maskName and sampleFreq

        Parameters
        ----------
        maskName : MaskData
            MaskData object
        sampleFreq : float
            The sampling frequency of the data

        Returns
        -------
        maskData : MaskData
            The MaskData object
        """

        # read the window file
        infoName, winName = self.getFileNames(maskName, sampleFreq)
        infoFile = open(infoName, "r")
        lines = infoFile.readlines()
        infoFile.close()
        # this is all passed into the constructor
        sampleFreq = float(lines[0].strip())
        numLevels = int(lines[1].strip())
        evalFreq = []
        for iL in range(0, numLevels):
            evalFreq.append(
                list(np.fromstring(lines[2 + iL].strip(), dtype=float, sep=","))
            )
        # read in the stats
        stats = lines[2 + numLevels].strip().split(",")
        for idx, s in enumerate(stats):
            stats[idx] = stats[idx].strip()
        # now create a MaskData object
        maskData = MaskData(sampleFreq, numLevels, evalFreq, name=maskName, stats=stats)
        lines = lines[3 + numLevels :]
        # now read in the statistics
        evalFreqSorted = sorted(list(maskData.constraints.keys()))
        eIdx = -1
        for l in lines:
            if "Frequency" in l:
                eIdx = eIdx + 1
            elif "Statistic" in l:
                stat = l.strip().split("=")[1]
                stat = stat.strip()
            else:
                split = (
                    l.strip().split()
                )  # component is 0, min is 1, max is 2, inout is 3
                maskData.constraints[evalFreqSorted[eIdx]][stat][split[0]] = [
                    float(split[1]),
                    float(split[2]),
                ]
                inOut = False
                if split[3] == "True":
                    inOut = True
                maskData.insideOut[evalFreqSorted[eIdx]][stat][split[0]] = inOut
        # now read the window mask file
        winMaskArray = np.load(winName + ".npy")
        for eIdx, eFreq in enumerate(evalFreqSorted):
            maskData.maskWindows[eFreq] = set(winMaskArray[eIdx])
            # remove -1 from the set
            maskData.maskWindows[eFreq] = maskData.maskWindows[eFreq] - set([-1])
        # now want to return as MaskData object
        return maskData

    def write(self, maskData: MaskData) -> None:
        """Write the maskData out to datapath

        Mask data is saved as a numpy binary object

        Parameters
        ----------
        maskData : MaskData
            MaskData object
        """

        infoName, winName = self.getFileNames(maskData.maskName, maskData.sampleFreq)
        infoFile = open(infoName, "w")
        # first write out constraints
        infoFile.write("{:.9f}\n".format(maskData.sampleFreq))
        infoFile.write("{}\n".format(maskData.numLevels))
        for iL in range(0, maskData.numLevels):
            infoFile.write("{}\n".format(listToString(maskData.evalFreq[iL])))
        infoFile.write("{}\n".format(", ".join(maskData.stats)))
        # now write out the data file
        # first get a sorted list of all the evaluations frequencies to loop through
        evalFreq = sorted(list(maskData.constraints.keys()))
        for eFreq in evalFreq:
            infoFile.write("Frequency = {:.9f}\n".format(eFreq))
            for stat in maskData.stats:
                infoFile.write("Statistic = {}\n".format(stat))
                for component in maskData.constraints[eFreq][stat]:
                    minVal = maskData.constraints[eFreq][stat][component][0]
                    maxVal = maskData.constraints[eFreq][stat][component][1]
                    infoFile.write(
                        "{}\t{}\t{}\t{}\n".format(
                            component,
                            minVal,
                            maxVal,
                            maskData.insideOut[eFreq][stat][component],
                        )
                    )
        # then loop through each
        infoFile.close()
        maskSize = 0
        for eFreq in evalFreq:
            if len(maskData.maskWindows[eFreq]) > maskSize:
                maskSize = len(maskData.maskWindows[eFreq])
        # create window mask array and initalise to -1
        winMaskArray = np.ones(shape=(len(evalFreq), maskSize), dtype=int) * -1
        # now fill the array
        for eIdx, eFreq in enumerate(evalFreq):
            lst = list(maskData.maskWindows[eFreq])
            winMaskArray[eIdx, 0 : len(lst)] = lst
        np.save(winName, winMaskArray)

    def getFileNames(self, maskName: str, sampleFreq: float) -> Tuple[str, str]:
        """Get the name of a mask file
        
        This method is here to give consistent file names for mask files.

        Parameters
        ----------
        maskName : str
            The name of the mask
        sampleFreq : float
            The sampling frequency of the data

        Returns
        -------
        infoFile : str
            The name of the mask infoFile
        winFile : str
            The name of the mask winFile
        """

        checkAndMakeDir(self.datapath)
        sampleFreqStr = fileFormatSampleFreq(sampleFreq)
        name: str = maskName + "_{}".format(sampleFreqStr)
        infoFile: str = os.path.join(self.datapath, name + ".info")
        winFile: str = os.path.join(
            self.datapath, name
        )  # no need for extension here, numpy adds one
        return infoFile, winFile

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """

        textLst = []
        if self.datapath == "":
            textLst.append(
                "No datapath given. Please set the datapath attribute of the class"
            )
        else:
            textLst.append("Datapath = {}".format(self.datapath))

        return textLst

