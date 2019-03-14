import os
from copy import deepcopy
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any

# import from package
from resistics.ioHandlers.ioHandler import IOHandler
from resistics.dataObjects.spectrumData import SpectrumData
from resistics.utilities.utilsWindow import gIndex2datetime
from resistics.utilities.utilsPrint import breakComment


class SpectrumReader(IOHandler):
    """Reads in spectra data for measurements

    Spectrum reader reads in the info file for the spectra data and .dat files (for ascii formatted spectra data) and .bin files (for binary formatted data).

    Spectra files are located in:
    project -> specData -> site -> datapath -> spectra data files

	Attributes
	----------
    datapath : str 
        Data root for spectra data
    headerKeys :
        Spectra file header keys
    headers : Dict
        Spectra file header values
    comments: List[str]
        Spectra file comments
    dataType :
        Data type of spectra data
    dataByteSize : int 
        Byte size of a single data point
    filepath : str
        Filepath for spectra files
    file : bool 
        The data file 

	Methods
	-------
	__init__(datapath)
		Initialise the SpectrumReader
	getReferenceTime()
		Get the reference time used for the spectrum calculation
    def getChannels()
        Get the channels in the spectra data
    getNumChannels()
        Get the number of channels
    getSampleFreq()
        Get the sampling frequency of the data
    getNumWindows()
        Get the number of windows in the spectra file        
    getWindowSize()
        Get the window size used for the data
    getWindowOverlap()
        Get the window overlap size
    def getDataSize()
        Get the number of samples in a spectra window
    getGlobalOffset()
        Get the window global offset
    getGlobalRange()
        Get the window global indices range
    getFrequencyArray()
        Get the frequency array of the frequencies of the spectra data points
    getComments()
        Get a deepcopy of the comments
    openBinaryForReading(filename, fileInc):
        Open a binary spectra file given by filename and fileInc (the decimation level)
    readBinaryWindowLocal(localIndex)
        Read a spectra window given by a local index in binary format
    readBinaryWindowGlobal(globalIndex)
        Read a spectra window given by a global index (relative to reference time) in binary format
    openAsciiForReading(filename, fileInc):
        Open a ascii spectra file given by filename and fileInc (the decimation level)
    readAsciiWindowLocal(localIndex)
        Read a spectra window given by a local index in ascii format
    readAsciiWindowGlobal(globalIndex)
        Read a spectra window given by a global index (relative to reference time) in ascii format
    readInfoFile(filepath)
        Read the spectra information file given by filepath
    getInfoValue(header, line)
        Put header value in the correct format for header and line
    readCommentsFile(filepath):
        Read the comments file given by filepath   
    getFileBase(filename, fileInc):
        Get the spectra file name
    closeFile()
        Close spectra data file 
    printList()
        Class status returned as list of strings       
	"""

    def __init__(self, datapath: str) -> None:
        """Initialise

        Parameters
        ----------
        datapath : str, optional
            Path to spectra directory
        """

        self.datapath: str = datapath
        if not os.path.exists(self.datapath):
            self.printWarning("Directory {} does not exist".format(self.datapath))
        self.headerKeys: List[str] = [
            "Reference time",
            "Sample frequency",
            "Window size",
            "Window overlap",
            "Global offset",
            "Number of windows",
            "Number of channels",
            "Data size",
            "Channels",
        ]
        self.headers: Dict = {}
        self.comments: List[str] = []
        self.dataType = np.dtype("complex64")
        self.dataByteSize: int = self.dataType.itemsize
        self.filepath: str = ""
        self.file: bool = False

    # def getDataRoot(self):
    #     return self.datapath

    # def getHeaders(self):
    #     return self.headerKeys

    # def getHeaderDict(self):
    #     return self.headers

    # def getComments(self):
    #     return self.comments

    # def setDataRoot(self, datapath):
    #     self.datapath = datapath

    def getReferenceTime(self) -> datetime:
        """Get reference time for spectra calculation

        Returns
        -------
        datetime
            The reference time used for the spectra calculation
        """

        return self.headers["Reference time"]

    def getSampleFreq(self) -> float:
        """Get sampling frequency of time data used for spectra calculation

        Returns
        -------
        float
            Sampling frequency of the time data
        """

        return float(self.headers["Sample frequency"])

    def getNumChannels(self) -> int:
        """Get the number of channels

        Returns
        -------
        int
            The number of channels
        """

        return int(self.headers["Number of channels"])

    def getChannels(self) -> List[str]:
        """Get the channels in the spectra data

        Returns
        -------
        List[str]
            List of channels
        """

        return self.headers["Channels"]

    def getNumWindows(self) -> int:
        """Get number of windows for which spectra have been calculated

        Returns
        -------
        int
            The number of windows
        """

        return int(self.headers["Number of windows"])

    def getWindowSize(self) -> int:
        """Get the size of the time data window in samples

        Returns
        -------
        int
            Size of time data window in samples
        """

        return int(self.headers["Window size"])

    def getWindowOverlap(self) -> int:
        """Get the size of the time data overlap in samples

        Returns
        -------
        int
            Size of time data overlap in samples
        """

        return int(self.headers["Window overlap"])

    def getDataSize(self) -> int:
        """Get the size of the corresponding spectrum data for a window

        Returns
        -------
        int
            Number of samples in one spectrum window
        """

        return int(self.headers["Data size"])

    def getGlobalOffset(self) -> int:
        """Get global window offset

        The global window offset references windows to the reference time rather than the start time of the time data

        Returns
        -------
        int
            Global window offset
        """

        return int(self.headers["Global offset"])

    def getGlobalRange(self) -> List[int]:
        """Get the range of window global indices

        The first global index is given by getGlobalOffset() as the counting starts from zero. The last global index is self.getGlobalOffset() + self.getNumWindows() - 1.

        Returns
        -------
        List[int]
            List with two elements, the first being the global index of the first spectrum window and the second, the global index of the last spectrum window
        """

        return [
            self.getGlobalOffset(),
            self.getGlobalOffset() + self.getNumWindows() - 1,
        ]

    def getFrequencyArray(self) -> np.ndarray:
        """Returns the frequency array

        Frequency array are the frequency points in the spectra

        Returns
        -------
        np.ndarray
            Frequency array
        """

        return np.linspace(0, self.getSampleFreq() / 2.0, self.getDataSize())

    def getComments(self) -> List[str]:
        """Get a deepcopy of the comments
        
        Returns
        -------
        List[str]
            Dataset comments as a list of strings
        """

        return deepcopy(self.comments)

    def openBinaryForReading(self, filename: str, fileInc: int) -> bool:
        """Open a binary data file for reading

        Parameters
        ----------
        filename: str
            Filename of spectra files
        fileInc : int
            The decimation level
        """

        filebase = self.getFileBase(filename, fileInc)
        filepathInfo = os.path.join(self.datapath, filebase + ".info")
        filepathComments = os.path.join(self.datapath, "comments.txt")
        self.filepath = os.path.join(self.datapath, filebase + ".bin")
        # check files exist
        if not os.path.exists(filepathInfo) or not os.path.exists(self.filepath):
            self.printWarning(
                "No data found in either {} or {}".format(filepathInfo, self.filepath)
            )
            return False
        # read info file
        self.readInfoFile(filepathInfo)
        self.readCommentsFile(filepathComments)
        self.channelByteSize = self.dataByteSize * self.getDataSize()
        self.windowByteSize = self.channelByteSize * self.getNumChannels()
        # set file to filepath - this is because binary files do not require opening using memap
        # self.file = self.filepath
        return True

    def readBinaryWindowLocal(self, localIndex: int) -> SpectrumData:
        """Get spectrum data for a window defined by a local index (for binary formatted data)

        Parameters
        ----------
        localIndex: int
            The local index
        """

        if localIndex >= self.getNumWindows():
            self.printWarning("Local index {:d} out of bounds".format(localIndex))
            self.printWarning(
                "Min index = {:d}, Max index = {:d}".format(0, self.getNumWindows() - 1)
            )
        # with binary files, want the correct bytes
        byteOff = localIndex * self.windowByteSize
        data = {}
        for cI, c in enumerate(self.getChannels()):
            chanOff = cI * self.channelByteSize
            data[c] = np.memmap(
                self.filepath,
                dtype=self.dataType,
                mode="r",
                offset=byteOff + chanOff,
                shape=(self.getDataSize()),
            )
        # return data
        startTime, stopTime = gIndex2datetime(
            localIndex + self.getGlobalOffset(),
            self.getReferenceTime(),
            self.getSampleFreq(),
            self.getWindowSize(),
            self.getWindowOverlap(),
        )
        return SpectrumData(
            windowSize=self.getWindowSize(),
            dataSize=self.getDataSize(),
            sampleFreq=self.getSampleFreq(),
            startTime=startTime,
            stopTime=stopTime,
            data=data,
            comments=self.comments,
        )

    def readBinaryWindowGlobal(self, globalIndex: int) -> SpectrumData:
        """Get spectrum data for a window defined by a global index (for binary formatted data)

        Parameters
        ----------
        globalIndex: int
            The global index
        """

        if (
            globalIndex >= self.getNumWindows() + self.getGlobalOffset()
            or globalIndex < self.getGlobalOffset()
        ):
            self.printWarning("Global index {:d} out of bounds".format(globalIndex))
            self.printWarning(
                "Min index = {:d}, Max index = {:d}".format(
                    self.getGlobalOffset(),
                    self.getGlobalOffset() + self.getNumWindows() - 1,
                )
            )
        # convert global index to local index and return readAsciiWindowLocal
        localIndex = globalIndex - self.getGlobalOffset()
        return self.readBinaryWindowLocal(localIndex)

    def openAsciiForReading(self, filename: str, fileInc: int) -> bool:
        """Open a ascii data file for reading

        Parameters
        ----------
        filename: str
            Filename of spectra files
        fileInc : int
            The decimation level
        """

        filebase = self.getFileBase(filename, fileInc)
        filepathInfo = os.path.join(self.datapath, filebase + ".info")
        self.filepath = os.path.join(self.datapath, filebase + ".dat")
        # check files exist
        if not os.path.exists(filepathInfo) or not os.path.exists(self.filepath):
            self.printWarning(
                "No data found in either {} or {}".format(filepathInfo, self.filepath)
            )
            return False
        # read info file
        self.readInfoFile(filepathInfo)
        # open file for reading
        self.file = open(self.filepath, "rb")
        # run through and find line endings
        self.lineOffset = []
        offset = 0
        for line in self.file:
            self.lineOffset.append(offset)
            offset += len(line)
        self.file.seek(0)
        return True

    def readAsciiWindowLocal(self, localIndex: int) -> SpectrumData:
        """Get spectrum data for a window defined by a local index (for ascii formatted data)

        Parameters
        ----------
        localIndex: int
            The local index
        """

        if localIndex >= self.getNumWindows():
            self.printWarning("Local index {:d} out of bounds".format(localIndex))
            self.printWarning(
                "Min index = {:d}, Max index = {:d}".format(0, self.getNumWindows() - 1)
            )
        # with ascii files, want the correct lines
        # find line where local index starts
        windowStartLine = localIndex * self.getNumChannels()
        data = {}
        for cI, c in enumerate(self.getChannels()):
            indexC = windowStartLine + cI
            self.file.seek(self.lineOffset[indexC])
            line = self.file.readline()
            data[c] = np.loadtxt(line.strip().split(","), dtype=complex)
        # return data
        startTime, stopTime = gIndex2datetime(
            localIndex + self.getGlobalOffset(),
            self.getReferenceTime(),
            self.getSampleFreq(),
            self.getWindowSize(),
            self.getWindowOverlap(),
        )
        return SpectrumData(
            windowSize=self.getWindowSize(),
            dataSize=self.getDataSize(),
            sampleFreq=self.getSampleFreq(),
            startTime=startTime,
            stopTime=stopTime,
            data=data,
            comments=self.comments,
        )

    def readAsciiWindowGlobal(self, globalIndex: int) -> SpectrumData:
        """Get spectrum data for a window defined by a global index (for ascii formatted data)

        Parameters
        ----------
        globalIndex: int
            The global index
        """

        if (
            globalIndex >= self.getNumWindows() + self.getGlobalOffset()
            or globalIndex < self.getGlobalOffset()
        ):
            self.printWarning("Global index {:d} out of bounds".format(globalIndex))
            self.printWarning(
                "Min index = {:d}, Max index = {:d}".format(
                    self.getGlobalOffset(),
                    self.getGlobalOffset() + self.getNumWindows() - 1,
                )
            )
        # convert global index to local index and return readAsciiWindowLocal
        localIndex = globalIndex - self.getGlobalOffset()
        return self.readAsciiWindowLocal(localIndex)

    def readInfoFile(self, filepath: str) -> None:
        """Read the info file for the spectra

        Parameters
        ----------
        filepath : str
            Info file path
        """

        infoFile = open(filepath, "r")
        lines = infoFile.readlines()
        infoFile.close()
        # loop through all headers and get values
        for h in self.headerKeys:
            for l in lines:
                if h in l:
                    self.headers[h] = self.getInfoValue(h, l)
                    break

    def getInfoValue(self, header: str, line: str) -> Any:
        """Format some of the info file values

        Parameters
        ----------
        header : str
            The header 
        line : str
            The line from the info file
        """

        split = line.split("=")
        split[1] = split[1].strip()
        if header == "Channels":
            return split[1].split()
        elif header == "Reference time":
            return datetime.strptime(split[1], "%Y-%m-%d %H:%M:%S.%f")
        elif header == "Sample frequency":
            return float(split[1])
        else:
            return int(float(split[1]))

    def readCommentsFile(self, filepath: str) -> None:
        """Read comments file from filepath

        Parameters
        ----------
        filepath : str
            Comments file path
        """

        if os.path.exists(filepath):
            f = open(filepath, "r")
            self.comments = f.readlines()
            f.close()
            for idx, comment in enumerate(self.comments):
                self.comments[idx] = comment.rstrip()
        # add a new comment
        self.comments.append("Reading spectra data in path {}".format(self.datapath))

    def getFileBase(self, filename: str, fileInc: int) -> str:
        """Read comments file from filepath

        Parameters
        ----------
        filename: str
            Filename of spectra files
        fileInc : int
            The decimation level
        """

        return filename + "{:02d}".format(fileInc)

    def closeFile(self):
        """Close spectra file"""

        if self.filepath != "" and self.file:
            self.printText("Closing file {}".format(self.filepath))
            self.file.close()
            self.filepath = ""
        else:
            print("No file open")

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """

        textLst = []
        textLst.append("Data root = {}".format(self.datapath))
        if len(self.headers) > 0:
            textLst.append("Filepath = {}".format(self.filepath))
            for h in self.headerKeys:
                textLst.append("{} = {}".format(h, self.headers[h]))
            if len(self.comments) > 0:
                textLst.append("Comments")
                for comment in self.comments:
                    textLst.append("\t{}".format(comment))
            else:
                textLst.append("No comments")
        return textLst

