import os
from copy import deepcopy
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any, Union

from resistics.common.base import ResisticsBase
from resistics.common.print import breakComment, arrayToStringSci
from resistics.common.io import checkAndMakeDir
from resistics.common.math import intdiv
from resistics.spectra.data import SpectrumData
from resistics.window.utils import gIndex2datetime


class SpectrumReader(ResisticsBase):
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
    getChannels()
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
    getDataSize()
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
        self.file = None

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

    def getSpectrumData(
        self, localIndex: int, data: Dict[str, np.ndarray]
    ) -> SpectrumData:
        """Return a spectrum data object from a data dictionary and the local index
        
        Parameters
        ----------
        localIndex : int
            The local index of the spectra window
        data : Dict[str, np.ndarray]
            The data dictionary

        Returns
        -------
        SpectrumData
            A SpectrumData object
        """

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

    def openBinaryForReading(self, filename: str, fileInc: int) -> bool:
        """Open a binary data file for reading

        self.file is not set in this method because spectra data is read using memmap.

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
        return self.getSpectrumData(localIndex, data)

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

    def readBinaryBatchGlobal(
        self, globalIndices: Union[List[int], None] = None
    ) -> Union[List[SpectrumData], np.ndarray]:
        """Batch read binary windows

        Batch reading binary windows allows the data for calculation to be split over multi processes

        Parameters
        ----------
        globalIndices : List[int], None, optional
            The global indices to batch read. By default, all windows will be returned if not specified

        Returns
        -------
        List[SpectrumData], np.ndarray
            An array or list of SpectrumData objects
        """
        if globalIndices is not None and len(globalIndices) == 0:
            # zero windows request, return empty list
            return []

        self.file = open(self.filepath, "rb")
        batchData = np.fromfile(self.file, dtype=self.dataType)
        self.printText(
            "Reading {:.8f} GB of data from file {}".format(
                os.path.getsize(self.filepath) / 1e9, self.filepath
            )
        )
        self.file.close()
        self.file = None

        # find the windows to get
        if globalIndices is not None and len(globalIndices) > 0:
            localIndices = np.sort(np.array(list(globalIndices))) - self.getGlobalOffset()
            localIndices = localIndices[localIndices >= 0]
            localIndices = localIndices[localIndices < self.getNumWindows()]
            globalIndices = localIndices + self.getGlobalOffset()
        else:
            localIndices = np.arange(0, self.getNumWindows())
            globalIndices = localIndices + self.getGlobalOffset()

        specData = []
        dataSize = self.getDataSize()
        windowSize = dataSize * self.getNumChannels()
        for localIndex in localIndices:
            intOff = localIndex * windowSize
            data = {}
            for cI, c in enumerate(self.getChannels()):
                chanOff = intOff + cI * dataSize
                data[c] = batchData[chanOff : chanOff + dataSize]
            specData.append(self.getSpectrumData(localIndex, data))
        return specData, globalIndices

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
        if self.filepath != "":
            self.printText("Closing file {}".format(self.filepath))
            self.filepath = ""
            if self.file is not None:
                self.file.close()
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


class SpectrumWriter(ResisticsBase):
    """Writes spectra data for measurements

    Spectrum writer writes out spectra data as either binary (recommended) or ascii (more space required). A spectra file should be written out for each decimation level along with an information file, again for each decimation level. A single comments file is written out to record the processing sequence.

    Spectra files are located in:
    project -> specData -> site -> datapath -> spectra data files

    .dat files are ascii formatted data
    .bin files are binary formatted data

	Attributes
	----------
    datapath : str 
        Data root for spectra data
    filepath : str
        Filepath for spectra files
    refTime : datetime
        The reference time for the project
    file : bool 
        The data file 

	Methods
	-------
	__init__(datapath, refTime)
		Initialise the SpectrumWriter
    openBinaryForWriting(filename, fileInc, sampleFreq, winSize, winOverlap, globalOffset, numWindows, channels)
        Open file for writing binary spectra data
    writeBinary(specData)  
        Write out binary spectra data for a single time window          
    openAsciiForWriting(filename, fileInc, sampleFreq, winSize, winOverlap, globalOffset, numWindows, channels)
        Open file for writing ascii spectra data 
    writeAscii(specData)  
        Write out ascii spectra data for a single time window          
    writeInfoFile(filepath, sampleFreq, winSize, winOverlap, globalOffset, numWindows, channels)
        Write out the spectra information file
    writeCommentsFile(comments)   
        Write out the comments file      
    getFileBase(filename, fileInc)
        Get the spectra file name      
    closeFile()
        Close spectra data file 
    printList()
        Class status returned as list of strings       
	"""

    def __init__(self, datapath: str, refTime: datetime):
        """Initialise spectrum writer 

        Parameters
        ----------
        datapath : str
            Root pathway for spectra data
        refTime : datetime
            Reference time
        """
        self.datapath: str = datapath
        self.filepath: str = ""
        self.refTime: datetime = refTime
        self.file = None

    def openBinaryForWriting(
        self,
        filename,
        fileInc,
        sampleFreq,
        winSize,
        winOverlap,
        globalOffset,
        numWindows,
        channels,
    ) -> None:
        """Write binary spectrum file 

        Parameters
        ----------
        filename : str
            Filename for spectra files
        fileInc : int
            The decimation level
        sampleFreq : float
            Sampling frequeny of time data
        winSize : int
            Window size in samples for time data windows
        winOverlap : int
            Overlap size in samples for time data windows
        globalOffset : int
            Global offset for local and global indices
        numWindows : int
            The number of windows in the time data
        channels : List[str]
            Channels in data
        """
        # sort channels alphabetically - matching the order in the data files
        self.channels = sorted(channels)

        checkAndMakeDir(self.datapath)
        filebase: str = filename + "{:02d}".format(fileInc)
        # info file
        filepathInfo: str = os.path.join(self.datapath, filebase + ".info")
        self.writeInfoFile(
            filepathInfo,
            sampleFreq,
            winSize,
            winOverlap,
            globalOffset,
            numWindows,
            self.channels,
        )
        # open file for data
        self.filepath: str = os.path.join(self.datapath, filebase + ".bin")
        self.printText("Opening file {}".format(self.filepath))
        self.file = open(self.filepath, "wb")

    def writeBinary(self, specData):
        """Write spectrum data to binary file 

        Parameters
        ----------
        specData : SpectrumData
            Spectrum data
        """
        for c in self.channels:
            # save as complex64 instead of 128 - otherwise too big
            self.file.write(specData.data[c].astype("complex64").tobytes())

    def openAsciiForWriting(
        self,
        filename: str,
        fileInc: str,
        sampleFreq: float,
        winSize: int,
        winOverlap: int,
        globalOffset: int,
        numWindows: int,
        channels: List[str],
    ) -> None:
        """Write ascii spectrum file 

        Parameters
        ----------
        filename : str
            Filename for spectra files
        fileInc : int
            The decimation level
        sampleFreq : float
            Sampling frequeny of time data
        winSize : int
            Window size in samples for time data windows
        winOverlap : int
            Overlap size in samples for time data windows
        globalOffset : int
            Global offset for local and global indices
        numWindows : int
            The number of windows in the time data
        channels : List[str]
            Channels in data
        """
        # sort channels alphabetically - matching the order in the data files
        self.channels = sorted(channels)

        checkAndMakeDir(self.datapath)
        filebase: str = filename + "{:02d}".format(fileInc)
        # info file
        filepathInfo: str = os.path.join(self.datapath, filebase + ".info")
        self.writeInfoFile(
            filepathInfo,
            sampleFreq,
            winSize,
            winOverlap,
            globalOffset,
            numWindows,
            self.channels,
        )
        # open file for data
        self.filepath: str = os.path.join(self.datapath, filebase + ".dat")
        self.printText("Opening file {}".format(self.filepath))
        self.file = open(self.filepath, "w")

    def writeAscii(self, specData: SpectrumData) -> None:
        """Write spectrum data to ascii file 

        Parameters
        ----------
        specData : SpectrumData
            Spectrum data
        """
        for c in self.channels:
            outStr = arrayToStringSci(specData.data[c])
            outStr = outStr + "\n"
            self.file.write(outStr)

    def writeInfoFile(
        self,
        filepath: str,
        sampleFreq: float,
        winSize: int,
        winOverlap: int,
        globalOffset: int,
        numWindows: int,
        channels: List[str],
    ) -> None:
        """Write info file 

        Parameters
        ----------
        filepath : str
            Filepath for info file
        sampleFreq : float
            Sampling frequeny of time data
        winSize : int
            Window size in samples for time data windows
        winOverlap : int
            Overlap size in samples for time data windows
        globalOffset : int
            Global offset for local and global indices
        numWindows : int
            The number of windows in the time data
        channels : List[str]
            Channels in data
        """
        infoFile = open(filepath, "w")
        # write out header information
        numChannels = len(channels)
        tmp = winSize + 1  # if winSize is odd, this will go down
        if winSize % 2 == 0:
            tmp = tmp + 1
        dataSize = intdiv(tmp, 2)
        infoFile.write(
            "Reference time = {}\nSample frequency = {:.8f}\nWindow size = {:d}\nWindow overlap = {:d}\nGlobal offset = {:d}\n".format(
                self.refTime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                sampleFreq,
                winSize,
                winOverlap,
                globalOffset,
            )
        )
        infoFile.write(
            "Number of windows = {:d}\nData size = {:d}\n".format(numWindows, dataSize)
        )
        infoFile.write("Number of channels = {:d}\n".format(numChannels))
        infoFile.write("Channels = " + " ".join(channels))
        infoFile.close()

    def writeCommentsFile(self, comments: List[str]) -> None:
        """Write comments file 

        Parameters
        ----------
        comments : List[str]
            List of comments
        """
        import resistics

        with open(os.path.join(self.datapath, "comments.txt"), "w") as f:
            for c in comments:
                f.write("{}\n".format(c))
            f.write(
                "Spectra data written out to {} on {} using resistics {}\n".format(
                    self.datapath, datetime.now(), resistics.__version__
                )
            )
            f.write(breakComment())

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
        if not (self.file is None):
            textLst.append("Current file open: {}".format(self.filepath))
        return textLst
