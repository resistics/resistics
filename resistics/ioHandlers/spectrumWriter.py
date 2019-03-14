import os
import numpy as np
from datetime import datetime
from typing import List

# import from package
from resistics.ioHandlers.ioHandler import IOHandler
from resistics.dataObjects.spectrumData import SpectrumData
from resistics.utilities.utilsIO import checkAndMakeDir
from resistics.utilities.utilsPrint import breakComment, arrayToStringSci
from resistics.utilities.utilsMath import intdiv


class SpectrumWriter(IOHandler):
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

    # def getDataRoot(self):
    #     return self.datapath

    # def getFilePath(self):
    #     return self.filepath

    # def getFile(self):
    #     return self.file

    # def getRefTime(self):
    #     return self.refTime

    # def setDataRoot(self, datapath):
    #     self.datapath = datapath

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
            channels,
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
            channels,
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

        filepathComments: str = os.path.join(self.datapath, "comments.txt")
        commentFile = open(filepathComments, "w")
        for comment in comments:
            commentFile.write("{}\n".format(comment))
        # add a comment about writing out
        commentFile.write(
            "Spectra data written out to {} on {}\n".format(
                self.datapath, datetime.now()
            )
        )
        commentFile.write(breakComment())
        commentFile.close()

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

