from os.path import basename, splitext
import numpy as np
import math
import collections
from typing import List, Dict, Union

# import from package
from resistics.ioHandlers.ioHandler import IOHandler
from resistics.dataObjects.transferFunctionData import TransferFunctionData


class TransferFunctionReader(IOHandler):
    """Class for reading and writing maskData

    Transfer function files are located in:
    project -> transFuncData -> site -> filepath

	Attributes
	----------
    filepath : str 
        Transfer function file
    headers : OrderedDict()
        Transfer function file header information

	Methods
	-------
	__init__(filepath="")
		Initialise the transferFunctionReader file
	read(filepath)
        Set and read a new transfer function file
    readFile()
        Read the transfer function file
    readInternal()
        Read internally formatted transfer function files
    formatHeaders()
        Format header values for internally formatted transfer function files
    readEdi()
        Read Edi formatted tranfer function files
    readNumericBlock(lines)
        Read a numeric block for Edi transfer function files
    constructComplex(real, imag)
        Construct a complex valued array from arrays of the real and imaginary values of the complex numbers
    printList()
        Class status returned as list of strings     
	"""

    def __init__(self, filepath: str = "") -> None:
        """Initialise

        Parameters
        ----------
        filepath : str, optional
            Path to transfer function file file
        """

        self.filepath: str = filepath
        self.headers = collections.OrderedDict()
        # read file if given
        if self.filepath != "":
            self.readFile()

    def read(self, filepath: str) -> None:
        """Read a new file

        Parameters
        ----------
        filepath : str
            Filepath to transfer function file
        """

        self.filepath = filepath
        self.readFile()

    def readFile(self) -> TransferFunctionData:
        """Read a transfer function file

        Returns
        -------
        TransferFunctionData
            Tranfser function data object
        """

        # get the extension and decide how to read it
        filename, extension = splitext(self.filepath)
        if "edi" in extension:
            return self.readEdi()
        else:
            return self.readInternal()

    def readInternal(self) -> None:
        """Read an internally formatted transfer function file

        Returns
        -------
        TransferFunctionData
            Tranfser function data object
        """

        key = ""
        impedances: Dict = {}
        variances: Dict = {}
        freq: Union[List[float], np.ndarray] = []
        with open(self.filepath, "r") as inFile:
            lines = inFile.readlines()
        numLines = len(lines)
        # find the headers
        for i in range(0, numLines):
            line = lines[i].strip()
            if line == "":
                continue
            if "Evaluation frequencies" not in line:
                split = line.split("=")
                self.headers[split[0].strip()] = split[1].strip()
            else:
                break
        # format headers
        self.formatHeaders()
        # read in the frequencies
        freq = np.loadtxt(lines[i + 1].strip().split(","), dtype="float")
        # read in the rest of the data
        for j in range(i + 2, numLines):
            line = lines[j].strip()
            if "Z" in line or "Var" in line:
                split = line.split("-")
                key = split[1].strip()
                j += 1
                if "Z" in line:
                    impedances[key] = np.loadtxt(
                        lines[j].strip().split(","), dtype="complex"
                    )
                if "Var" in line:
                    variances[key] = np.loadtxt(
                        lines[j].strip().split(","), dtype="float"
                    )
        self.tfData = TransferFunctionData(freq, impedances, variances)

    def formatHeaders(self) -> None:
        """Format  header values for the internal transfer function file format"""

        if "sampleFreq" in self.headers:
            self.headers["sampleFreq"] = float(self.headers["sampleFreq"])
        test = ["insite", "outsite", "remotesite"]
        for header in test:
            if header in self.headers and self.headers[header] == "False":
                self.headers[header] = False
        test = ["inchans", "outchans", "remotechans"]
        for header in test:
            if header in self.headers and self.headers[header] == "False":
                self.headers[header] = []

    def readEdi(self) -> None:
        """Read an EDI transfer function file

        EDI files have an >END that signifies the end of the file. This works quite nicely to our advantage when reading an EDI file.

        Returns
        -------
        TransferFunctionData
            Tranfser function data object
        """

        with open(self.filepath, "r") as inFile:
            lines = inFile.readlines()
        numLines = len(lines)
        section: Dict = {}
        sectionNum = 0
        sectionBreaks = []
        for i in range(0, numLines):
            # get the starting line of the sections
            # at the same time strip out all new lines characters
            lines[i] = lines[i].strip()
            if ">" in lines[i]:
                split = lines[i].split(" ")
                name = split[0][1:]
                # save the section
                section[name] = sectionNum
                # save the starting line of the section
                sectionBreaks.append(i)
                sectionNum = sectionNum + 1

        ## set headers
        startLine = sectionBreaks[section["HEAD"]]
        endLine = sectionBreaks[section["HEAD"] + 1]
        for i in range(startLine + 1, endLine):
            if lines[i] != "":
                split = lines[i].split("=")
                self.headers[split[0]] = split[1]

        ## set frequency information
        # get number of frequencies
        startLine = sectionBreaks[section["FREQ"]]
        endLine = sectionBreaks[section["FREQ"] + 1]
        split = lines[startLine].split(" ")
        numFreq = 0
        for s in split:
            if "NFREQ" in s:
                split2 = s.split("=")
                numFreq = int(split2[1])
        freq = self.readNumericBlock(lines[startLine + 1 : endLine])

        ## read impedance tensor components and variances
        components = ["ZXX", "ZYY", "ZXY", "ZYX"]
        polarisations = ["ExHx", "EyHy", "ExHy", "EyHx"]
        variances = {}
        impedances = {}

        for comp, pol in zip(components, polarisations):
            # real
            realComp = "{}R".format(comp)
            startLine = sectionBreaks[section[realComp]]
            endLine = sectionBreaks[section[realComp] + 1]
            real = self.readNumericBlock(lines[startLine + 1 : endLine])
            # imag
            imagComp = "{}I".format(comp)
            startLine = sectionBreaks[section[imagComp]]
            endLine = sectionBreaks[section[imagComp] + 1]
            imag = self.readNumericBlock(lines[startLine + 1 : endLine])
            # impedance
            impedances[pol] = self.constructComplex(real, imag)
            # variance
            varComp = "{}.VAR".format(comp)
            startLine = sectionBreaks[section[varComp]]
            endLine = sectionBreaks[section[varComp] + 1]
            variances[pol] = self.readNumericBlock(lines[startLine + 1 : endLine])

        # initialise impedance tensor
        self.tfData = TransferFunctionData(freq, impedances, variances)

    def readNumericBlock(self, lines: List[str]):
        """Read an EDI numeric block

        Parameters
        ----------
        lines : List[str]
            List of strings (or the lines in the file) that are a numeric block

        Returns
        -------
        np.ndarray
            Numeric data as a numpy array
        """

        tmp = []
        size = len(lines)
        for i in range(0, size):
            if lines[i] == "":
                continue
            split = lines[i].split(" ")
            # add the numbers
            tmp = tmp + split
        # convert to float
        arr = np.zeros(shape=(len(tmp)))
        for idx, t in enumerate(tmp):
            arr[idx] = float(t)
        return arr

    def constructComplex(self, real, imag):
        """Construct a complex array from separate real and imaginary arrays 

        Parameters
        ----------
        real : np.ndarray
            An array of the real part of the complex data
        imag : np.ndarray
            An array of the complex part of the complex data

        Returns
        -------
        np.ndarray
            Numpy array of complex data
        """

        size = len(real)
        arr = np.zeros(shape=(size), dtype="complex")
        for i in range(0, size):
            arr[i] = complex(real[i], imag[i])
        return arr

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """

        textLst = []
        textLst.append("Headers in file")
        for h, val in self.headers.items():
            textLst.append("\t{} = {}".format(h, val))
        return textLst

