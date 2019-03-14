from os.path import basename, splitext
import numpy as np
import collections
from typing import List, Dict, IO, Any

# import from package
from resistics.ioHandlers.ioHandler import IOHandler
from resistics.dataObjects.transferFunctionData import TransferFunctionData
from resistics.utilities.utilsPrint import listToString, arrayToString


class TransferFunctionWriter(IOHandler):
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
	__init__(filepath, tfData)
		Initialise the transferFunctionWriter
	defaultHeaders()
        Get a dictionary of default header values
    setHeaders(**kwargs)
        Set header values
    addHeader(header, val)
        Add a new header or update an existing one
    setPathAndData(filepath, tfData)
        Set a new filepath for writing and new transfer function data
    write()
        Write transfer function data to internal format transfer function file
    writeEdi()
        Write transfer function data to Edi file
    writeNumericBlock(lines)
        Write a numeric block for Edi files
    printList()
        Class status returned as list of strings     
	"""

    def __init__(self, filepath: str, tfData: TransferFunctionData, **kwargs) -> None:
        """Initialise

        Parameters
        ----------
        filepath : str
            Path to write a transfer function file file
        tfData : TransferFunctionData
            The transfer function data
        sampleFreq : float, optional
            The sampling frequency of the original time data
        insite : str
            The input site
        inchans : List[str]
            The input channels
        outsite : str
            The output site
        outchans : List[str]
            The output channels
        remotesite : str
            The remote reference site
        remotechans : List[str]
            The remote reference chans used
        """

        self.filepath: str = filepath
        self.tfData: TransferFunctionData = tfData
        self.headers: Dict = self.defaultHeaders()
        for h in self.headers:
            if h in kwargs:
                self.headers[h] = kwargs[h]
        # do the polarisations
        self.polarisations = []
        if len(self.headers["inchans"]) > 0 and len(self.headers["outchans"]) > 0:
            for oC in self.headers["outchans"]:
                for iC in self.headers["inchans"]:
                    self.polarisations.append("{}{}".format(oC, iC))
        else:
            self.polarisations = sorted(tfData.polarisations)

    def defaultHeaders(self) -> Dict[str, Any]:
        """Get the default header values

        Returns
        -------
        default : Dict
            A dictionary with default header keys and values
        """

        default = collections.OrderedDict()
        default["sampleFreq"] = 0
        default["insite"] = False
        default["inchans"] = []
        default["outsite"] = False
        default["outchans"] = []
        default["remotesite"] = False
        default["remotechans"] = []
        return default

    def setHeaders(self, **kwargs):
        """Set header values

        Parameters
        ----------
        sampleFreq : float, optional
            The sampling frequency of the original time data
        insite : str
            The input site
        inchans : List[str]
            The input channels
        outsite : str
            The output site
        outchans : List[str]
            The output channels
        remotesite : str
            The remote reference site
        remotechans : List[str]
            The remote reference chans used
        """        

        for h in self.headers:
            if h in kwargs:
                self.headers[h] = kwargs[h]

    def addHeader(self, header: str, value: Any) -> None:
        """Add a header value for writing out
        
        Parameters
        ----------
        header : str
            The name of the header
        value : Any
            The value for the header
        """

        self.headers[header] = value

    def setPathAndData(self, filepath: str, tfData: TransferFunctionData) -> None:
        """Add a header value for writing out
        
        Parameters
        ----------
        filepath : str
            The filepath to write out to
        tfData : TransferFunctionData
            Transfer function data object
        """

        self.filepath = filepath
        self.tfData = tfData

    def write(self):
        """Write transfer function data in internal format file"""

        outF = open(self.filepath, "w")
        self.printText("Writing out transfer function data to file {}".format(self.filepath))
        # write out the headers
        for h, v in self.headers.items():
            if isinstance(v, list):
                v = listToString(v)
                if len(v) == 0:
                    v = "False"
            outF.write("{} = {}\n".format(h, v))

        # write evaluation frequencies
        outF.write("Evaluation frequencies\n")
        outF.write("{}\n".format(arrayToString(self.tfData.freq)))

        # now need to write out the other data
        for pol in self.polarisations:
            outF.write("Z-{}\n".format(pol))
            outF.write("{}\n".format(arrayToString(self.tfData.data[pol])))

        # variances
        for pol in self.polarisations:
            outF.write("Var-{}\n".format(pol))
            outF.write("{}\n".format(arrayToString(self.tfData.variances[pol])))

        outF.close()

    def writeEdi(self) -> None:
        """Write transfer function data in Edi file"""

        # Write the head and other meta information
        # Then write the values section by section
        outF = open(self.filepath, "w")
        outF.write(">HEAD\n\n")
        # for key, value in self.head.items():
        #     outF.write('{}={}\n'.format(key,value))
        # write out info
        outF.write("\n\n>INFO MAXINFO 5000\n\nempty\n\n")
        # write out define meas
        outF.write(">=DEFINEMEAS\n\nMAXCHAN=7\nMAXRUN=99\nMAXMEAS=9999\n")
        outF.write("UNITS=M\nREFLAT=+47:32:06.08\nREFLONG=+08:01:51.55\nREFELEV=0.00")
        outF.write(">HMEAS ID= 11.001 CHTYPE=HX X= 0 Y= 0 AZM= 0.\n")
        outF.write(">HMEAS ID= 12.001 CHTYPE=HY X= 0 Y= 0 AZM= 90.\n")
        outF.write(">HMEAS ID= 13.001 CHTYPE=HZ X= 0 Y= 0 AZM= 0.\n")
        outF.write(">EMEAS ID= 14.001 CHTYPE=EX X= 0 Y= 0 X2= 0 Y2= 0\n")
        outF.write(">EMEAS ID= 15.001 CHTYPE=EY X= 0 Y= 0 X2= 0 Y2= 0\n")
        outF.write("\n\n")
        # write out MTsect
        outF.write(">=MTSECT\n\nSECTID={}\nNFREQ={:d}\n\n")
        outF.write("HX= 11.001\nHY= 12.001\nHZ= 13.001\nEX= 14.001\nEY= 15.001\n\n")
        # write out frequencies
        numFreq = self.tfData.freq.size
        outF.write(">FREQ NFREQ={:d} ORDER=DEC // {:d}\n".format(numFreq, numFreq))
        self.writeNumericBlock(outF, self.tfData.freq)
        # write out impedances and variances
        for pol in self.polarisations:
            d = "{}{}".format(pol[1], pol[3])
            d = d.upper()
            # need to do a real and imaginary part
            outF.write("\n>Z{}R ROT=0.0 // {:d}\n".format(d, numFreq))
            self.writeNumericBlock(outF, self.tfData.data[pol].real)
            outF.write("\n>Z{}I ROT=0.0 // {:d}\n".format(d, numFreq))
            self.writeNumericBlock(outF, self.tfData.data[pol].imag)
            outF.write("\n>Z{}.VAR ROT=0.0 // {:d}\n".format(d, numFreq))
            self.writeNumericBlock(outF, self.tfData.variances[pol].real)
        # write end of file
        outF.write("\n>END")

    def writeNumericBlock(self, outF: IO[str], vals: np.ndarray) -> None:
        """Write numeric blocks for edi files

        The format is: 
        5 values on a line
        Each value has 5 decimal places and is in scientific notation

        Parameters
        ----------
        outF : file object
            Output file
        vals : np.ndarray
            Numeric files to write out
        """

        numVals = len(vals)
        for i in range(0, numVals):
            if i > 0 and i % 5 == 0:
                outF.write("\n")
            outF.write(" {:.5e}".format(vals[i]))

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """

        textLst: List[str] = []
        textLst.append("Filepath = {}".format(self.filepath))
        textLst.append("Polarisations = {}".format(self.polarisations))
        textLst.append("Headers set")
        for h, val in self.headers.items():
            textLst.append("\t{} = {}".format(h, val))
        return textLst
