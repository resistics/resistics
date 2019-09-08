import os
import glob
import struct
from datetime import datetime, timedelta
import numpy as np
from typing import List, Tuple, Dict, Any

# import from package
from resistics.ioHandlers.dataReaderLemiB423 import (
    DataReaderLemiB423,
    readB423Params,
    readB423Header,
)
from resistics.dataObjects.timeData import TimeData
from resistics.utilities.utilsChecks import isMagnetic, isElectric, consistentChans
from resistics.utilities.utilsPrint import blockPrint
from resistics.utilities.utilsClean import (
    removeZeros,
    removeZerosSingle,
    removeNansSingle,
)


def folderB423EHeaders(
    folderpath: str,
    sampleFreq: float,
    ex: str = "E1",
    ey: str = "E2",
    dx: float = 1,
    dy: float = 1,
    folders: List = [],
) -> None:
    """Construct B423E headers for subfolders of a folder

    Parameters
    ----------
    folderpath : str
        The path to the folder
    sampleFreq : float
        The sampling frequency of the data
    ex : str, optional
        The channel E1, E2, E3 or E4 in the data that represents Ex. Default E1.
    ey : str, optional
        The channel E1, E2, E3 or E4 in the data that represents Ey. Default E2.
    dx : float, optional
        Distance between x electrodes
    dy : float, optional
        Distance between y electrodes
    folder : List, optional
        An optional list of subfolders
    """

    if len(folders) == 0:
        folders = [f.path for f in os.scandir(folderpath) if f.is_dir()]
    # now construct headers for each folder
    for folder in folders:
        measB423EHeaders(folder, sampleFreq, ex=ex, ey=ey, dx=dx, dy=dy)


def measB423EHeaders(
    datapath: str,
    sampleFreq: float,
    ex: str = "E1",
    ey: str = "E2",    
    dx: float = 1,
    dy: float = 1,
) -> None:
    """Read B423E files and construct some headers
    
    Parameters
    ----------
    site : str
        The path to the site
    sampleFreq : float
        The sampling frequency of the data
    ex : str, optional
        The channel E1, E2, E3 or E4 in the data that represents Ex. Default E1.
    ey : str, optional
        The channel E1, E2, E3 or E4 in the data that represents Ey. Default E2.        
    dx : float, optional
        Distance between x electrodes
    dy : float, optional
        Distance between y electrodes
    """

    from resistics.utilities.utilsPrint import generalPrint, warningPrint, errorPrint
    from resistics.ioHandlers.dataWriter import DataWriter

    dataFiles = glob.glob(os.path.join(datapath, "*.B423"))
    dataFilenames = [os.path.basename(dFile) for dFile in dataFiles]
    starts = []
    stops = []
    gains1 = {}
    gains2 = {}
    cumSamples = 0
    for idx, dFile in enumerate(dataFiles):
        generalPrint("constructB423EHeaders", "Reading data file {}".format(dFile))
        dataHeaders, firstDatetime, lastDatetime, numSamples = readB423Params(
            dFile, sampleFreq, 1024, 26
        )
        print(dataHeaders)
        generalPrint(
            "constructB423EHeaders",
            "start time = {}, end time = {}".format(firstDatetime, lastDatetime),
        )
        generalPrint(
            "constructB423EHeaders", "number of samples = {}".format(numSamples)
        )
        cumSamples += numSamples
        starts.append(firstDatetime)
        stops.append(lastDatetime)
        # gains1
        gains1[dataFilenames[idx]] = dict()
        gains1[dataFilenames[idx]]["E1"] = dataHeaders["Ke1"]
        gains1[dataFilenames[idx]]["E2"] = dataHeaders["Ke2"]
        gains1[dataFilenames[idx]]["E3"] = dataHeaders["Ke3"]
        gains1[dataFilenames[idx]]["E4"] = dataHeaders["Ke4"]
        # gains2
        gains2[dataFilenames[idx]] = dict()
        gains2[dataFilenames[idx]]["E1"] = dataHeaders["Ae1"]
        gains2[dataFilenames[idx]]["E2"] = dataHeaders["Ae2"]
        gains2[dataFilenames[idx]]["E3"] = dataHeaders["Ae3"]
        gains2[dataFilenames[idx]]["E4"] = dataHeaders["Ae4"]
    # now need to search for any missing data
    sampleTime = timedelta(seconds=1.0 / sampleFreq)
    # sort by start times
    sortIndices = sorted(list(range(len(starts))), key=lambda k: starts[k])
    check = True
    for i in range(1, len(dataFiles)):
        # get the stop time of the previous dataset
        stopTimePrev = stops[sortIndices[i - 1]]
        startTimeNow = starts[sortIndices[i]]
        if startTimeNow != stopTimePrev + sampleTime:
            warningPrint(
                "constructB423EHeaders", "There is a gap between the datafiles"
            )
            warningPrint(
                "constructB423EHeaders",
                "Please separate out datasets with gaps into separate folders",
            )
            warningPrint("constructB423EHeaders", "Gap found between datafiles:")
            warningPrint(
                "constructB423EHeaders", "1. {}".format(dataFiles[sortIndices[i - 1]])
            )
            warningPrint(
                "constructB423EHeaders", "2. {}".format(dataFiles[sortIndices[i]])
            )
            check = False
    # if did not pass check, then exit
    if not check:
        errorPrint(
            "constructB423EHeaders",
            "All data for a single recording must be continuous.",
            quitRun=True,
        )

    # time of first and last sample
    datetimeStart = starts[sortIndices[0]]
    datetimeStop = stops[sortIndices[-1]]

    # global headers
    globalHeaders = {
        "sample_freq": sampleFreq,
        "num_samples": cumSamples,
        "start_time": datetimeStart.strftime("%H:%M:%S.%f"),
        "start_date": datetimeStart.strftime("%Y-%m-%d"),
        "stop_time": datetimeStop.strftime("%H:%M:%S.%f"),
        "stop_date": datetimeStop.strftime("%Y-%m-%d"),
        "meas_channels": 4,
    }
    writer = DataWriter()
    globalHeaders = writer.setGlobalHeadersFromKeywords({}, globalHeaders)

    # channel headers
    channels = ["E1", "E2", "E3", "E4"]
    chanMap = {"E1": 0, "E2": 1, "E3": 2, "E4": 3}
    sensors = {"E1": "0", "E2": "0", "E3": "0", "E4": "0"}
    posX2 = {"E1": 1, "E2": 1, "E3": 1, "E4": 1}
    posY2 = {"E1": 1, "E2": 1, "E3": 1, "E4": 1}
    posX2[ex] = dx
    posY2[ey] = dy

    chanHeaders = []
    for chan in channels:
        # sensor serial number
        cHeader = dict(globalHeaders)
        cHeader["ats_data_file"] = " ,".join(dataFilenames)
        if ex == chan:
            cHeader["channel_type"] = "Ex"
        elif ey == chan:
            cHeader["channel_type"] = "Ey"
        else:
            cHeader["channel_type"] = chan
        cHeader["scaling_applied"] = False
        cHeader["ts_lsb"] = 1
        # cHeader["gain_stage1"] = ", ".join(
        #     ["{:.6e}".format(gains1[dFile][chan]) for dFile in dataFilenames]
        # )
        # cHeader["gain_stage2"] = ", ".join(
        #     ["{:.6e}".format(gains2[dFile][chan]) for dFile in dataFilenames]
        # )
        cHeader["gain_stage1"] = 1
        cHeader["gain_stage2"] = 1
        cHeader["hchopper"] = 0
        cHeader["echopper"] = 0
        cHeader["pos_x1"] = 0
        cHeader["pos_x2"] = posX2[chan]
        cHeader["pos_y1"] = 0
        cHeader["pos_y2"] = posY2[chan]
        cHeader["pos_z1"] = 0
        cHeader["pos_z2"] = 1
        cHeader["sensor_sernum"] = sensors[chan]
        chanHeaders.append(cHeader)
    chanHeaders = writer.setChanHeadersFromKeywords(chanHeaders, {})
    writer.setOutPath(datapath)
    writer.writeHeaders(
        globalHeaders, channels, chanMap, chanHeaders, rename=False, ext="h423E"
    )


class DataReaderLemiB423E(DataReaderLemiB423):
    """Data reader for Lemi B423E data

    Lemi B423E data has the following characteristics:

    - Lemi B423E records only telluric data, channels E1, E2, E3 and E4
    - To calculate Ex and Ey 
    - Lemi B423E raw data is signed long integer format   
    - Getting unscaled samples returns data with unit count for both the electric and magnetic fields. 
    - There is no header file for Lemi B423E data. There are some headers in the data file, but nothing for number of samples, sampling rate etc
    - 1024 bytes of headers in the data file

    In situations where a Lemi B423E dataset is recorded in multiple files, it is required that the recording is continuous. 

    Attributes
    ----------
    recChannels : Dict
        Channels in each data file
    dtype : np.float32
        The data type
    numHeaderFiles : int
        The number of header files
    numDataFiles : int
        The number of data files

    Methods
    -------
    __init__(dataPath)
        Initialise with path to the data directory
    setParameters()
        Set parameters specific to a data format
    getUnscaledSamples(**kwargs)
        Get raw, unscaled data
    getPhysicalSamples(**kwargs)
        Get data in physical units
    spamHeaders()
        Get sections and section headers to be read in for SPAM data
    chanDefaults()
        Get defaults values for channel headers
    readHeader()
        Read SPAM header files
    readHeaderXTR(headerFile)
        Read a XTR header file
    readHeaderXTRX(headerFile)
        Read a XTRX header files
    headersFromRawFile(rawFile, headers)
        Read headers from the data files
    mergeHeaders(headersList, chanHeadersList)
        Merge the headers from all the data files
    printDataFileList()
        Get data file information as a list of strings
    printDataFiles()
        Print data file information to terminal
    """

    def setParameters(self) -> None:
        """Set some data reader parameters for reading Lemi B423E data"""

        # get a list of the header and data files in the folder
        self.headerExt = "h423E"
        self.headerF = glob.glob(
            os.path.join(self.dataPath, "*.{}".format(self.headerExt))
        )
        self.dataF = glob.glob(os.path.join(self.dataPath, "*.B423"))
        # data byte information
        self.dataByteSize = 4
        self.recordByteSize = 26
        self.dataByteOffset = 1024
        # data type
        self.dtype = np.int_
        # get the number of data files and header files - this should be equal
        self.numHeaderFiles: int = len(self.headerF)
        self.numDataFiles: int = len(self.dataF)

    def getPhysicalSamples(self, **kwargs):
        """Get data scaled to physical values
        
        resistics uses field units, meaning physical samples will return the following:

        - Electrical channels in mV/km
        - Magnetic channels in mV
        - To get magnetic fields in nT, calibration needs to be performed

        Notes
        -----
        Once Lemi B423E data is scaled (which optionally happens in getUnscaledSamples), the magnetic channels is in pT and the electric channels is uV (micro volts). Therefore, both magnetic and electric channels need to divided by 1000 along with dipole length division (east-west spacing and north-south spacing).
        
        To get magnetic fields in nT, they have to be calibrated.

        Parameters
        ----------
        chans : List[str]
            List of channels to return if not all are required
        startSample : int
            First sample to return
        endSample : int
            Last sample to return
        remaverage : bool
            Remove average from the data
        remzeros : bool
            Remove zeroes from the data
        remnans: bool
            Remove NanNs from the data

        Returns
        -------
        TimeData
            Time data object
        """

        # initialise chans, startSample and endSample with the whole dataset
        options = self.parseGetDataKeywords(kwargs)
        # get unscaled data but with gain scalings applied
        timeData = self.getUnscaledSamples(
            chans = options["chans"],
            startSample=options["startSample"],
            endSample=options["endSample"],
            scale=True,
        )
        # convert to field units and divide by dipole lengths
        for chan in options["chans"]:
            # divide by the 1000 to convert electric channels from microvolt to millivolt
            timeData.data[chan] = timeData.data[chan] / 1000.0
            timeData.addComment("Dividing channel {} by 1000 to convert microvolt to millivolt".format(chan))
            if chan == "Ex":
                # multiply by 1000/self.getChanDx same as dividing by dist in km
                timeData.data[chan] = (
                    1000.0 * timeData.data[chan] / self.getChanDx(chan)
                )
                timeData.addComment(
                    "Dividing channel {} by electrode distance {} km to give mV/km".format(
                        chan, self.getChanDx(chan) / 1000.0
                    )
                )
            if chan == "Ey":
                # multiply by 1000/self.getChanDy same as dividing by dist in km
                timeData.data[chan] = 1000 * timeData.data[chan] / self.getChanDy(chan)
                timeData.addComment(
                    "Dividing channel {} by electrode distance {} km to give mV/km".format(
                        chan, self.getChanDy(chan) / 1000.0
                    )
                )

            # if remove zeros - False by default
            if options["remzeros"]:
                timeData.data[chan] = removeZerosSingle(timeData.data[chan])
            # if remove nans - False by default
            if options["remnans"]:
                timeData.data[chan] = removeNansSingle(timeData.data[chan])
            # remove the average from the data - True by default
            if options["remaverage"]:
                timeData.data[chan] = timeData.data[chan] - np.average(
                    timeData.data[chan]
                )

        # add comments
        timeData.addComment(
            "Remove zeros: {}, remove nans: {}, remove average: {}".format(
                options["remzeros"], options["remnans"], options["remaverage"]
            )
        )
        return timeData

    def getScalars(self, paramsDict: Dict) -> Dict:
        """Returns the scalars from a parameter dictionary

        Parameters
        ----------
        paramsDict : Dict
            The parameter dictionary for a data file usually read from the headers in the file

        Returns
        -------
        Dict 
            Dictionary with channels as keys and scalings as values
        """

        # need to get the channel orders here
        chans = []
        for cH in self.chanHeaders:
            chans.append(cH["channel_type"])
        print(chans)
        return {
            chans[0]: [paramsDict["Ke1"], paramsDict["Ae1"]],
            chans[1]: [paramsDict["Ke2"], paramsDict["Ae2"]],
            chans[2]: [paramsDict["Ke3"], paramsDict["Ae3"]],
            chans[3]: [paramsDict["Ke4"], paramsDict["Ae4"]],
        }

    def printDataFileList(self) -> List[str]:
        """Information about the data files as a list of strings
        
        Returns
        -------
        List[str]
            List of information about the data files
        """

        textLst: List[str] = []
        textLst.append("Data File\t\tSample Ranges")
        for dFile, sRanges in zip(self.dataFileList, self.dataRanges):
            textLst.append("{}\t\t{} - {}".format(dFile, sRanges[0], sRanges[1]))
        textLst.append("Total samples = {}".format(self.getNumSamples()))
        return textLst

    def printDataFileInfo(self) -> None:
        """Print a list of the data files"""

        blockPrint(
            "{} Data File List".format(self.__class__.__name__),
            self.printDataFileList(),
        )

