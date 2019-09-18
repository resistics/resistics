import os
import glob
import struct
from datetime import datetime, timedelta
import numpy as np
from typing import List, Tuple, Dict, Any

# import from package
from resistics.ioHandlers.dataReaderInternal import DataReaderInternal
from resistics.dataObjects.timeData import TimeData
from resistics.utilities.utilsChecks import isMagnetic, isElectric, consistentChans
from resistics.utilities.utilsPrint import blockPrint
from resistics.utilities.utilsClean import (
    removeZeros,
    removeZerosSingle,
    removeNansSingle,
)


def folderB423Headers(
    folderpath: str,
    sampleFreq: float,
    hxSensor: int = 0,
    hySensor: int = 0,
    hzSensor: int = 0,
    hGain: int = 1,
    dx: float = 1,
    dy: float = 1,
    folders: List = [],
) -> None:
    """Construct B423 headers for subfolders of a folder

    Parameters
    ----------
    folderpath : str
        The path to the folder
    sampleFreq : float
        The sampling frequency of the data
    hxSensor : str, optional
        The x direction magnetic sensor, used for calibration
    hySensor : str, optional
        The y direction magnetic sensor, used for calibration
    hzSensor : str, optional
        The z direction magnetic sensor, used for calibration
    hGain : int
        Any gain on the magnetic channels which will need to be removed
    dx : float, optional
        Distance between x electrodes
    dy : float, optional
        Distance between y electrodes
    folder : List, optional
        An optional list of subfolders
    """

    if len(folders) == 0:
        folders = [f.path for f in os.scandir(folderpath) if f.is_dir()]
    print(folders)        
    # now construct headers for each folder
    for folder in folders:
        measB423Headers(folder, sampleFreq, hxSensor, hySensor, hzSensor, hGain, dx, dy)


def measB423Headers(
    datapath: str,
    sampleFreq: float,
    hxSensor: int = 0,
    hySensor: int = 0,
    hzSensor: int = 0,
    hGain: int = 1,
    dx: float = 1,
    dy: float = 1,
) -> None:
    """Read a single B423 measurement directory and construct headers
    
    Parameters
    ----------
    datapath : str
        The path to the measurement
    sampleFreq : float
        The sampling frequency of the data
    hxSensor : str, optional
        The x direction magnetic sensor, used for calibration
    hySensor : str, optional
        The y direction magnetic sensor, used for calibration
    hzSensor : str, optional
        The z direction magnetic sensor, used for calibration
    hGain : int
        Any gain on the magnetic channels which will need to be removed
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
    cumSamples = 0
    for idx, dFile in enumerate(dataFiles):
        generalPrint("constructB423Headers", "Reading data file {}".format(dFile))
        dataHeaders, firstDatetime, lastDatetime, numSamples = readB423Params(
            dFile, sampleFreq, 1024, 30
        )
        print(dataHeaders)
        generalPrint(
            "constructB423Headers",
            "start time = {}, end time = {}".format(firstDatetime, lastDatetime),
        )
        generalPrint(
            "constructB423Headers", "number of samples = {}".format(numSamples)
        )
        cumSamples += numSamples
        starts.append(firstDatetime)
        stops.append(lastDatetime)
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
            warningPrint("constructB423Headers", "There is a gap between the datafiles")
            warningPrint(
                "constructB423Headers",
                "Please separate out datasets with gaps into separate folders",
            )
            warningPrint("constructB423Headers", "Gap found between datafiles:")
            warningPrint(
                "constructB423Headers", "1. {}".format(dataFiles[sortIndices[i - 1]])
            )
            warningPrint(
                "constructB423Headers", "2. {}".format(dataFiles[sortIndices[i]])
            )
            check = False
    # if did not pass check, then exit
    if not check:
        errorPrint(
            "constructB423Headers",
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
        "meas_channels": 5,
    }
    writer = DataWriter()
    globalHeaders = writer.setGlobalHeadersFromKeywords({}, globalHeaders)

    # channel headers
    channels = ["Hx", "Hy", "Hz", "Ex", "Ey"]
    chanMap = {"Hx": 0, "Hy": 1, "Hz": 2, "Ex": 3, "Ey": 4}
    sensors = {"Hx": hxSensor, "Hy": hySensor, "Hz": hzSensor, "Ex": "0", "Ey": "0"}
    posX2 = {"Hx": 1, "Hy": 1, "Hz": 1, "Ex": dx, "Ey": 1}
    posY2 = {"Hx": 1, "Hy": 1, "Hz": 1, "Ex": 1, "Ey": dy}

    chanHeaders = []
    for chan in channels:
        # sensor serial number
        cHeader = dict(globalHeaders)
        cHeader["ats_data_file"] = ", ".join(dataFilenames)
        cHeader["channel_type"] = chan
        cHeader["scaling_applied"] = False
        cHeader["ts_lsb"] = 1
        cHeader["gain_stage1"] = hGain if isMagnetic(chan) else 1
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
        globalHeaders, channels, chanMap, chanHeaders, rename=False, ext="h423"
    )


def readB423Params(
    dataFile: str, sampleFreq: float, dataByteOffset: int, recordByteSize: int
):
    """Get the parameters of the B423 data file
    
    Parameters
    ----------
    dataFile : str
        The data file as a string
    sampleFreq : float
        The sampling frequency in Hz
    dataByteOffset : int
        The offset till the start of the data in bytes
    recordByteSize : int
        The size in bytes of a record
    
    Returns
    -------
    headers : dict
        The header values as a dictionary
    firstDatetime : datetime
        The time of the first sample in the data file
    lastDatetime : datetime
        The time of the last sample in the data file
    numSamples : int   
        The number of samples in the data file given the sampling frequency
    """

    from resistics.utilities.utilsPrint import errorPrint

    filesize = os.path.getsize(dataFile)
    numSamples: float = (filesize - dataByteOffset) / recordByteSize
    if not numSamples.is_integer():
        errorPrint(
            "readB423Params",
            "Non-integer number of samples. Maybe the sampling frequency is incorrect",
            quitRun=True,
        )
    else:
        numSamples = int(numSamples)

    f = open(dataFile, "rb")
    hdrBytes = f.read(dataByteOffset)
    dataHeaders = readB423Header(hdrBytes)
    # read the first record and get the timestamp
    bts = f.read(6)
    firstTimestamp = struct.unpack("L", bts[0:4])[0]
    firstSampleNum = struct.unpack("H", bts[4:])[0]
    firstDatetime = datetime.utcfromtimestamp(
        firstTimestamp + (firstSampleNum / sampleFreq)
    )
    # now seek to the end
    f.seek(filesize - recordByteSize)
    bts = f.read(6)
    lastTimestamp = struct.unpack("L", bts[0:4])[0]
    lastSampleNum = struct.unpack("H", bts[4:])[0]
    lastDatetime = datetime.utcfromtimestamp(
        lastTimestamp + (lastSampleNum / sampleFreq)
    )

    # now check number of samples based on datetimes
    numSeconds = np.timedelta64(lastDatetime - firstDatetime) / np.timedelta64(1, "s")
    # +1 to numSamplesCalc because needs to be inclusive of the end
    numSamplesCalc = int(numSeconds * sampleFreq) + 1
    if numSamplesCalc != numSamples:
        errorPrint("readB423Params", "There is a gap in data file {}".format(dataFile))
        errorPrint("readB423Params", "No gaps allowed within data files")
        errorPrint(
            "readB423Params", "Please remove this file from the recording", quitRun=True
        )
    return dataHeaders, firstDatetime, lastDatetime, numSamples


def readB423Header(hdrStr: str):
    """Read B423 header for a single Lemi B423 file
    
    Parameters
    ----------
    hdrStr : str
        The headers as a string
    
    Returns
    -------
    dict 
        A dictionary of header values
    """

    hdr = hdrStr.decode()
    hdr = hdr.split("\r\n")
    headers = dict()
    for idx, h in enumerate(hdr):
        hdr[idx] = h.replace("%", "")
        if "=" in h:
            sp = hdr[idx].split("=")
            key = sp[0].strip()
            value = float(sp[1].strip())
            headers[key] = value
        if "Lat" in h or "Lon" in h or "Alt" in h:
            key = hdr[idx][0:3]
            value = hdr[idx][3:].strip()
            headers[key] = value
    return headers


class DataReaderLemiB423(DataReaderInternal):
    """Data reader for Lemi B423 data

    Lemi B423 data has the following characteristics:

    - 1024 bytes of ASCII headers in the data file with basic scaling information
    - There is no separate header file for Lemi B423 data. No information for number of samples, sampling rate etc
    - Header files need to be constructed before Lemi B423 data can be read in by resistics. There are helper methods to do this
    - Lemi B423 raw measurement data is signed long integer format   
    - Getting unscaled samples returns data with unit count for both the electric and magnetic fields. 
    - Scalings specified in B423 files convert electric channels to uV (microvolt) and magnetic channels to millivolts with internal gain still applied
    - Getting physical samples converts the electric channels to mV/km by dividing the uV by 1000 and then dividing by the dipole length in km
    - For the magnetic channels, the gain is removed to give the magnetic measurements in mV. Calibrating these will give measurements in nT 

    In situations where a Lemi B423 dataset is recorded in multiple files, it is required that the recording is continuous. 

    Attributes
    ----------
    dtype : np.int_
        The data type, a long integer
    recordByteSize : int
        The size of a record in bytes
    numHeaderFiles : int
        The number of header files
    numDataFiles : int
        The number of data files

    Methods
    -------
    setParameters()
        Set parameters specific to a data format
    getUnscaledSamples(**kwargs)
        Get raw, unscaled data
    getDataFilesForSamples(startSample, endSample)
        Get the data files that contribute to the request data
    readRecords(bts, numRecords)
        Read the data records from a provided bytes object
    getPhysicalSamples(**kwargs)
        Get data in physical units
    def readHeader()
        Read the B423 measurement file headers
    readMeasParams(sampleFreq)
        Calculate out the a B423 parameters including start and end date and number of samples
    getScalars(paramsDict)
        Get the scalars for each channel as given in the data files 
    printDataFileList()
        Get data file information as a list of strings
    printDataFiles()
        Print data file information to terminal
    """

    def setParameters(self) -> None:
        """Set some data reader parameters for reading Lemi B423 data"""

        # get a list of the header and data files in the folder
        self.headerExt = "h423"
        self.headerF = glob.glob(
            os.path.join(self.dataPath, "*.{}".format(self.headerExt))
        )
        self.dataF = glob.glob(os.path.join(self.dataPath, "*.B423"))
        # data byte information
        self.dataByteSize = 4
        self.recordByteSize = 30
        self.dataByteOffset = 1024
        # data type
        self.dtype = np.int_
        # get the number of data files and header files - this should be equal
        self.numHeaderFiles: int = len(self.headerF)
        self.numDataFiles: int = len(self.dataF)

    def getUnscaledSamples(self, **kwargs) -> TimeData:
        """Get raw data from data file, returned in mV

        Lemi B423 data always has five channels, in order Hx, Hy, Hz, Ex, Ey. The raw data is integer counts. Therefore, getting unscaled samples returns raw counts for the measurement. There are additional scalings which can be applied using the scale optional argument. Lemi B423 is recorded in multiple files. It has not been verified whether it is possible for each individual file to have different scaling. 

        Without the scale option, the data is returned in:
        
        - Counts for both magnetic and electric channels (reading long integers)

        With the scaling option, the data is returned in:

        - microvolts for the electric channels
        - millivolts for the magnetic with the gain applied

        Applying the scaling does not appear to remove the internal gain of the Lemi. This will be removed when getting physical samples and the appropriate value must be set in the headers.

        Parameters
        ----------
        chans : List[str], optional
            List of channels to return if not all are required
        startSample : int, optional
            First sample to return
        endSample : int, optional
            Last sample to return
        scale : bool, optional
            Boolean flag for applying the gain scaling

        Returns
        -------
        TimeData
            Time data object 
        """

        # initialise chans, startSample and endSample with the whole dataset
        options = self.parseGetDataKeywords(kwargs)

        # get the files to read and the samples to take from them, in the correct order
        dataFilesToRead, samplesToRead, scalings = self.getDataFilesForSamples(
            options["startSample"], options["endSample"]
        )
        numSamples = options["endSample"] - options["startSample"] + 1
        # set up the dictionary to hold the data
        dtype = np.float32 if options["scale"] else self.dtype
        data = {}
        for chan in options["chans"]:
            data[chan] = np.zeros(shape=(numSamples), dtype=dtype)

        # prepare comments
        startTime, stopTime = self.sample2time(
            options["startSample"], options["endSample"]
        )
        comments = []
        comments.append(
            "Unscaled data {} to {} read in from measurement {}, samples {} to {}".format(
                startTime,
                stopTime,
                self.dataPath,
                options["startSample"],
                options["endSample"],
            )
        )
        comments.append("Sampling frequency {}".format(self.getSampleFreq()))
        comments.append("Data read from {} files in total".format(len(dataFilesToRead)))
        comments.append("Scaling = {}".format(options["scale"]))

        # loop through chans and get data
        sampleCounter = 0
        for dFile, sToRead, scalar in zip(dataFilesToRead, samplesToRead, scalings):
            # calculate the starting byte and the number of bytes to read
            byteReadStart = self.dataByteOffset + sToRead[0] * self.recordByteSize
            dSamples = sToRead[1] - sToRead[0] + 1
            dSamplesRead = dSamples * self.getNumChannels()
            bytesToRead = dSamples * self.recordByteSize
            # read
            dFileHandle = open(dFile, "rb")
            dFileHandle.seek(byteReadStart, 0)  # seek to start byte from start of file
            dataBytes = dFileHandle.read(bytesToRead)
            dFileHandle.close()
            dataRead = self.readRecords(dataBytes, dSamples)

            # now need to unpack this
            for chan in options["chans"]:
                # check to make sure channel exists
                self.checkChan(chan)
                # get the channel index - the chanIndex should give the right order in the data file
                chanIndex = self.chanMap[chan]
                # use the range sampleCounter -> sampleCounter +  dSamples, because this actually means sampleCounter + dSamples - 1 as python ranges are not inclusive of the end value
                data[chan][sampleCounter : sampleCounter + dSamples] = dataRead[
                    chanIndex : dSamplesRead : self.getNumChannels()
                ]
                if options["scale"]:
                    data[chan][sampleCounter : sampleCounter + dSamples] = (
                        data[chan][sampleCounter : sampleCounter + dSamples]
                        * scalar[chan][0]
                        + scalar[chan][1]
                    )
                    comments.append(
                        "Scaling channel {} of file {} with multiplier {} and adding {}".format(
                            chan, dFile, scalar[chan][0], scalar[chan][1]
                        )
                    )
            # increment sample counter
            sampleCounter = sampleCounter + dSamples  # get ready for the next data read

        # return data
        return TimeData(
            sampleFreq=self.getSampleFreq(),
            startTime=startTime,
            stopTime=stopTime,
            data=data,
            comments=comments,
        )

    def getDataFilesForSamples(
        self, startSample: int, endSample: int
    ) -> Tuple[List[str], List[List[int]], List[float]]:
        """Get the data files that have to be read to cover the sample range 
        
        Parameters
        ----------
        startSample : int
            Starting sample of the sample range
        endSamples : int
            Ending sample of the sample range

        Returns
        -------
        dataFilesToRead
            Time data object
        """

        # have the datafiles saved in sample order beginning with the earliest first
        # go through each datafile and find the range to be read
        dataFilesToRead = []
        samplesToRead = []
        scalings = []
        for idx, dFile in enumerate(self.dataFileList):
            fileStartSamp = self.dataRanges[idx][0]
            fileEndSamp = self.dataRanges[idx][1]
            if fileStartSamp > endSample or fileEndSamp < startSample:
                continue  # nothing to read from this file
            # in this case, there is some overlap with the samples to read
            dataFilesToRead.append(dFile)
            readFrom = 0  # i.e. the first sample in the datafile
            readTo = fileEndSamp - fileStartSamp  # this the last sample in the file
            if fileStartSamp < startSample:
                readFrom = startSample - fileStartSamp
            if fileEndSamp > endSample:
                readTo = endSample - fileStartSamp
            # this is an inclusive number readFrom to readTo including readTo
            samplesToRead.append([readFrom, readTo])
            scalings.append(self.scalings[idx])
        return dataFilesToRead, samplesToRead, scalings

    def readRecords(self, bts: bytes, numRecords: int):
        """Read a number of B423 records from bytes
        
        Parameters
        ----------
        bts : bytes
            The bytes to be read
        numRecords : int
            The number of records to read. Size of bytes should be numRecords * recordByteSize
        
        Returns
        -------
        data : np.ndarray
            Array holding the data. The data repeats in channel blocks e.g. [Hx, Hy, Hx, Ex, Ey][Hx, Hy, Hx, Ex, Ey]
        """

        numChans = self.getNumChannels()
        structStr = "{:d}l".format(numChans)
        data = np.zeros(shape=(numRecords * numChans), dtype=self.dtype)
        for ii in range(0, numRecords):
            recordStart = ii * self.recordByteSize
            readBytes = bts[recordStart : recordStart + self.recordByteSize]
            recordValues = np.array(
                struct.unpack(structStr, readBytes[6:-4]), dtype=np.int_
            )
            dataStart = ii * numChans
            data[dataStart : dataStart + numChans] = recordValues
        return data

    def getPhysicalSamples(self, **kwargs):
        """Get data scaled to physical values
        
        resistics uses field units, meaning physical samples will return the following:

        - Electrical channels in mV/km
        - Magnetic channels in mV
        - To get magnetic fields in nT, calibration needs to be performed

        Notes
        -----
        Once Lemi B423 data is scaled (which optionally happens in getUnscaledSamples), the magnetic channels is in mV with gain applied and the electric channels is uV (microvolts). Therefore:
        
        - Electric channels need to divided by 1000 along with dipole length division in km (east-west spacing and north-south spacing) to return mV/km.
        - Magnetic channels need to be divided by the internal gain value which should be set in the headers
        
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
            chans=options["chans"],
            startSample=options["startSample"],
            endSample=options["endSample"],
            scale=True,
        )
        # convert to field units and divide by dipole lengths
        for chan in options["chans"]:
            if isElectric(chan):
                timeData.data[chan] = timeData.data[chan] / 1000.0
                timeData.addComment("Dividing channel {} by 1000 to convert microvolt to millivolt".format(chan))
            if isMagnetic(chan):
                timeData.data[chan] = timeData.data[chan] / self.getChanGain1(chan)
                timeData.addComment("Removing gain of {} from channel {}".format(self.getChanGain1(chan), chan))                
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
                    "Dividing channel {} by electrode distance {:.6f} km to give mV/km".format(
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

    def readHeader(self) -> None:
        """Read the B423 measurement file headers"""

        DataReaderInternal.readHeader(self)
        self.readMeasParams()

    def readMeasParams(self, sampleFreq: float = None) -> None:
        """Get the recording parameters for each of the subfiles
        
        Parameters
        ----------
        sampleFreq : float
            The sampling frequency in Hz
        """

        if sampleFreq is None:
            if "sample_freq" in self.headers:
                sampleFreq = float(self.getSampleFreq())
            else:
                self.printError(
                    "Lemi data has not been setup correctly. Please refer to documentation"
                )

        self.dataParamsDict: Dict = {}
        starts: List = []
        stops: List = []
        numSamples: List = []
        for dFile in self.dataF:
            # dataHeaders, startTime, stopTime, nSamples = self.readB423Params(
            #     dFile, sampleFreq
            # )
            dataHeaders, startTime, stopTime, nSamples = readB423Params(
                dFile, sampleFreq, self.dataByteOffset, self.recordByteSize
            )
            self.printText(
                "Reading data file {}, start = {}, end = {}, samples = {}".format(
                    dFile, startTime, stopTime, nSamples
                )
            )
            dFileParams: Dict = {}
            dFileParams["start"] = startTime
            dFileParams["stop"] = stopTime
            dFileParams["headers"] = dataHeaders
            starts.append(startTime)
            stops.append(stopTime)
            numSamples.append(nSamples)
            self.dataParamsDict[dFile] = dataHeaders

        # search for any missing data
        sampleTime = timedelta(seconds=1.0 / sampleFreq)
        # sort by start times
        sortIndices = sorted(list(range(len(starts))), key=lambda k: starts[k])
        check = True
        for ii in range(1, len(self.dataF)):
            # get the stop time of the previous dataset
            stopTimePrev = stops[sortIndices[ii - 1]]
            startTimeNow = starts[sortIndices[ii]]
            if startTimeNow != stopTimePrev + sampleTime:
                self.printWarning("There is a gap between the datafiles")
                self.printWarning(
                    "Please separate out datasets with gaps into separate folders"
                )
                self.printWarning("Gap found between datafiles:")
                self.printWarning("1. {}".format(self.dataF[sortIndices[ii - 1]]))
                self.printWarning("2. {}".format(self.dataF[sortIndices[ii]]))
                check = False
        # if did not pass check, then exit
        if not check:
            self.printError(
                "All data for a single recording must be continuous.", quitRun=True
            )

        # get a list of all the datafiles, scalings and the sample ranges
        self.dataFileList: List = []
        self.dataRanges: List = []
        self.scalings: List = []
        sample = -1
        # create lookup table to say where the sample ranges are
        for ii in range(0, len(self.dataF)):
            iSort = sortIndices[ii]  # get the sorted index
            self.dataFileList.append(self.dataF[iSort])
            startSample = sample + 1
            # end sample -1 because this is inclusive of the start sample
            endSample = startSample + numSamples[iSort] - 1
            self.dataRanges.append([startSample, endSample])
            # the scalings for the file
            self.scalings.append(self.getScalars(self.dataParamsDict[dFile]))
            # increment sample
            sample = endSample

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

        return {
            "Hx": [paramsDict["Kmx"], paramsDict["Ax"]],
            "Hy": [paramsDict["Kmy"], paramsDict["Ay"]],
            "Hz": [paramsDict["Kmz"], paramsDict["Az"]],
            "Ex": [paramsDict["Ke1"], paramsDict["Ae1"]],
            "Ey": [paramsDict["Ke2"], paramsDict["Ae2"]],
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
