import os
import glob
from copy import deepcopy
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Tuple, Any, Union

# import from package
from resistics.ioHandlers.ioHandler import IOHandler
from resistics.dataObjects.timeData import TimeData
from resistics.utilities.utilsPrint import blockPrint
from resistics.utilities.utilsChecks import isElectric, isMagnetic
from resistics.utilities.utilsClean import removeNansSingle, removeZerosSingle


class DataReader(IOHandler):
    """Base class for data readers

    DataReader is the base class for the time data readers. It implements much of the non-format dependent methods.

    The DataReaders for the different formats are meant to give a consistent output. Conventions used include:

    - The start time is the time of the first sample
    - The end time is the time of the last sample

    Physical data is always returned in:

    - Electrical channels in mV/km
    - Magnetic channels in mV
    - To get magnetic fields in nT, calibration needs to be performed    

    Attributes
    ----------
    dataPath : str
        Path to the data folder.
    headers : Dict
        Dictionary mapping header words to values
    chans : List[str]
        List of channels
    numChannels : int
        Number of channels
    chanHeaders : List	
        Headers specific to channels
    chanMap : Dict
        Map from channel name to index for chanHeaders
    comments : List[str]
        List of comments associated with data
    headerF	: List[str]
        List of header files (with extension .hdr)
    dataF : List[str]
        List of data files (with extension .npy)
    dataByteOffset : int
        Number of bytes to offset before reading
    dataByteSize : int
        Byte size of a single data value

    Methods
    -------
    __init__(dataPath)
        Initialise with path to the data directory
    setParameters()
        Set parameters specific to a data format
    checkFiles()
        Check to see header and data files found in data directory
    getComments()
        Get a deepcopy of the comments
    getHeaders()
        Get copy of header dictionary
    getChannels()
        Get a list of the channels
    getNumChannels()   
        Get number of channels
    getSampleFreq(integer = False)
        Get sampling frequency in Hz as float or optionally as integer
    getSampleRate()
        Get sample rate in s
    getNumSamples()
        Get the number of samples
    getStartDatetime()
        Get the data start time as a datetime object
    getStopDatetime()
        Get the data stop time as a datetime object
    getGain1() 
        Get gain stage 1 for each channel in an array
    getGain2()
        Get gain stage 2 for each channel in an array
    getChanMap()
        Get the channel map
    getChanHeader(chan, header)
        Get a channel header
    getChanType(chan)
        Get channel type (electric, magnetic)
    getChanGain1(chan)
        Get gain stage 1 for chan
    getChanGain2(chan)
        Get gain stage 2 for chan
    getChanSamples(chan)
        Get number of samples in channel
    getChanSampleFreq(chan)
        Get sampling frequency for a channel
    getChanLSB(chan)
        Get the channel least significant bit
    getChanScalingApplied(chan)
        Get a bool flag designating whether LSB has been applied
    getChanDataFile(chan)
        Get the channel data file
    getChanDx(chan)
        Get the distance between x electrodes
    getChanDy(chan)
        Get the distance between y electrodes
    getChanDz(chan)
        Get the distance between z electrodes
    getChanSensor(chan)
        Get sensor value of a channel
    getChanSerial(chan)
        Get serial value of a channel
    getChanChopper(chan)
        Get chopper value for a channel
    getSensors(chans)
        Get sensor values for a list of channels
    getSerials(chans)
        Get serial values for a list of channels
    getChoppers(chans)
        Get chopper values for a list of channels
    setHeader(headerName, headerVal) 
        Set a header value
    setChanHeader(chan, headerName, headerVal)
        Set a chan header value
    getUnscaledSamples(**kwargs)
        Get raw, unscaled data
    getUnscaledData(startTime, endTime, **kwargs)
        Get raw, unscaled data for a date range
    getPhysicalSamples(**kwargs)
        Get data in physical units
    getPhysicalData(startTime, endTime, **kwargs)
        Get data in physical units for a date range
    parseGetDataKeywords(keywords)
        Parse get data keywords
    time2sample(timeStart, timeEnd)
        Convert dates to samples
    sample2time(sampleStart, sampleEnd)
        Convert samples to datetimes
    getDataTimes(timeStart, timeEnd)
        Check data times and return checked and corrected ones
    readHeader()
        Read header data (implemented in child classes)
    formatHeaderData()
        Format the header data
    intHeaders()
        Return a list of headers to be formatted as int    
    floatHeaders()
        Return a list of headers to be formatted as float
    boolHeaders()
        Return a list of headers to be formatted as bool
    prepare()
        Prepare class information after reading header files
    checkChan(chan)
        Check channel exists in data
    printList()
        Class information as list of strings
    printCommentsList()
        Comment information as list of strings
    printComments()
        Print comments to terminal
    """

    def __init__(self, dataPath: str) -> None:
        """Initialise with path to data directory

        Parameters
        ----------
        dataPath : str
            Path to data directory
        """

        self.dataPath: str = dataPath
        self.headers: Dict = {}
        self.chans = []
        self.numChannels: int = 0
        self.chanHeaders: List = []
        self.chanMap: Dict = {}
        self.comments: List[str] = []
        # get a list of the xml files in the folder
        self.setParameters()
        if not self.checkFiles():
            self.printError("No header or data files found", quitRun=True)
        self.readHeader()
        self.formatHeaderData()
        self.prepare()

    def setParameters(self) -> None:
        """Set data reader parameters

        This will vary for the different data formats. By default, setup for the internal data format.
        """

        self.headerF = glob.glob(os.path.join(self.dataPath, "*.hdr"))
        self.dataF = glob.glob(os.path.join(self.dataPath, "*.dat"))
        self.dataByteOffset = 0
        self.dataByteSize = 4

    def checkFiles(self) -> bool:
        """Check to make sure data files found"""

        check = True
        if len(self.headerF) == 0:
            check = check and False
            self.printWarning("No header files found")
        if len(self.dataF) == 0:
            check = check and False
            self.printWarning("No data files found")
        return check

    def getComments(self) -> List[str]:
        """Get a deepcopy of the comments
        
        Returns
        -------
        List[str]
            Dataset comments as a list of strings
        """

        return deepcopy(self.comments)

    def getHeaders(self) -> Dict:
        """Get the data headers

        Returns
        -------
        Dict
            Data headers
        """

        return deepcopy(self.headers)

    def getChannels(self) -> List[str]:
        """Get channels in data

        Returns
        -------
        List[str]
            Data channels
        """

        return deepcopy(self.chans)

    def getNumChannels(self) -> int:
        """Get number of channels in the data

        Returns
        -------
        int
            Number of channels
        """

        return self.headers["meas_channels"]

    def getSampleFreq(self, integer: bool = False):
        """Get data sampling frequency in Hz

        Returns
        -------
        float, int
            Sampling frequency
        """

        if integer:
            return int(self.headers["sample_freq"])
        return self.headers["sample_freq"]

    def getSampleRate(self) -> float:
        """Get data sampling rate in  s

        Returns
        -------
        float
            Sampling rate in s
        """

        return 1.0 / self.getSampleFreq()

    def getNumSamples(self) -> int:
        """Get number of samples

        Returns
        -------
        int
            Number of samples
        """

        return self.headers["num_samples"]

    def getStartDatetime(self) -> datetime:
        """Get datetime of first sample

        Returns
        -------
        datetime
            Date and time of first sample
        """

        return deepcopy(self.datetimeStart)

    def getStopDatetime(self) -> datetime:
        """Get datetime of last sample

        Returns
        -------
        datetime
            Date and time of last sample
        """

        return deepcopy(self.datetimeStop)

    def getGain1(self) -> np.ndarray:
        """Get value of gain 1

        Returns
        -------
        np.ndarray
            Array of gains for channels
        """

        gain1 = np.zeros(shape=(self.numChannels), dtype=bool)
        for iChan in range(0, self.numChannels):
            gain1[iChan] = self.getChanGain1(iChan)
        return gain1

    def getGain2(self) -> np.ndarray:
        """Get value of gain 2

        Returns
        -------
        np.ndarray
            Array of gains for channels
        """

        gain2 = np.zeros(shape=(self.numChannels), dtype=bool)
        for iChan in range(0, self.numChannels):
            gain2[iChan] = self.getChanGain2(iChan)
        return gain2

    def getChanHeaders(self, chans: List[str] = []) -> Tuple[List, Dict]:
        """Get channel headers

        Parameters
        ----------
        chans: List[str], optional
            List of channels for which chan headers are wanted
        Returns
        -------
        chanHeaders : List
            List of channel headers
        chanMap : Dict
            Map from channel to index for the chanHeaders
        """

        if len(chans) == 0:
            return deepcopy(self.chanHeaders), deepcopy(self.chanMap)

        chanHeaders = []
        chanMap = {}
        for idx, c in enumerate(chans):
            chanIdx = self.getChanMap()[c]
            chanHeaders.append(self.chanHeaders[chanIdx])
            chanMap[c] = idx
        return chanHeaders, chanMap

    def getChanMap(self):
        return self.chanMap

    def getChanHeader(self, chan, header):
        self.checkChan(chan)
        iChan = self.chanMap[chan]
        return self.chanHeaders[iChan][header]

    def getChanType(self, chan) -> str:
        """Get the channel type (electric or magnetic)

        Returns
        -------
        str
            String of channel type
        """

        return self.getChanHeader(chan, "channel_type")

    def getChanGain1(self, chan) -> int:
        """Get channel gain 1 

        Returns
        -------
        int
            Channel gain 1
        """

        return self.getChanHeader(chan, "gain_stage1")

    def getChanGain2(self, chan) -> int:
        """Get channel gain 2

        Returns
        -------
        int
            Channel gain 2
        """

        return self.getChanHeader(chan, "gain_stage2")

    def getChanSamples(self, chan) -> int:
        """Get channel number of samples

        Returns
        -------
        int
            Channel number of samples
        """

        return self.getChanHeader(chan, "num_samples")

    def getChanSampleFreq(self, chan) -> float:
        """Get channel sampling frequency 

        Returns
        -------
        float
            Sampling frequency in Hz
        """

        return self.getChanHeader(chan, "sample_freq")

    def getChanLSB(self, chan):
        """Get channel least significant bit 

        Returns
        -------
        float
            Channel least significant bit
        """

        return self.getChanHeader(chan, "ts_lsb")

    def getChanScalingApplied(self, chan) -> bool:
        """A flag to mark whether a channel has the lsb applied

        Returns
        -------
        bool
            Flag to designate whether channel lsb applied
        """

        return self.getChanHeader(chan, "scaling_applied")

    def getChanDataFile(self, chan) -> str:
        """Get the data file for the channel 

        Returns
        -------
        str
            Data file for the channel
        """

        return self.getChanHeader(chan, "ats_data_file")

    def getChanDx(self, chan):
        """Get the electric channel spacing in the x direction 

        Returns
        -------
        float
            Electric channel spacing in x direction in metres
        """

        x1 = np.absolute(self.getChanHeader(chan, "pos_x1"))
        x2 = np.absolute(self.getChanHeader(chan, "pos_x2"))
        return x2 + x1

    def getChanDy(self, chan):
        """Get the electric channel spacing in the y direction 

        Returns
        -------
        float
            Electric channel spacing in y direction in metres
        """

        y1 = np.absolute(self.getChanHeader(chan, "pos_y1"))
        y2 = np.absolute(self.getChanHeader(chan, "pos_y2"))
        return y2 + y1

    def getChanDz(self, chan):
        """Get the electric channel spacing in the z direction 

        Returns
        -------
        float
            Electric channel spacing in z direction in metres
        """

        z1 = np.absolute(self.getChanHeader(chan, "pos_z1"))
        z2 = np.absolute(self.getChanHeader(chan, "pos_z2"))
        return z2 + z1

    def getChanSensor(self, chan) -> str:
        """Get channel sensor type

        Returns
        -------
        str
            Channel sensor type
        """

        return self.getChanHeader(chan, "sensor_type")

    def getChanSerial(self, chan) -> str:
        """Get channel serial number

        Returns
        -------
        str
            Channel serial number
        """

        return self.getChanHeader(chan, "sensor_sernum")

    def getChanChopper(self, chan) -> bool:
        """Get channel chopper
        
        The chopper is an amplifier present in some instruments. There might be different calibration files for chopper on or off.

        Returns
        -------
        bool
            Flag designating whether chopper is on or off
        """

        echopper = self.getChanHeader(chan, "echopper")
        hchopper = self.getChanHeader(chan, "hchopper")
        # return true if the chopper amplifier was on
        if isElectric(chan) and echopper:
            return True
        if isMagnetic(chan) and hchopper:
            return True
        return False

    def getSensors(self, chans: List[str]) -> Dict[str, str]:
        """Get sensors for multiple chans

        Returns
        -------
        Dict[str, str]
            Dictionary with channels as keys and the sensor types as values 
        """

        sensors = {}
        for chan in chans:
            sensors[chan] = self.getChanSensor(chan)
        return sensors

    def getSerials(self, chans: List[str]) -> Dict[str, str]:
        """Get serials for multiple chans

        Returns
        -------
        Dict[str, str]
            Dictionary with channels as keys and the serials numbers as values 
        """

        serials = {}
        for chan in chans:
            serials[chan] = self.getChanSerial(chan)
        return serials

    def getChoppers(self, chans: List[str]) -> Dict[str, bool]:
        """Get choppers for multiple chans

        Returns
        -------
        Dict[str, str]
            Dictionary with channels as keys and the serials numbers as values 
        """

        choppers = {}
        for chan in chans:
            choppers[chan] = self.getChanChopper(chan)
        return choppers

    def setHeader(self, headerName: str, headerVal: Any) -> None:
        """Set a header value

        Parameters
        ----------
        headerName : str
            The name of the header to set
        headerVal : Any
            Header value
        """

        self.headerName = headerVal

    def setChanHeader(self, chan: str, headerName: str, headerVal: Any) -> None:
        """Set a channel header value

        Parameters
        ----------
        channel : str
            The channel
        headerName : str
            The name of the header to set
        headerVal : Any
            Header value
        """

        chanIndex = self.chanMap[chan]
        self.chanHeaders[chanIndex][headerName] = headerVal

    def getUnscaledSamples(self, **kwargs) -> TimeData:
        """Get raw data from data file

        Depending on the data format, this could be raw counts or in some physical unit. The method implemented in the base DataReader can read from ATS and internal files. SPAM and Phoenix data readers have their own implementations.

        The raw data units for ATS and internal data formats are as follows: 

        - ATS data format has raw data in counts.
        - The raw data unit of the internal format is dependent on what happened to the data before writing it out in the internal format. If the channel header scaling_applied is set to True, no scaling happens in either getUnscaledSamples or getPhysicalSamples. However, if the channel header scaling_applied is set to False, the internal format data will be treated like ATS data, meaning raw data in counts.
        
        Parameters
        ----------
        chans : List[str], optional
            List of channels to return if not all are required
        startSample : int, optional
            First sample to return
        endSample : int, optional
            Last sample to return

        Returns
        -------
        TimeData
            Time data object
        """

        # initialise chans, startSample and endSample with the whole dataset
        options = self.parseGetDataKeywords(kwargs)
        # get samples - this is inclusive
        dSamples = options["endSample"] - options["startSample"] + 1

        # loop through chans and get data
        data = {}
        for chan in options["chans"]:
            # check to make sure channel exists
            self.checkChan(chan)
            # get data file
            dFile = os.path.join(self.dataPath, self.getChanDataFile(chan))
            # get the data
            byteOff = self.dataByteOffset + options["startSample"] * self.dataByteSize
            # now check if lsb applied or not and read data as float32 or int32 accordingly
            if self.getChanScalingApplied(chan):
                data[chan] = np.memmap(
                    dFile, dtype="float32", mode="r", offset=byteOff, shape=(dSamples)
                )
            else:
                data[chan] = np.memmap(
                    dFile, dtype="int32", mode="r", offset=byteOff, shape=(dSamples)
                )

        # get data start and stop time
        startTime, stopTime = self.sample2time(
            options["startSample"], options["endSample"]
        )
        # dataset comments
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
        if len(self.comments) > 0:
            comments = self.comments + comments
        return TimeData(
            sampleFreq=self.getSampleFreq(),
            startTime=startTime,
            stopTime=stopTime,
            data=data,
            comments=comments,
        )

    def getUnscaledData(self, startTime, endTime, **kwargs) -> TimeData:
        """Get raw data from data file between a start and end date

        Calculates the start and end sample given the data range and returns getUnscaledSamples for that sample range.

        Parameters
        ----------
        startTime : datetime
            Start time of data to read
        endTime : datetime
            End time of data to read       
        chans : List[str], optional
            List of channels to return if not all are required

        Returns
        -------
        TimeData
            Time data object
        """

        options = self.parseGetDataKeywords(kwargs)
        startSample, endSample = self.time2sample(startTime, endTime)
        return self.getUnscaledSamples(
            chans=options["chans"], startSample=startSample, endSample=endSample
        )

    def getPhysicalSamples(self, **kwargs):
        """Get data scaled to physical values
        
        Depending on the data format, the scalings required to convert to field physical units is different. The method in the base DataReader class covers ATS and internal file format.

        resistics will always provide physical samples in field units. That means

        - Electrical channels in mV/km
        - Magnetic channels in mV
        - To get magnetic fields in nT, calibration needs to be performed
        
        If the channel header scaling_applied is set to True, no scaling of the unscaled data is done. This is to cover the internal data format where all scalings may already have been applied.

        Notes
        -----
        The raw data units for ATS data are in counts. To get data in field units, ATS data is first multipled by the least significat bit (lsb) defined in the header files,

        .. code-block:: text  
        
            data = data * leastSignificantBit,
        
        giving data in mV. The lsb includes the gain removal, so no separate gain removal needs to be performed.
        
        For electrical channels, there is additional step of dividing by the electrode spacing, which is provided in metres. The extra factor of a 1000 is to convert this to km to give mV/km for electric channels
        
        .. code-block:: text  
            
            data = (1000 * data)/electrodeSpacing
        
        Finally, to get magnetic channels in nT, the magnetic channels need to be calibrated.

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

        options = self.parseGetDataKeywords(kwargs)
        timeData = self.getUnscaledSamples(
            chans=options["chans"],
            startSample=options["startSample"],
            endSample=options["endSample"],
        )
        # multiply each chan by least significant bit of chan
        for chan in options["chans"]:
            if not self.getChanScalingApplied(chan):
                # apply LSB to give data in mV
                timeData.data[chan] = timeData.data[chan] * self.getChanLSB(chan)
                timeData.addComment(
                    "Scaling channel {} with scalar {} to give mV".format(
                        chan, self.getChanLSB(chan)
                    )
                )

                # divide by the distance - this should only be for the electric channels
                # again, this might already be applied
                if chan == "Ex":
                    # multiply by 1000/self.getChanDx same as dividing by dist in km
                    timeData.data[chan] = (
                        1000 * timeData.data[chan] / self.getChanDx(chan)
                    )
                    timeData.addComment(
                        "Dividing channel {} by electrode distance {} km to give mV/km".format(
                            chan, self.getChanDx(chan) / 1000.0
                        )
                    )
                if chan == "Ey":
                    # multiply by 1000/self.getChanDy same as dividing by dist in km
                    timeData.data[chan] = (
                        1000 * timeData.data[chan] / self.getChanDy(chan)
                    )
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
            # do this after all scaling and removing nans and zeros
            if options["remaverage"]:
                timeData.data[chan] = timeData.data[chan] - np.average(
                    timeData.data[chan]
                )

        timeData.addComment(
            "Remove zeros: {}, remove nans: {}, remove average: {}".format(
                options["remzeros"], options["remnans"], options["remaverage"]
            )
        )
        return timeData

    def getPhysicalData(self, startTime, endTime, **kwargs):
        """Get physical data from data file between a start and end data

        Calculates the start and end sample given the data range and returns getPhysicalSamples for that sample range.
        
        Parameters
        ----------
        startTime : datetime
            Start time of data to read
        endTime : datetime
            End time of data to read       
        chans : List[str]
            List of channels to return if not all are required
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

        options = self.parseGetDataKeywords(kwargs)
        startSample, endSample = self.time2sample(startTime, endTime)
        return self.getPhysicalSamples(
            chans=options["chans"], startSample=startSample, endSample=endSample
        )

    def parseGetDataKeywords(self, keywords) -> Dict:
        """Parse the get data keywords
        
        Parameters
        ----------
        keywords : Dict
            The keywords passed to get data methods

        Returns
        -------
        Dict
            A dictionary of parsed keywords with defaults where nothing is provided by the user
        """

        # defaults
        options = {}
        options["chans"] = self.getChannels()
        options["startSample"] = 0
        options["endSample"] = self.getNumSamples() - 1
        options["startTime"] = self.getStartDatetime()
        options["endTime"] = self.getStopDatetime()
        options["remaverage"] = True
        options["remzeros"] = False
        options["remnans"] = False
        # now take the options from the keywords
        for w in options:
            if w in keywords:
                options[w] = keywords[w]
        # do some checks
        if options["endSample"] >= self.getNumSamples():
            options["endSample"] = self.getNumSamples() - 1
            self.printWarning(
                "End sample greater than number of samples. Adjusted to {:d}".format(
                    options["endSample"]
                )
            )
        if options["startSample"] >= options["endSample"]:
            options["startSample"] = (
                options["endSample"] - 100
            )  # let's give 100 samples
            self.printWarning(
                "Start sample greater than end sample. Adjusted to {:d}".format(
                    options["startSample"]
                )
            )
        if options["startSample"] < 0:
            options["startSample"] = 0
            self.printWarning(
                "Start sample < 0. Adjusted to {:d}".format(options["startSample"])
            )
        return options

    def time2sample(
        self, timeStart: Union[str, datetime], timeEnd: Union[str, datetime]
    ) -> Tuple[int, int]:
        """Converts a start and end time to start and end samples
        
        Note: The first sample is zero

        Parameters
        ----------
        timeStart : datetime, str
            Start time of date range
        timeEnd : datetime, str
            End time of date range       

        Returns
        -------
        sampleStart : int
            The correspoding start sample for timeStart
        sampleEnd : int
            The corresponding end sample for timeEnd
        """

        # if timeStart and timeEnd are strings, then convert them to datetime objects
        if isinstance(timeStart, str):
            timeStart = datetime.strptime(timeStart, "%Y-%m-%d %H:%M:%S")
        if isinstance(timeEnd, str):
            timeEnd = datetime.strptime(timeEnd, "%Y-%m-%d %H:%M:%S")
        # check to see times within range
        timeStart, timeEnd = self.getDataTimes(timeStart, timeEnd)
        # start sample
        deltaStart = timeStart - self.getStartDatetime()
        sampleStart = deltaStart.total_seconds() * self.getSampleFreq()
        sampleStart = int(
            round(sampleStart)
        )  # this will hopefully deal with fractional sampling
        # end sample
        deltaEnd = timeEnd - timeStart
        deltaSamples = deltaEnd.total_seconds() / self.getSampleRate()
        deltaSamples = int(round(deltaSamples))
        sampleEnd = sampleStart + deltaSamples
        # return samples
        return sampleStart, sampleEnd

    def sample2time(
        self, sampleStart: int, sampleEnd: int
    ) -> Tuple[datetime, datetime]:
        """Converts a start and end sample to start and end times
        
        Note: The first sample is zero

        Parameters
        ----------
        sampleStart : int
            The starting sample for the sample range
        sampleEnd : int
            The ending sample for the sample range   

        Returns
        -------
        timeStart : datetime, str
            Corresponding start time of date range
        timeEnd : datetime, str
            Corresponding end time of date range            
        """

        # convert samples to some data format
        deltaStart = timedelta(seconds=self.getSampleRate() * sampleStart)
        # delta end is inclusive
        deltaEnd = timedelta(seconds=self.getSampleRate() * (sampleEnd - sampleStart))
        timeStart = self.getStartDatetime() + deltaStart
        timeEnd = timeStart + deltaEnd
        return timeStart, timeEnd

    def getDataTimes(self, timeStart, timeEnd):
        """Checks and converts a date range to make sure it's within data start and end
        
        Parameters
        ----------
        timeStart : datetime, str
            Start time of date range
        timeEnd : datetime, str
            End time of date range     

        Returns
        -------
        timeStart : datetime, str
            Checked and corrected start time of data range
        timeEnd : datetime, str
            Checked and corrected end time of data range        
        """

        deltaStart = timeStart - self.getStartDatetime()
        deltaEnd = self.getStopDatetime() - timeEnd
        if deltaStart.total_seconds() < 0:
            self.printText(
                "Date {} before start of recording. Start date adjusted to {}".format(
                    timeStart, self.getStartDatetime()
                )
            )
            timeStart = self.getStartDatetime()
        if deltaEnd.total_seconds() < 0:
            self.printText(
                "Date {} after end of recording. Stop date adjusted to {}".format(
                    timeEnd, self.getStopDatetime()
                )
            )
            timeEnd = self.getStopDatetime()
        return timeStart, timeEnd

    def readHeader(self) -> None:
        """Function to read header data and populate reader information
        
        This is implemented in child classes as all header formats are different
        """

        raise NotImplementedError(
            "Read header not implemented in parent class. Only child classes should ever be instantiated."
        )

    def formatHeaderData(self) -> None:
        """Format header data to the correct formats"""

        # do the int formatting
        intGlobal, intChan = self.intHeaders()
        floatGlobal, floatChan = self.floatHeaders()
        boolGlobal, boolChan = self.boolHeaders()
        # deal with the global headers
        for h in intGlobal:
            self.headers[h] = int(self.headers[h])
        for h in floatGlobal:
            self.headers[h] = float(self.headers[h])
        for h in boolGlobal:
            self.headers[h] = bool(self.headers[h])
        # deal with the channel headers
        numChans = len(self.chanHeaders)
        for iChan in range(0, numChans):
            for cH in self.chanHeaders[iChan]:
                if cH in intChan:
                    self.chanHeaders[iChan][cH] = int(self.chanHeaders[iChan][cH])
                if cH in floatChan:
                    self.chanHeaders[iChan][cH] = float(self.chanHeaders[iChan][cH])
                if cH in boolChan:
                    if isinstance(
                        self.chanHeaders[iChan][cH], str
                    ):  # if string, parse true and false
                        if self.chanHeaders[iChan][cH] == "True":
                            self.chanHeaders[iChan][cH] = True
                        else:
                            self.chanHeaders[iChan][cH] = False
                    else:  # a bool or an integer
                        self.chanHeaders[iChan][cH] = bool(self.chanHeaders[iChan][cH])

    def intHeaders(self) -> Tuple[List[str], List[str]]:
        """List of headers which are expected to have integer values
        
        Returns
        -------
        intGlobal : List[str]
            List of global integer headers
        intChan : List[str]
            List of chan integer headers
        """

        intGlobal = ["meas_channels"]
        intChan = [
            "gain_stage1",
            "gain_stage2",
            "hchopper",
            "echopper",
            "num_samples",
            "sensor_sernum",
        ]
        return intGlobal, intChan

    def floatHeaders(self) -> Tuple[List[str], List[str]]:
        """List of headers which are expected to have boolean values
        
        Returns
        -------
        floatGlobal : List[str]
            List of global float headers
        floatChan : List[str]
            List of chan float headers
        """

        floatGlobal = ["sample_freq"]
        floatChan = [
            "sample_freq",
            "ts_lsb",
            "pos_x1",
            "pos_x2",
            "pos_y1",
            "pos_y2",
            "pos_z1",
            "pos_z2",
        ]
        return floatGlobal, floatChan

    def boolHeaders(self) -> Tuple[List[str], List[str]]:
        """List of headers which are expected to have boolean values
        
        Returns
        -------
        boolGlobal : List[str]
            List of global boolean headers
        boolChan : List[str]
            List of chan boolean headers
        """

        boolGlobal = []
        boolChan = ["scaling_applied"]
        return boolGlobal, boolChan

    def prepare(self) -> None:
        """Set some intial values
        
        This method does some checks and prepares some of the storage for the channels.                 
        
        Notes
        -----
        The end time of the data will be checked. Different data formats record an end time after the last sample. For example, ATS end time appears to be one sample after the number of samples. The end time is checked by using the number of samples and the sampling frequency. 
        
        The internal convention is that the start and end times should reflect the times of the first and last sample
        """

        # create the type - index map
        self.chans = []
        self.chanMap = {}
        for iChan in range(0, self.getNumChannels()):
            chanType = self.chanHeaders[iChan]["channel_type"]
            self.chanMap[chanType] = iChan
            self.chans.append(chanType)

        # check the number of samples of each channel
        numSamples = []
        for c in self.chans:
            numSamples.append(self.getChanSamples(c))

        self.headers["num_samples"] = min(numSamples)
        for c, n in zip(self.chans, numSamples):
            if n != self.getNumSamples():
                self.printWarning("Not all channels have the same number of samples")
                self.printWarning(
                    "{} has {:d} samples more than the minimum".format(
                        c, n - self.getNumSamples()
                    )
                )

        # create datetime objects
        datetimeStart = "{} {}".format(
            self.headers["start_date"], self.headers["start_time"]
        )
        datetimeStop = "{} {}".format(
            self.headers["stop_date"], self.headers["stop_time"]
        )
        self.datetimeStart = datetime.strptime(datetimeStart, "%Y-%m-%d %H:%M:%S.%f")
        self.datetimeStop = datetime.strptime(datetimeStop, "%Y-%m-%d %H:%M:%S.%f")

        # check the stop time
        startTime, endTime = self.sample2time(0, self.getNumSamples() - 1)
        if endTime != self.datetimeStop:
            self.datetimeStop = endTime
            self.headers["stop_date"] = self.datetimeStop.strftime("%Y-%m-%d")
            self.headers["stop_time"] = self.datetimeStop.strftime("%H:%M:%S.%f")
            for idx, chan in enumerate(self.chanHeaders):
                self.chanHeaders[idx]["stop_date"] = self.datetimeStop.strftime(
                    "%Y-%m-%d"
                )
                self.chanHeaders[idx]["stop_time"] = self.datetimeStop.strftime(
                    "%H:%M:%S.%f"
                )

    def checkChan(self, chan: str) -> None:
        """Check channel exists in data

        Parameters
        ----------
        chan : str
            Channel to check
        """

        if chan not in self.chans:
            self.printError(
                "Error - Channel {} does not exist".format(chan), quitRun=True
            )

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """

        textLst = []
        textLst.append("Data Path = {}".format(self.dataPath))
        textLst.append("Global Headers")
        textLst.append("{}".format(self.headers))
        textLst.append("Channels found:")
        textLst.append("{}".format(self.chans))
        textLst.append("Channel Map")
        textLst.append("{}".format(self.chanMap))
        textLst.append("Channel Headers")
        for c in self.chans:
            textLst.append(c)
            textLst.append("{}".format(self.chanHeaders[self.chanMap[c]]))
        textLst.append(
            "Note: Field units used. Physical data has units mV/km for electric fields and mV for magnetic fields"
        )
        textLst.append("Note: To get magnetic field in nT, please calibrate")
        return textLst

    def printCommentsList(self) -> List[str]:
        """Dataset comments as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """

        textLst: List[str] = []
        textLst.append("Dataset Comments")
        if len(self.comments) == 0:
            textLst.append("No comments")
        else:
            for idx, comment in enumerate(self.comments):
                textLst.append("{:02d} : {}".format(idx, comment))
        return textLst

    def printComments(self) -> None:
        """Print out dataset comments"""

        blockPrint(
            "{} Comments".format(self.__class__.__name__), self.printCommentsList()
        )

