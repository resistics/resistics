import os
import glob
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import numpy as np
from typing import List, Tuple, Dict, Any

# import from package
from resistics.ioHandlers.dataReader import DataReader
from resistics.dataObjects.timeData import TimeData
from resistics.utilities.utilsChecks import isMagnetic, isElectric, consistentChans
from resistics.utilities.utilsPrint import blockPrint
from resistics.utilities.utilsClean import (
    removeZeros,
    removeZerosSingle,
    removeNansSingle,
)


class DataReaderSPAM(DataReader):
    """Data reader for SPAM data

    SPAM data has the following characteristics:

    - SPAM raw data is single precision floats with unit Volts. 
    - Getting unscaled samples returns data with unit mV for both the electric and magnetic fields. This is because gain is removed in unscaled samples to ensure consistency when a single recording is made up of multiple data files, each with different gain settings
    - The start time in XTR files is the time of the first sample in the data
    - The end time in XTR files is the time of the last sample in the data

    In situations where a SPAM dataset is recorded in multiple small files, it is required that the recording is continuous. 

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

    Notes
    -----
    Getting unscaled samples for SPAM data removes the gain rather than return exactly the values in the data files. In cases where there are multiple data files, it is not necessary that they have been recorded with the same gain. Therefore, to ensure consistency when looking at raw data, the gain is removed at the getUnscaledSamples stage rather than getPhysicalSamples, where it would have probably been more appropriate. This means that getUnscaledSamples returns data where all channels are in mV.

    The scalings to convert the raw data to mV are stored in the ts_lsb chan header and calculated out as the header files are being read.

    .. todo::
        Implement reading of XTRX header files
    """

    def setParameters(self) -> None:
        """Set some data reader parameters for reading SPAM data"""

        # get a list of the header and data files in the folder
        self.headerF = glob.glob(os.path.join(self.dataPath, "*.XTR"))
        if len(self.headerF) == 0:
            self.headerF = glob.glob(os.path.join(self.dataPath, "*.XTRX"))
        self.dataF = glob.glob(os.path.join(self.dataPath, "*.RAW"))
        # data byte information might be different for each file
        # so it is a dictionary
        self.dataByteOffset: Dict = {}
        self.recChannels = {}
        self.dataByteSize = 4
        # data type
        self.dtype = np.float32
        # get the number of data files and header files - this should be equal
        self.numHeaderFiles: int = len(self.headerF)
        self.numDataFiles: int = len(self.dataF)

    def getUnscaledSamples(self, **kwargs) -> TimeData:
        """Get raw data from data file, returned in mV

        SPAM raw data is single precision float with unit Volts. Calling this applies the ts_lsb calculated when the headers are read. This is because when a recording consists of multiple data files, each channel of each data file might have a different scaling. The only way to make the data consistent is to apply the ts_lsb scaling.  
        
        Therefore, this method returns the data in mV for all channels.

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

        # get the files to read and the samples to take from them, in the correct order
        dataFilesToRead, samplesToRead, scalings = self.getDataFilesForSamples(
            options["startSample"], options["endSample"]
        )
        numSamples = options["endSample"] - options["startSample"] + 1
        # set up the dictionary to hold the data
        data = {}
        for chan in options["chans"]:
            data[chan] = np.zeros(shape=(numSamples), dtype=self.dtype)

        # loop through chans and get data
        sampleCounter = 0
        for dFile, sToRead, scalar in zip(dataFilesToRead, samplesToRead, scalings):
            # get samples - this is inclusive
            dSamples = sToRead[1] - sToRead[0] + 1
            dSamplesRead = (
                dSamples * self.recChannels[dFile]
            )  # because spam files always record 5 channels
            # read the data
            byteOff = (
                self.dataByteOffset[dFile]
                + sToRead[0] * self.recChannels[dFile] * self.dataByteSize
            )
            dFilePath = os.path.join(self.dataPath, dFile)
            dataRead = np.memmap(
                dFilePath,
                dtype=self.dtype,
                mode="r",
                offset=byteOff,
                shape=(dSamplesRead),
            )
            # now need to unpack this
            for chan in options["chans"]:
                # check to make sure channel exists
                self.checkChan(chan)
                # get the channel index - the chanIndex should give the right order in the data file
                # as it is the same order as in the header file
                chanIndex = self.chanMap[chan]
                # use the range sampleCounter -> sampleCounter +  dSamples, because this actually means sampleCounter + dSamples - 1 as python ranges are not inclusive of the end value
                # scale by the lsb scalar here - note that these can be different for each file in the run
                data[chan][sampleCounter : sampleCounter + dSamples] = (
                    dataRead[chanIndex : dSamplesRead : self.recChannels[dFile]]
                    * scalar[chan]
                )
            # increment sample counter
            sampleCounter = sampleCounter + dSamples  # get ready for the next data read

        # return data
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
        comments.append("Data read from {} files in total".format(len(dataFilesToRead)))
        comments.append(
            "Data scaled to mV for all channels using scalings in header files"
        )
        comments.append("Sampling frequency {}".format(self.getSampleFreq()))
        return TimeData(
            sampleFreq=self.getSampleFreq(),
            startTime=startTime,
            stopTime=stopTime,
            data=data,
            comments=comments,
        )

    def getDataFilesForSamples(
        self, startSample, endSample
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

    def getPhysicalSamples(self, **kwargs):
        """Get data scaled to physical values
        
        resistics uses field units, meaning physical samples will return the following:

        - Electrical channels in mV/km
        - Magnetic channels in mV
        - To get magnetic fields in nT, calibration needs to be performed

        Notes
        -----
        The method getUnscaledSamples multiplies the raw data by the ts_lsb converting it to mV. Because gain is removed when getting the unscaledSamples and all channel data is in mV, the only calculation that has to be done is to divide by the dipole lengths (east-west spacing and north-south spacing).
        
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
        # get data
        timeData = self.getUnscaledSamples(
            chans=options["chans"],
            startSample=options["startSample"],
            endSample=options["endSample"],
        )
        # Ais applied in getUnscaledSamples to convert to mV - this is for ease of calculation and because each data file in the run might have a separate scaling
        # so all that is left is to divide by the dipole length in km and remove the average
        for chan in options["chans"]:
            if chan == "Ex":
                # multiply by 1000/self.getChanDx same as dividing by dist in km
                timeData.data[chan] = 1000 * timeData.data[chan] / self.getChanDx(chan)
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

    def spamHeaders(self) -> Tuple[List[str], Dict[str, str]]:
        """Get the sections in SPAM header files (XTR and XTRX)

        Returns
        -------
        sections : List[str]
            The sections in the header files
        sectionHeaders : Dict[str, str]
            The headers in each section to be read in 
        """

        sections = ["STATUS", "TITLE", "PROJECT", "FILE", "SITE", "CHANNAME", "DATA"]
        sectionHeaders = {}
        sectionHeaders["STATUS"] = ["STATUS"]
        sectionHeaders["TITLE"] = ["AUTHOR", "VERSION", "DATE", "COMMENT"]
        sectionHeaders["FILE"] = ["NAME", "FREQBAND", "DATE"]
        sectionHeaders["CHANNAME"] = ["ITEMS", "NAME"]
        sectionHeaders["DATA"] = ["ITEMS", "CHAN"]
        return sections, sectionHeaders

    def chanDefaults(self) -> Dict[str, Any]:
        """Get defaults for channel headers

        Returns
        -------
        Dict[str, Any]
            Dictionary of headers for channels and default values
        """

        chanH = {}
        chanH["gain_stage1"] = 1
        chanH["gain_stage2"] = 1
        chanH["hchopper"] = 0  # this depends on sample frequency
        chanH["echopper"] = 0
        # channel output information (sensor_type, channel_type, ts_lsb, pos_x1, pos_x2, pos_y1, pos_y2, pos_z1, pos_z2, sensor_sernum)
        chanH["ats_data_file"] = ""
        chanH["num_samples"] = 0
        chanH["sensor_type"] = ""
        chanH["channel_type"] = ""
        chanH["ts_lsb"] = 1
        # the lsb/scaling is not applied. data is raw voltage which needs to be scaled
        # an lsb is constructed from the scaling in the XTR/XTRX file to take the data to mV
        chanH["scaling_applied"] = False  # check this
        chanH["pos_x1"] = 0
        chanH["pos_x2"] = 0
        chanH["pos_y1"] = 0
        chanH["pos_y2"] = 0
        chanH["pos_z1"] = 0
        chanH["pos_z2"] = 0
        chanH["sensor_sernum"] = 0
        return chanH

    def readHeader(self) -> None:
        """Read header files

        For SPAM data, the may be more than one header file as data can be split up into smaller files as it is recorded. In that case, the header information should be somehow merged.
    
        All sampling frequencies should be the same
        """

        # read header files
        self.headersList = []
        self.chanHeadersList = []
        for headerFile in self.headerF:
            if "xtrx" in headerFile.lower():
                headers, chanHeaders = self.readHeaderXTRX(headerFile)
            else:
                headers, chanHeaders = self.readHeaderXTR(headerFile)
            self.headersList.append(headers)
            self.chanHeadersList.append(chanHeaders)

        # check to make sure no gaps
        # calculate out the sample ranges
        # and list the data files for each sample
        self.mergeHeaders(self.headersList, self.chanHeadersList)

    def readHeaderXTR(self, headerFile: str) -> None:
        """Read a XTR header file

        The raw data for SPAM is in single precision Volts. However, if there are multiple data files for a single recording, each one may have a different gain. Therefore, a scaling has to be calculated for each data file and channel. This scaling will convert all channels to mV. 

        For the most part, this method only reads recording information. However, it does additionally calculate out the lsb scaling and store it in the ts_lsb channel header. More information is provided in the notes.

        Notes
        -----
        The raw data for SPAM is in single precision floats and record the raw Voltage measurements of the sensors. However, if there are multiple data files for a single continuous recording, each one may have a different gain. Therefore, a scaling has to be calculated for each data file. 

        For electric channels, the scaling begins with the scaling provided in the header file in the DATA section. This incorporates any gain occuring in the device. This scaling is further amended by a conversion to mV and polarity reversal,

        .. code-block:: text
        
            scaling = read scaling from DATA section of header file
            scaling = 1000 * scaling , 
            scaling = -1000 * scaling ,
            ts_lsb = scaling ,
        
        where the reason for the 1000 factor in line 2 is not clear, nor is the polarity reversal. However, this information was provided by people more familiar with the data format.
        
        For magnetic channels, the scaling in the header file DATA section is ignored. This is because it includes a static gain correction, which would be duplicated at the calibration stage. Therefore, this is not included at this point.

        .. code-block:: text 
        
            scaling = -1000 ,
            ts_lsb = scaling ,
        
        This scaling converts the magnetic data from V to mV.

        Parameters
        ----------
        headerFile : str
            The XTR header file to read in
        """

        with open(headerFile, "r") as f:
            lines = f.readlines()
        sectionLines = {}
        # let's get data
        for line in lines:
            line = line.strip()
            line = line.replace("'", " ")
            # continue if line is empty
            if line == "":
                continue
            if "[" in line:
                sec = line[1:-1]
                sectionLines[sec] = []
            else:
                sectionLines[sec].append(line)
        # the base class is built around a set of headers based on ATS headers
        # though this is a bit more work here, it saves lots of code repetition
        headers = {}
        # recording information (start_time, start_date, stop_time, stop_date, ats_data_file)
        fileLine = sectionLines["FILE"][0]
        fileSplit = fileLine.split()
        headers["sample_freq"] = np.absolute(float(fileSplit[-1]))
        timeLine = sectionLines["FILE"][2]
        timeSplit = timeLine.split()
        # these are the unix time stamps
        startDate = float(timeSplit[1] + "." + timeSplit[2])
        datetimeStart = datetime.utcfromtimestamp(startDate)
        stopDate = float(timeSplit[3] + "." + timeSplit[4])
        datetimeStop = datetime.utcfromtimestamp(stopDate)
        headers["start_date"] = datetimeStart.strftime("%Y-%m-%d")
        headers["start_time"] = datetimeStart.strftime("%H:%M:%S.%f")
        headers["stop_date"] = datetimeStop.strftime("%Y-%m-%d")
        headers["stop_time"] = datetimeStop.strftime("%H:%M:%S.%f")
        # here calculate number of samples
        deltaSeconds = (datetimeStop - datetimeStart).total_seconds()
        # calculate number of samples - have to add one because the time given in SPAM recording is the actual time of the last sample
        numSamples = int(deltaSeconds * headers["sample_freq"]) + 1
        # put these in headers for ease of future calculations in merge headers
        headers["num_samples"] = numSamples
        # spam datasets only have the one data file for all channels
        headers["ats_data_file"] = fileSplit[1]
        # data information (meas_channels, sample_freq)
        chanLine = sectionLines["CHANNAME"][0]
        # this gets reformatted to an int later
        headers["meas_channels"] = chanLine.split()[1]
        numChansInt = int(headers["meas_channels"])
        # deal with the channel headers
        chanHeaders = []
        for iChan in range(0, numChansInt):
            chanH = self.chanDefaults()
            # set the sample frequency from the main headers
            chanH["sample_freq"] = headers["sample_freq"]
            # line data - read through the data in the correct channel order
            chanLine = sectionLines["CHANNAME"][iChan + 1]
            chanSplit = chanLine.split()
            dataLine = sectionLines["DATA"][iChan + 1]
            dataSplit = dataLine.split()
            # channel input information (gain_stage1, gain_stage2, hchopper, echopper)
            chanH["gain_stage1"] = 1
            chanH["gain_stage2"] = 1
            # channel output information (sensor_type, channel_type, ts_lsb, pos_x1, pos_x2, pos_y1, pos_y2, pos_z1, pos_z2, sensor_sernum)
            chanH["ats_data_file"] = fileSplit[1]
            chanH["num_samples"] = numSamples

            # channel information
            # spams often use Bx, By - use H within the software as a whole
            chanH["channel_type"] = consistentChans(chanSplit[2])
            # the sensor number is a bit of a hack - want MFSXXe or something - add MFS in front of the sensor number - this is liable to break
            # at the same time, set the chopper
            calLine = sectionLines["200{}003".format(iChan + 1)][0]
            calSplit = calLine.split()
            if isMagnetic(chanH["channel_type"]):
                chanH["sensor_sernum"] = calSplit[
                    2
                ]  # the last three digits is the serial number
                sensorType = calSplit[1].split("_")[1][-2:]
                chanH["sensor_type"] = "MFS{:02d}".format(int(sensorType))
                if "LF" in calSplit[1]:
                    chanH["hchopper"] = 1
            else:
                chanH["sensor_type"] = "ELC00"
                if "LF" in calLine:
                    chanH["echopper"] = 1

            # data is raw voltage of sensors
            # both E and H fields need polarity reversal (from email with Reinhard)
            # get scaling from headers
            scaling = float(dataSplit[-2])
            if isElectric(chanH["channel_type"]):
                # the factor of 1000 is not entirely clear
                lsb = 1000.0 * scaling
                # volts to millivolts and a minus to switch polarity giving data in mV
                lsb = -1000.0 * lsb
            else:
                # volts to millivolts and a minus to switch polarity giving data in mV
                # scaling in header file is ignored because it duplicates static gain correction in calibration
                lsb = -1000.0
            chanH["ts_lsb"] = lsb

            # the distances
            if chanSplit[2] == "Ex":
                chanH["pos_x1"] = float(dataSplit[4]) / 2
                chanH["pos_x2"] = chanH["pos_x1"]
            if chanSplit[2] == "Ey":
                chanH["pos_y1"] = float(dataSplit[4]) / 2
                chanH["pos_y2"] = chanH["pos_y1"]
            if chanSplit[2] == "Ez":
                chanH["pos_z1"] = float(dataSplit[4]) / 2
                chanH["pos_z2"] = chanH["pos_z1"]

            # append chanHeaders to the list
            chanHeaders.append(chanH)

        # check information from raw file headers
        self.headersFromRawFile(headers["ats_data_file"], headers)
        # return the headers and chanHeaders from this file
        return headers, chanHeaders

    def readHeaderXTRX(self, headerFile):
        """Read a XTRX header files

        XTRX are newer header files and will supercede XTR

        Parameters
        ----------
        headerFile : str
            The XTRX header file to read in
        """

        raise NotImplementedError("Support for XTRX files has not yet been implemented")

    def headersFromRawFile(self, rawFile: str, headers: Dict) -> None:
        """Read headers from the raw data files
        
        Read the headers from the raw file and figure out the data byte offset.     

        Parameters
        ----------
        rawFile : str
            The .RAW data file
        headers : Dict
            A headers dictionary

        Notes
        -----
        Open with encoding ISO-8859-1 because it has a value for all bytes unlike other encoding. In particular, want to find number of samples and the size of the header. The extended header is ignored.
        """

        dFile = open(os.path.join(self.dataPath, rawFile), "r", encoding="ISO-8859-1")
        generalHeaderString = dFile.read(1000)  # this should be long enough
        generalSplit = generalHeaderString.split()
        # read GENERAL HEADER
        generalHeader = {}
        generalHeader["recLength"] = int(generalSplit[0])
        generalHeader["fileType"] = generalSplit[1]
        generalHeader["wordLength"] = int(generalSplit[2])
        generalHeader["version"] = generalSplit[3]
        generalHeader["procId"] = generalSplit[4]
        generalHeader["numCh"] = int(generalSplit[5])
        generalHeader["totalRec"] = int(generalSplit[6])
        generalHeader["firstEvent"] = int(generalSplit[7])
        generalHeader["numEvent"] = int(generalSplit[8])
        generalHeader["extend"] = int(generalSplit[9])

        # read EVENT HEADER - there can be multiple of these, but normally only the one
        # Multiple events are largely deprecated. Only a single event is used
        eventHeaders = []
        fileSize = os.path.getsize(os.path.join(self.dataPath, rawFile))
        record = generalHeader["firstEvent"]
        for ir in range(0, generalHeader["numEvent"]):
            seekPt = (record - 1) * generalHeader["recLength"]
            if not seekPt > fileSize:
                # seek from beginning of file
                dFile.seek(seekPt, 0)
                # read extra to make sure
                eventString = dFile.read(1000)
                eventSplit = eventString.split()
                eH = {}
                eH["start"] = int(eventSplit[0])
                eH["startms"] = int(eventSplit[1])
                eH["stop"] = int(eventSplit[2])
                eH["stopms"] = int(eventSplit[3])
                eH["cvalue1"] = float(eventSplit[4])
                eH["cvalue2"] = float(eventSplit[5])
                eH["cvalue3"] = float(eventSplit[6])
                eH["EHInfile"] = int(eventSplit[7])
                eH["nextEH"] = int(eventSplit[8])
                eH["previousEH"] = int(eventSplit[9])
                eH["numData"] = int(eventSplit[10])
                eH["startData"] = int(eventSplit[11])
                eH["extended"] = int(eventSplit[12])
                eventHeaders.append(eH)
                if eH["nextEH"] < generalHeader["totalRec"]:
                    record = eH["nextEH"]  # set to go to next eH
                else:
                    break  # otherwise break out of for loops
        # close the data file
        dFile.close()
        # now compare number of samples with that calculated previously
        if eventHeaders[0]["numData"] != headers["num_samples"]:
            self.printWarning("Data file: {}".format(dFile))
            self.printWarning(
                "Number of samples in raw file header {} does not equal that calculated from data {}".format(
                    eventHeaders[0]["numData"], headers["num_samples"]
                )
            )
            self.printWarning("Number of samples calculated from data will be used")
        # set the byte offset for the file
        self.dataByteOffset[rawFile] = (
            eventHeaders[0]["startData"] - 1
        ) * generalHeader["recLength"]
        self.recChannels[rawFile] = generalHeader["numCh"]

    def mergeHeaders(self, headersList: List, chanHeadersList: List) -> None:
        """Merge headers from all the header files

        Checks all the header files to see if there are any gaps and calculates the sample ranges for each file together with the total number of samples. Sets the start and end time of the recording and class variables datetimeStart and datetimeStop.        

        Parameters
        ----------
        headersList : List
            List of headers from each data file
        chanHeadersList :
            List of chan headers from each data file
        """

        # take the first header as an example
        self.headers = headersList[0]
        self.chanHeaders = chanHeadersList[0]
        if len(headersList) == 1:
            # just fill in the data file list and data ranges
            self.dataFileList = [self.headers["ats_data_file"]]
            self.dataRanges = [[0, self.headers["num_samples"] - 1]]
            self.scalings = []
            tmp = {}
            for cHeader in self.chanHeaders:
                tmp[cHeader["channel_type"]] = cHeader["ts_lsb"]
            self.scalings.append(tmp)
            return  # then there was only one file - no need to do all the below

        # make sure that all headers have the same sample rate
        # and save the start and stop times and dates
        startTimes = []
        stopTimes = []
        numSamples = []
        for idx, header in enumerate(headersList):
            if header["sample_freq"] != self.headers["sample_freq"]:
                self.printWarning(
                    "Not all datasets in {} have the same sample frequency".format(
                        self.dataPath
                    )
                )
                self.printWarning("Exiting")
                exit()
            if header["meas_channels"] != self.headers["meas_channels"]:
                self.printWarning(
                    "Not all datasets in {} have the same number of channels".format(
                        self.dataPath
                    )
                )
                self.printWarning("Exiting")
                exit()
            # now store startTimes, stopTimes and numSamples
            # do this as datetimes, will be easier
            startString = "{} {}".format(header["start_date"], header["start_time"])
            stopString = "{} {}".format(header["stop_date"], header["stop_time"])
            datetimeStart = datetime.strptime(startString, "%Y-%m-%d %H:%M:%S.%f")
            datetimeStop = datetime.strptime(stopString, "%Y-%m-%d %H:%M:%S.%f")
            startTimes.append(datetimeStart)
            stopTimes.append(datetimeStop)
            numSamples.append(header["num_samples"])
        # check the start and end times
        sampleTime = timedelta(seconds=1.0 / self.headers["sample_freq"])
        # sort by start times
        sortIndices = sorted(list(range(len(startTimes))), key=lambda k: startTimes[k])
        # now sort stop times by the same indices
        check = True
        for i in range(1, self.numHeaderFiles):
            # get the stop time of the previous dataset
            stopTimePrev = stopTimes[sortIndices[i - 1]]
            startTimeNow = startTimes[sortIndices[i]]
            if startTimeNow != stopTimePrev + sampleTime:
                self.printWarning(
                    "There is a gap between the datafiles in {}".format(self.dataPath)
                )
                self.printWarning(
                    "Please separate out datasets with gaps into separate folders"
                )
                # print out where the gap was found
                self.printWarning("Gap found between datafiles:")
                self.printWarning(
                    "1. {}".format(headersList[sortIndices[i - 1]]["ats_data_file"])
                )
                self.printWarning(
                    "2. {}".format(headersList[sortIndices[i]]["ats_data_file"])
                )
                # set check as false
                check = False
        # if did not pass check, then exit
        if not check:
            exit()

        # make sure there are no gaps
        totalSamples = sum(numSamples)

        # get a list of all the datafiles, scalings and the sample ranges
        self.dataFileList = []
        self.dataRanges = []
        self.scalings = []
        sample = -1
        # now need some sort of lookup table to say where the sample ranges are
        for i in range(0, self.numHeaderFiles):
            iSort = sortIndices[i]  # get the sorted index
            self.dataFileList.append(headersList[iSort]["ats_data_file"])
            startSample = sample + 1
            endSample = (
                startSample + numSamples[iSort] - 1
            )  # -1 because this is inclusive of the start sample
            self.dataRanges.append([startSample, endSample])
            # increment sample
            sample = endSample
            # save the scalings for each chan
            tmp = {}
            for cHeader in self.chanHeadersList[iSort]:
                tmp[cHeader["channel_type"]] = cHeader["ts_lsb"]
            self.scalings.append(tmp)

        # now set the LSB information for the chanHeaders
        # i.e. if they change, this should reflect that
        for i in range(0, len(self.chanHeaders)):
            chan = self.chanHeaders[i]["channel_type"]
            lsbSet = set()
            for scalar in self.scalings:
                lsbSet.add(scalar[chan])
            if len(lsbSet) == 1:
                self.chanHeaders[i]["ts_lsb"] = list(lsbSet)[0]
            else:
                self.printWarning(
                    "Multiple different LSB values found for chan {}: {}".format(
                        chan, list(lsbSet)
                    )
                )
                self.printWarning(
                    "This is handled, but the header information given will show only a single LSB value"
                )
                self.chanHeaders[i]["ts_lsb"] = list(lsbSet)[0]

        # set start and end time for headers and chan headers
        # do the same with number of samples
        datetimeStart = min(startTimes)
        datetimeStop = max(stopTimes)
        self.headers["start_date"] = datetimeStart.strftime("%Y-%m-%d")
        self.headers["start_time"] = datetimeStart.strftime("%H:%M:%S.%f")
        self.headers["stop_date"] = datetimeStop.strftime("%Y-%m-%d")
        self.headers["stop_time"] = datetimeStop.strftime("%H:%M:%S.%f")
        self.headers["num_samples"] = totalSamples
        # set datafiles = the whole list of datafiles
        self.headers["ats_data_file"] = self.dataFileList
        for iChan in range(0, len(self.chanHeaders)):
            self.chanHeaders[iChan]["start_date"] = datetimeStart.strftime("%Y-%m-%d")
            self.chanHeaders[iChan]["start_time"] = datetimeStart.strftime(
                "%H:%M:%S.%f"
            )
            self.chanHeaders[iChan]["stop_date"] = datetimeStop.strftime("%Y-%m-%d")
            self.chanHeaders[iChan]["stop_time"] = datetimeStop.strftime("%H:%M:%S.%f")
            self.chanHeaders[iChan]["num_samples"] = totalSamples
            self.chanHeaders[iChan]["ats_data_file"] = self.dataFileList

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

