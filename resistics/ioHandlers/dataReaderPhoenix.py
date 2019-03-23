import os
import glob
import re, struct
import collections
import copy
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Tuple

# import from package
from resistics.ioHandlers.dataReader import DataReader
from resistics.ioHandlers.dataWriterInternal import DataWriterInternal
from resistics.dataObjects.timeData import TimeData
from resistics.utilities.utilsChecks import consistentChans, isMagnetic, isElectric
from resistics.utilities.utilsClean import (
    removeZeros,
    removeZerosSingle,
    removeNansSingle,
)
from resistics.utilities.utilsMath import intdiv
from resistics.utilities.utilsPrint import blockPrint


class DataReaderPhoenix(DataReader):
    """Data reader for Phoenix data

    The Phoenix data and recording format is different and does not nicely fit with the way resistics tries to model data.
    
    There are three frequencies recorded concurrently (e.g. 2400Hz, 150Hz, 15Hz). The lowest sampling frequency is continuous whilst the others record data files at regular intervals. There is no issue with the continous sampling frequency. 
    
    However, as resistics separates out data into continuous recordings, the consistent gaps for the higher frequencies will lead to lots of small data folders if converted to internal data format.

    This class returns the lowest frequency recording (the continuous one) when time series data is requested. However, higher frequencies can be converted to the internal data format using the methods available here.

    Warnings
    --------
    The appropriate scaling for Phoenix data to return field units has not yet been verified.

    It is not actually recommended to reformat the high frequency recordings as this will lead to potentially thousands of data folders. There is currently no straight-forward way to support the high-frequency Phoenix recordings.

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
    setParameters()
        Set data reader parameters for Phoenix files
    getSamplesRatesTS()
        Get the sampling frequencies of the time series data
    getNumberSamplesTS()
        Get the number of samples for each time series file
    getUnscaledSamples(**kwargs)
        Get raw data from data file
    getRecordsForSamples(startSample, endSample)
        Get the records to read for a sample range
    readTag(dataFile)
        Read the tag from a data file
    readRecord(dataFile, numChans, numScans)
        Read numScans from a record
    twosComplement(dataBytes)
        Read the two's complement data from the file
    getPhysicalSamples(**kwargs)
        Get data scaled to physical values
    chanDefaults()
        Get defaults for channel headers
    readHeader()
        Read header file
    readTable()
        Read table file
    removeControl(inBytes)
        Remove control characters from a byte string
    headersFromTable(tableData)
        Parse the information in the table file to get headers
    getDates(tableData)
        Get recording dates (start and end time)
    checkSamples()
        Check the number of samples for all the timeseries (ts) files
    reformatHigh(path, **kwargs)
        Write out high frequency time series in internal format
    reformatContinuous(path)
        Write out the continuous time series in internal format
    reformat(path)
        Write out all recorded time series to internal format
    printDataFileList()  
        Information about the data files as a list of strings
    printDataFileInfo()
        Print a list of the data files
    printTableFileList()
        Information about the table file as a list of strings
    printTableFileInfo()
        Print table file info

    Notes
    -----
    Phoenix data is stored in 3 bytes two's-complement format.
    """

    def setParameters(self) -> None:
        """Set data reader parameters for Phoenix files
        
        Phoenix time series data is not contiguous in the file and is separated into records. There are multiple time series data files, one for the continuous recording and two others for the other frequencies. Therefore, there are a few other class variables defined here than in the parent DataReader class.
        """

        # get a list of the header and data files in the folder
        self.headerF = glob.glob(os.path.join(self.dataPath, "*.TBL"))
        self.dataF = glob.glob(os.path.join(self.dataPath, "*.TS*"))
        # set the sample byte size
        self.sampleByteSize = 3  # two's complement
        self.tagByteSize = 32
        self.dtype = int
        # there will be multiple TS files in here
        # need to figure out
        self.numHeaderFiles = len(self.headerF)
        self.numDataFiles = len(self.dataF)

    def getSamplesRatesTS(self) -> Dict:
        """Get the sampling frequencies of the time series data

        Returns
        -------
        Dict
            Dictionary with the time series file number as keys and their sampling frequencies in Hz as values
        """

        info: Dict = {}
        for num, sr in zip(self.tsNums, self.tsSampleFreqs):
            info[num] = sr
        return info

    def getNumberSamplesTS(self) -> Dict:
        """Get the number of samples for each time series file

        Returns
        -------
        Dict
            Dictionary with the time series file number as keys and their number of samples as values
        """

        info = {}
        for num, ns in zip(self.tsNums, self.tsNumSamples):
            info[num] = ns
        return info

    def getUnscaledSamples(self, **kwargs) -> TimeData:
        """Get raw data from data file

        Only returns the continuous data. The continuous data is in 24 bit two's complement (3 bytes) format and is read in using struct as this is not supported by numpy.
        
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
        recordsToRead, samplesToRead = self.getRecordsForSamples(
            options["startSample"], options["endSample"]
        )
        numSamples = options["endSample"] - options["startSample"] + 1
        # set up the dictionary to hold the data
        data = {}
        for chan in options["chans"]:
            data[chan] = np.zeros(shape=(numSamples), dtype=self.dtype)

        # open the file
        dFile = open(self.continuousF, "rb")

        # loop through chans and get data
        sampleCounter = 0
        for record, sToRead in zip(recordsToRead, samplesToRead):
            # number of samples to read in record
            dSamples = sToRead[1] - sToRead[0] + 1
            # find the byte read start and byte read end
            recordByteStart = self.recordBytes[self.continuous][record]
            recordSampleStart = self.recordSampleStarts[self.continuous][record]
            # find the offset on the readFrom bytes
            # now recall, each sample is recorded as a scan (all channels recorded at the same time)
            # so multiply by number of channels to get the number of bytes to read
            byteReadStart = (
                recordByteStart
                + (sToRead[0] - recordSampleStart)
                * self.sampleByteSize
                * self.getNumChannels()
            )
            bytesToRead = dSamples * self.sampleByteSize * self.getNumChannels()
            # read the data - numpy does not support 24 bit two's complement (3 bytes) - hence use struct
            dFile.seek(byteReadStart, 0)  # seek to start byte from start of file
            dataBytes = dFile.read(bytesToRead)
            dataRead = self.twosComplement(dataBytes)
            # now need to unpack this
            for chan in options["chans"]:
                # check to make sure channel exists
                self.checkChan(chan)
                # get the channel index - the chanIndex should give the right order in the data file
                # as it is the same order as in the header file
                chanIndex = self.chanMap[chan]
                # now populate the channel data appropriately
                data[chan][sampleCounter : sampleCounter + dSamples] = dataRead[
                    chanIndex : dSamples * self.getNumChannels() : self.getNumChannels()
                ]
            # increment sample counter
            sampleCounter = sampleCounter + dSamples  # get ready for the next data read
        # close file
        dFile.close()

        # return data
        startTime, stopTime = self.sample2time(
            options["startSample"], options["endSample"]
        )
        comment = "Unscaled data {} to {} read in from measurement {}, samples {} to {}".format(
            startTime,
            stopTime,
            self.dataPath,
            options["startSample"],
            options["endSample"],
        )
        return TimeData(
            sampleFreq=self.getSampleFreq(),
            startTime=startTime,
            stopTime=stopTime,
            data=data,
            comments=comment,
        )

    def getRecordsForSamples(
        self, startSample: int, endSample: int
    ) -> Tuple[List, List]:
        """Get the records to read for a sample range

        Parameters
        ----------
        startSample : int
            The starting sample of the range
        endSample : int
            The ending sample of the range
        
        Returns
        -------
        recordsToRead : List
            The records to read from the time series data files
        samplesToRead : List
            The samples to read from each record
        """

        recordsToRead = []
        samplesToRead = []
        for record, timeStart in enumerate(self.recordStarts[self.continuous]):
            recordStartSamp = self.recordSampleStarts[self.continuous][record]
            recordEndSamp = self.recordSampleStops[self.continuous][record]
            if recordStartSamp > endSample or recordEndSamp < startSample:
                continue  # nothing to read from this file
            # in this case, there is some overlap with the samples to read
            recordsToRead.append(record)
            readFrom = recordStartSamp  # i.e. the first sample in the datafile
            readTo = recordEndSamp  # this the last sample in the file
            if recordStartSamp < startSample:
                readFrom = startSample
            if recordEndSamp > endSample:
                readTo = endSample
            # this is an inclusive number readFrom to readTo including readTo
            samplesToRead.append([readFrom, readTo])
        return recordsToRead, samplesToRead

    def readTag(self, dataFile) -> Tuple[str]:
        """Read the tag from a data file

        Parameters
        ----------
        dataFile : file handle
            File handle of the data file
        
        Returns
        -------
        numScans : List
            Number of scans in the tag
        numChans : List
            Number of channels in the tag
        dateString : str
            The dataString of the tag
        """

        second = struct.unpack("b", dataFile.read(1))[0]
        minute = struct.unpack("b", dataFile.read(1))[0]
        hour = struct.unpack("b", dataFile.read(1))[0]
        day = struct.unpack("b", dataFile.read(1))[0]
        month = struct.unpack("b", dataFile.read(1))[0]
        year = struct.unpack("b", dataFile.read(1))[0]
        dayOfWeek = struct.unpack("b", dataFile.read(1))[0]
        century = struct.unpack("b", dataFile.read(1))[0]
        dateString = "{:02d}{:02d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}.000".format(
            century, year, month, day, hour, minute, second
        )
        # serial number
        serialNum = struct.unpack("h", dataFile.read(2))
        # num scans
        numScans = struct.unpack("h", dataFile.read(2))[0]
        # channels per scan
        numChans = struct.unpack("b", dataFile.read(1))[0]
        # tag length
        tagLength = struct.unpack("b", dataFile.read(1))
        # status code
        statusCode = struct.unpack("b", dataFile.read(1))
        # bit-wise saturation flags
        saturationFlag = struct.unpack("b", dataFile.read(1))
        # reserved
        reserved = struct.unpack("b", dataFile.read(1))
        # sample length
        sampleLength = struct.unpack("b", dataFile.read(1))
        # sample rate
        sampleRate = struct.unpack("h", dataFile.read(2))
        # units of sample rate: 0 = Hz, 1 = minute, 2 = hour, 3 = day
        sampleUnits = struct.unpack("b", dataFile.read(1))
        # clock status
        clockStatus = struct.unpack("b", dataFile.read(1))
        # clock error in micro seconds
        clockError = struct.unpack("i", dataFile.read(4))
        # reserved
        res1 = struct.unpack("b", dataFile.read(1))
        res2 = struct.unpack("b", dataFile.read(1))
        res3 = struct.unpack("b", dataFile.read(1))
        res4 = struct.unpack("b", dataFile.read(1))
        res5 = struct.unpack("b", dataFile.read(1))
        res6 = struct.unpack("b", dataFile.read(1))
        # returnt the important variables
        return numScans, numChans, dateString

    def readRecord(self, dataFile, numChans, numScans):
        """Read numScans from a record

        Parameters
        ----------
        dataFile : file handle
            File handle of the data file
        numScans : List
            Number of scans in the tag
        numChans : List
            Number of channels in the tag

        Returns
        -------
        data : np.ndarray(int)
            Record data
        """

        data = np.zeros(shape=(numChans, numScans), dtype="int")
        for scan in range(0, numScans):
            for chan in range(0, numChans):
                dataBytes = dataFile.read(3)
                data[chan, scan] = self.twosComplement(dataBytes)
        return data

    def twosComplement(self, dataBytes):
        """Read the two's complement data from the file

        This parses two's complement 24-bit integer, little endian, unsigned and signed. The method is to pad out 3 bytes out with a null byte and read as unsigned integer with little endian (<).        

        Parameters
        ----------
        dataByes : bytes
            The bytes to parse

        Returns
        -------
        data : np.ndarray(int)
            Record data
        """

        if len(dataBytes) % self.sampleByteSize != 0:
            self.printError(
                "The number of bytes divided by the sample byte size does not give an exact number",
                quitRun=True,
            )
        # calculate num samples, this should be exact
        numSamples = intdiv(len(dataBytes), self.sampleByteSize)
        dataRead = np.zeros(shape=(numSamples), dtype=self.dtype)
        for i in range(0, numSamples):
            sampleBytes = dataBytes[
                i * self.sampleByteSize : (i + 1) * self.sampleByteSize
            ]
            unsigned = struct.unpack("<I", sampleBytes + b"\x00")[0]
            signed = unsigned if not (unsigned & 0x800000) else unsigned - 0x1000000
            dataRead[i] = signed
        return dataRead

    def getPhysicalSamples(self, **kwargs) -> TimeData:
        """Get data scaled to physical values

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
        # get data
        timeData = self.getUnscaledSamples(
            chans=options["chans"],
            startSample=options["startSample"],
            endSample=options["endSample"],
        )
        # need to remove the gain
        for chan in options["chans"]:
            # remove the gain
            timeData.data[chan] = 1.0 * timeData.data[chan] / self.getChanGain1(chan)
            timeData.addComment(
                "Scaling channel {} with scalar {} to give mV".format(
                    chan, 1.0 / self.getChanGain1(chan)
                )
            )

            # divide by distance in km
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
            "The required Phoneix scaling to field units is still unverified. This is experimental and use cautiously."
        )
        timeData.addComment(
            "Remove zeros: {}, remove nans: {}, remove average: {}".format(
                options["remzeros"], options["remnans"], options["remaverage"]
            )
        )
        return timeData

    def chanDefaults(self):
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
        chanH["scaling_applied"] = False
        chanH["pos_x1"] = 0
        chanH["pos_x2"] = 0
        chanH["pos_y1"] = 0
        chanH["pos_y2"] = 0
        chanH["pos_z1"] = 0
        chanH["pos_z2"] = 0
        chanH["sensor_sernum"] = 0
        return chanH

    def readHeader(self):
        """Read header file

        For phoenix data, the header file is the table file and it is binary formatted.
        """

        # first, find which ts files are available (2,3,4,5)
        # and the continuous recording frequency (the max)
        self.tsNums = []
        for tsfile in self.dataF:
            self.tsNums.append(int(tsfile[-1]))
        self.continuous = max(self.tsNums)
        self.continuousI = self.tsNums.index(self.continuous)
        self.continuousF = self.dataF[self.continuousI]
        # read the table data
        self.tableData = self.readTable()
        # and then populate the headers
        self.headers, self.chanHeaders = self.headersFromTable(self.tableData)
        # finally, check the number of samples in each file
        self.checkSamples()

    def readTable(self) -> Dict:
        """Read a header table

        Returns
        -------
        OrderedDict
            An ordered dictionary of header table data
        """

        if len(self.headerF) > 1:
            self.printWarning(
                "More table files than expected. Using: {}".format(self.headerF[0])
            )
        numBytes = os.path.getsize(self.headerF[0])
        tableFile = open(self.headerF[0], "rb")
        tableData = collections.OrderedDict()
        # loop through file and read
        bytesRead = 0
        headerWordSize = 4
        headerSize = 12
        dataSize = 13
        increment = headerSize + dataSize
        while bytesRead <= numBytes - increment:
            # formats for reading in
            # integers
            ints = [
                "SGIN",
                "EGNC",
                "HGNC",
                "EGN",
                "HGN",
                "ACDC",
                "ACDH",
                "V5SR",
                "MTSR",
                "LCHP",
                "L2NS",
                "L3NS",
                "L4NS",
                "DDAT",
                "TXPR",
                "TBVO",
                "TBVI",
                "INIT",
                "RQST",
                "MODE",
                "XDOS",
                "ATYP",
                "FNAM",
                "FLEN",
                "AQST",
                "HSMP",
                "CALS",
                "CCLS",
                "TEMP",
                "TMAX",
                "GFPG",
                "FFPG",
                "DSP",
                "CHEX",
                "CHEY",
                "CHHX",
                "CHHY",
                "CHHZ",
                "TCHN",
                "POTS",
                "NREF",
                "CCLT",
                "PZLT",
                "NSAT",
                "OCTR",
                "CLST",
                "TALS",
                "TCMB",
                "TERR",
                "LPFR",
                "LFRQ",
                "SNUM",
                "MXSC",
                "BADR",
                "NOBF",
                "SATR",
                "BAT1",
                "BAT2",
                "BAT3",
                "EXR",
                "EYR",
                "ELEV",
                "SRL2",
                "SRL3",
                "SRL4",
                "SRL5",
                "DISK",
                "STDE",
                "TOTL",
                "STDH",
            ]
            # UTC
            ints1_8 = [
                "TDSP",
                "LFIX",
                "TSYN",
                "STIM",
                "ETIM",
                "HTIM",
                "ETMH",
                "NUTC",
                "FTIM",
                "LTIM",
            ]
            # non-integer headers
            doubles = [
                "EXAC",
                "EXDC",
                "EYAC",
                "EYDC",
                "HXAC",
                "HXDC",
                "HYAC",
                "HYDC",
                "HZAC",
                "HZDC",
                "DXAC",
                "DXDC",
                "DYAC",
                "DYDC",
                "EXNR",
                "EXPR",
                "EYNR",
                "EYPR",
                "GNDR",
                "MAXR",
                "EAZM",
                "HAZM",
                "DECL",
                "TSTV",
                "FSCV",
                "CCMN",
                "CCMX",
                "HATT",
                "HAMP",
                "CPHC",
                "LFIX",
                "EXLN",
                "EYLN",
                "TSTR",
                "INPR",
                "CFMN",
                "CFMX",
                "HNOM",
            ]
            # get the header word
            header = struct.unpack(
                "{}s".format(headerWordSize), tableFile.read(headerWordSize)
            )
            header = self.removeControl(header[0])
            tableFile.seek(headerSize - headerWordSize, 1)
            if header == "":
                break  # get rid of empty lines at the end
            if header in ints:
                value = struct.unpack("i", tableFile.read(4))[0]
                tableFile.seek(dataSize - 4, 1)
            elif header in ints1_8:
                value = struct.unpack("8b", tableFile.read(8))
                tableFile.seek(dataSize - 8, 1)
            elif header in doubles:
                value = struct.unpack("d", tableFile.read(8))[0]
                tableFile.seek(dataSize - 8, 1)
            else:
                value = struct.unpack("{}s".format(dataSize), tableFile.read(dataSize))
                value = self.removeControl(value[0])
            tableData[header] = value
            # increment bytes read
            bytesRead += increment
        tableFile.close()
        return tableData

    def removeControl(self, inBytes: bytes) -> str:
        """Remove control characters from byte strings
        
        Parameters
        ----------
        inBytes : bytes
            Bytes from which to remove control 
        
        Returns
        -------
        str :
            Decodes bytes object with control character removed
        """

        inBytes = inBytes.strip(b"\x00")
        return inBytes.decode()

    def headersFromTable(self, tableData: Dict) -> Tuple[Dict, List]:
        """Populate the headers from the table values
        
        Parameters
        ----------
        tableData : OrederedDict
            Ordered dictionary with table data
        
        Returns
        -------
        headers : Dict
            Dictionary of general headers
        chanHeaders : Dict
            List of channel headers
        """

        # initialise storage
        headers = {}
        chanHeaders = []
        # get the sample freqs for each ts file
        self.tsSampleFreqs = []
        for tsNum in self.tsNums:
            self.tsSampleFreqs.append(tableData["SRL{}".format(tsNum)])
        # for sample frequency, use the continuous channel
        headers["sample_freq"] = self.tsSampleFreqs[self.continuousI]
        # these are the unix time stamps
        firstDate, firstTime, lastDate, lastTime = self.getDates(tableData)
        # the start date is equal to the time of the first record
        headers["start_date"] = firstDate
        headers["start_time"] = firstTime
        datetimeStart = datetime.strptime(
            "{} {}".format(firstDate, firstTime), "%Y-%m-%d %H:%M:%S.%f"
        )
        # the stop date
        datetimeLast = datetime.strptime(
            "{} {}".format(lastDate, lastTime), "%Y-%m-%d %H:%M:%S.%f"
        )
        # records are usually equal to one second (beginning on 0 and ending on the last sample before the next 0)
        datetimeStop = datetimeLast + timedelta(
            seconds=(1.0 - 1.0 / headers["sample_freq"])
        )
        # put the stop date and time in the headers
        headers["stop_date"] = datetimeStop.strftime("%Y-%m-%d")
        headers["stop_time"] = datetimeStop.strftime("%H:%M:%S.%f")
        # here calculate number of samples
        deltaSeconds = (datetimeStop - datetimeStart).total_seconds()
        # calculate number of samples - have to add one because the time given in SPAM recording is the actual time of the last sample
        numSamples = round(deltaSeconds * headers["sample_freq"]) + 1
        headers["num_samples"] = numSamples
        headers["ats_data_file"] = self.continuousF
        # deal with the channel headers
        # now want to do this in the correct order
        # chan headers should reflect the order in the data
        chans = ["Ex", "Ey", "Hx", "Hy", "Hz"]
        chanOrder = []
        for chan in chans:
            chanOrder.append(tableData["CH{}".format(chan.upper())])
        # sort the lists in the right order based on chanOrder
        chanOrder, chans = (
            list(x)
            for x in zip(*sorted(zip(chanOrder, chans), key=lambda pair: pair[0]))
        )
        for chan in chans:
            chanH = self.chanDefaults()
            # set the sample frequency from the main headers
            chanH["sample_freq"] = headers["sample_freq"]
            # channel output information (sensor_type, channel_type, ts_lsb, pos_x1, pos_x2, pos_y1, pos_y2, pos_z1, pos_z2, sensor_sernum)
            chanH["ats_data_file"] = self.dataF[self.continuousI]
            chanH["num_samples"] = numSamples
            # channel information
            chanH["channel_type"] = consistentChans(chan)  # consistent chan naming

            # magnetic channels only
            if isMagnetic(chanH["channel_type"]):
                chanH["sensor_sernum"] = tableData["{}SN".format(chan.upper())][-4:]
                chanH["sensor_type"] = "Phoenix"
                # channel input information (gain_stage1, gain_stage2, hchopper, echopper)
                chanH["gain_stage1"] = tableData["HGN"]
                chanH["gain_stage2"] = 1

            # electric channels only
            if isElectric(chanH["channel_type"]):
                # the distances
                if chan == "Ex":
                    chanH["pos_x1"] = float(tableData["EXLN"]) / 2.0
                    chanH["pos_x2"] = chanH["pos_x1"]
                if chan == "Ey":
                    chanH["pos_y1"] = float(tableData["EYLN"]) / 2.0
                    chanH["pos_y2"] = chanH["pos_y1"]
                # channel input information (gain_stage1, gain_stage2, hchopper, echopper)
                chanH["gain_stage1"] = tableData["EGN"]
                chanH["gain_stage2"] = 1

            # append chanHeaders to the list
            chanHeaders.append(chanH)

        # data information (meas_channels, sample_freq)
        headers["meas_channels"] = len(chans)  # this gets reformatted to an int later
        # return the headers and chanHeaders from this file
        return headers, chanHeaders

    def getDates(self, tableData) -> Tuple[str, str, str, str]:
        """Get recording dates (start and end time)
        
        Parameters
        ----------
        tableData : OrederedDict
            Ordered dictionary with table data
        
        Returns
        -------
        firstDate : str
            Date of first sample as string
        firstTime : str
            Time of first sample as string
        lastDate : str
            Date of last sample as string
        lastTime : str
            Time of last sample as string
        """

        firstSecond = tableData["FTIM"][0]
        firstMinute = tableData["FTIM"][1]
        firstHour = tableData["FTIM"][2]
        firstDay = tableData["FTIM"][3]
        firstMonth = tableData["FTIM"][4]
        firstYear = tableData["FTIM"][5]
        firstCentury = tableData["FTIM"][-1]
        firstDate = "{:02d}{:02d}-{:02d}-{:02d}".format(
            firstCentury, firstYear, firstMonth, firstDay
        )
        firstTime = "{:02d}:{:02d}:{:02d}.000".format(
            firstHour, firstMinute, firstSecond
        )
        # this is the start time of the last record
        lastSecond = tableData["LTIM"][0]
        lastMinute = tableData["LTIM"][1]
        lastHour = tableData["LTIM"][2]
        lastDay = tableData["LTIM"][3]
        lastMonth = tableData["LTIM"][4]
        lastYear = tableData["LTIM"][5]
        lastCentury = tableData["LTIM"][-1]
        lastDate = "{:02d}{:02d}-{:02d}-{:02d}".format(
            lastCentury, lastYear, lastMonth, lastDay
        )
        lastTime = "{:02d}:{:02d}:{:02d}.000".format(lastHour, lastMinute, lastSecond)
        return firstDate, firstTime, lastDate, lastTime

    def checkSamples(self) -> None:
        """Check the number of samples for all the timeseries (ts) files
        
        Recall, the format is 3 bytes two's complement per sample
        """

        self.recordStarts = {}
        self.recordScans = {}
        self.recordBytes = {}
        self.recordSampleStarts = {}
        self.recordSampleStops = {}
        # loop over the tsNums
        samplesDict = {}
        for dFileName in self.dataF:
            ts = int(dFileName[-1])
            self.recordStarts[ts] = []
            self.recordScans[ts] = []
            self.recordBytes[ts] = []
            self.recordSampleStarts[ts] = []
            self.recordSampleStops[ts] = []
            # start number of samples at 0
            samples = 0
            # get file size in samples
            numBytes = os.path.getsize(dFileName)
            bytesread = 0
            # now run through the file and figure out the number of samples
            dFile = open(dFileName, "rb")
            while bytesread < numBytes:
                # read 32 bytes tag
                numScans, numChans, dateString = self.readTag(dFile)
                self.recordBytes[ts].append(bytesread + self.tagByteSize)
                dataBytes = numScans * numChans * self.sampleByteSize
                dFile.seek(dataBytes, 1)
                bytesread += self.tagByteSize + dataBytes
                # save the record start times and scan lengths
                self.recordStarts[ts].append(dateString)
                self.recordScans[ts].append(numScans)
                # save the sample starts
                self.recordSampleStarts[ts].append(samples)
                # increment the number of samples
                # recall, a scan is all channels recorded at one time
                # this is equivalent to one sample
                samples += numScans  # this is the count
                # sample stop is samples -1 because inclusive of the current sample
                self.recordSampleStops[ts].append(samples - 1)
            dFile.close()
            # save number of samples in dict
            samplesDict[ts] = samples
            # logFile.close()

        self.tsNumSamples = []
        for tsNum in self.tsNums:
            self.tsNumSamples.append(samplesDict[tsNum])

        # check the samples of the continuous file
        if self.tsNumSamples[self.continuousI] != self.getNumSamples():
            self.printWarning(
                "Number of samples calculated from times is different to that in file"
            )
            self.printWarning(
                "{} samples in file, {} calculated from time".format(
                    self.tsNumSamples[self.continuousI], self.getNumSamples()
                )
            )

    def reformatHigh(self, path: str, **kwargs) -> None:
        """Write out high frequency time series in internal format
        
        Parameters
        ----------
        path : str
            Directory to write out the reformatted time series
        ts : List[int], optional
            A list of the high frequency ts files to reformat. By default, all of the higher frequency recordings are reformatted
        """

        writer = DataWriterInternal()
        for idx, ts in enumerate(self.tsNums):
            if "ts" in kwargs and ts not in kwargs["ts"]:
                continue  # do not reformat this one
            # let's get the headers
            headers = self.getHeaders()
            chanHeaders, chanMap = self.getChanHeaders()
            chans = self.getChannels()
            # now go through the different ts files to get ready to output
            if ts == self.continuous:
                continue
            sampleFreq = self.tsSampleFreqs[idx]
            # set sample frequency in headers
            headers["sample_freq"] = sampleFreq
            for cH in chanHeaders:
                cH["sample_freq"] = sampleFreq
            # now open the data file
            dFile = open(self.dataF[idx], "rb")
            # each record has to be read separately and then compare time to previous
            outStartTime = datetime.strptime(
                self.recordStarts[ts][0], "%Y-%m-%d %H:%M:%S.%f"
            )
            # set up the data dictionary
            data = {}
            for record, startDate in enumerate(self.recordStarts[ts]):
                # start date is a string
                startByte = self.recordBytes[ts][record]
                startDateTime = datetime.strptime(startDate, "%Y-%m-%d %H:%M:%S.%f")
                # read the record - numpy does not support 24 bit two's complement (3 bytes) - hence use struct
                bytesToRead = (
                    self.recordScans[ts][record]
                    * self.sampleByteSize
                    * self.getNumChannels()
                )
                dFile.seek(startByte, 0)  # seek to start byte from start of file
                dataBytes = dFile.read(bytesToRead)
                dataRead = self.twosComplement(dataBytes)
                dataRecord = {}
                for chan in chans:
                    # as it is the same order as in the header file
                    chanIndex = self.chanMap[chan]
                    dataRecord[chan] = dataRead[
                        chanIndex : self.recordScans[ts][record]
                        * self.getNumChannels() : self.getNumChannels()
                    ]
                # need to compare to previous record
                if record != 0 and startDateTime != prevEndTime:
                    # then need to write out the current data before saving the new data
                    # write out current data
                    outStopTime = prevEndTime - timedelta(
                        seconds=1.0 / sampleFreq
                    )  # because inclusive of first sample (previous end time for continuity comparison)
                    # calculate number of samples
                    numSamples = data[chans[0]].size
                    headers["start_date"] = outStartTime.strftime("%Y-%m-%d")
                    headers["start_time"] = outStartTime.strftime("%H:%M:%S.%f")
                    headers["stop_date"] = outStopTime.strftime("%Y-%m-%d")
                    headers["stop_time"] = outStopTime.strftime("%H:%M:%S.%f")
                    headers["num_samples"] = numSamples
                    for cH in chanHeaders:
                        cH["start_date"] = headers["start_date"]
                        cH["start_time"] = headers["start_time"]
                        cH["stop_date"] = headers["stop_date"]
                        cH["stop_time"] = headers["stop_time"]
                        cH["num_samples"] = numSamples
                    # get the outpath
                    dataOutpath = os.path.join(
                        path,
                        "meas_ts{}_{}_{}".format(
                            ts,
                            outStartTime.strftime("%Y-%m-%d-%H-%M-%S"),
                            outStopTime.strftime("%Y-%m-%d-%H-%M-%S"),
                        ),
                    )
                    # create the timeData object
                    comment = "Unscaled samples for interval {} to {} read in from measurement {}".format(
                        outStartTime, outStopTime, self.dataF[idx]
                    )
                    timeData = TimeData(
                        sampleFreq=self.getSampleFreq(),
                        startTime=outStartTime,
                        stopTime=outStopTime,
                        data=data,
                        comments=comment,
                    )
                    # write out
                    writer.setOutPath(dataOutpath)
                    writer.writeData(headers, chanHeaders, timeData)
                    # then save current data
                    outStartTime = startDateTime
                    data = copy.deepcopy(dataRecord)
                    prevEndTime = startDateTime + timedelta(
                        seconds=((1.0 / sampleFreq) * self.recordScans[ts][record])
                    )
                else:
                    # then record == 0 or startDateTime == prevEndTime
                    # update prevEndTime
                    prevEndTime = startDateTime + timedelta(
                        seconds=((1.0 / sampleFreq) * self.recordScans[ts][record])
                    )
                    if record == 0:
                        data = copy.deepcopy(dataRecord)
                        continue
                    # otherwise, want to concatenate the data
                    for chan in chans:
                        data[chan] = np.concatenate((data[chan], dataRecord[chan]))
            # close the data file
            dFile.close()

    def reformatContinuous(self, path: str):
        """Write out the continuous time series in internal format
        
        Parameters
        ----------
        path : str
            Path to write out reformatted continuous recording
        """

        writer = DataWriterInternal()
        outpath = "meas_ts{}_{}_{}".format(
            self.continuous,
            self.getStartDatetime().strftime("%Y-%m-%d-%H-%M-%S"),
            self.getStopDatetime().strftime("%Y-%m-%d-%H-%M-%S"),
        )
        outpath = os.path.join(path, outpath)
        writer.setOutPath(outpath)
        headers = self.getHeaders()
        chanHeaders, chanMap = self.getChanHeaders()
        writer.writeData(headers, chanHeaders, self.getPhysicalSamples(), physical=True)

    def reformat(self, path):
        """Write out all recorded time series to internal format
        
        Parameters
        ----------
        path : str
            Path to write out reformatted recordings
        """

        self.reformatContinuous(path)
        self.reformatHigh(path)

    def printDataFileList(self) -> List[str]:
        """Information about the data files as a list of strings
        
        Returns
        -------
        List[str]
            List of information about the data files
        """

        textLst = []
        textLst.append("TS File\t\tSampling frequency (Hz)\t\tNum Samples")
        for dF, tsF, tsN in zip(self.dataF, self.tsSampleFreqs, self.tsNumSamples):
            textLst.append("{}\t\t{}\t\t{}".format(os.path.basename(dF), tsF, tsN))
        textLst.append(
            "Continuous data file: {}".format(os.path.basename(self.continuousF))
        )
        return textLst

    def printDataFileInfo(self):
        """Print a list of the data files"""

        blockPrint(
            "{} Data File List".format(self.__class__.__name__),
            self.printDataFileList(),
        )

    def printTableFileList(self) -> List[str]:
        """Information about the table file as a list of strings
        
        Returns
        -------
        List[str]
            List of information about table file content
        """

        textLst = []
        for h, v in list(self.tableData.items()):
            textLst.append("{} = {}".format(h, v))
        return textLst

    def printTableFileInfo(self):
        """Print table file info"""

        blockPrint(
            "{} Table File Info".format(self.__class__.__name__),
            self.printTableFileList(),
        )
